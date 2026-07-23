"""Self-contained HTML report generator for Axetract benchmark results.

Reads one or more benchmark JSON files (as produced by
:mod:`benchmarks.run`) and emits a single, portable ``.html`` file with:

* Headline KPI cards.
* A sortable, color-graded summary table across all configs / runs.
* Inline-SVG charts (no external dependencies, works offline):
    - Latency distribution (p50 / p90 / p99) per config.
    - Throughput vs. batch size (docs/s and tokens/s).
    - Per-stage occupancy stacked bars.
    - Stage timeline (Gantt) from raw ``stage_events``.
    - Overlap efficiency vs. micro-batch size.
    - Resource usage (GPU util, VRAM, RSS).
    - Cold start vs. steady-state p50.
* Auto-generated, data-driven interpretation bullets under each chart.
* Per-config drill-down (collapsible).

Usage::

    # Single run
    python -m benchmarks.html_report benchmarks/results/bench_vllm_gpu_20260718_145208.json

    # Compare multiple runs (e.g. vllm vs hf, or across commits)
    python -m benchmarks.html_report results/*.json --out report.html

The generator uses only the Python standard library — no new dependencies.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("benchmarks.html_report")

# ── Brand palette ─────────────────────────────────────────────────────
# Anthropic-inspired: warm accent on a dark canvas.
BG = "#0f0e0d"          # page background
SURFACE = "#1a1816"      # cards / panels
SURFACE_2 = "#24201d"   # table rows (alt)
BORDER = "#3a342f"
TEXT = "#f5f1ec"
TEXT_DIM = "#a89e93"
ACCENT = "#f15822"      # primary brand accent
ACCENT_2 = "#d97757"    # muted secondary (Anthropic clay)
ACCENT_3 = "#c2976b"    # tertiary
GOOD = "#7fb069"        # success / "higher is better"
BAD = "#e07a5f"         # warning / "lower is better"
NEUTRAL = "#6b6258"

# Stage colors (consistent with the rest of the benchmark suite).
STAGE_COLORS = {
    "preprocess": "#d97757",
    "prune": "#f15822",
    "setup": "#8a6f5c",      # GC + torch.cuda.empty_cache() between stages
    "extract": "#c2976b",
    "postprocess": "#7fb069",
}
STAGE_ORDER = ("preprocess", "prune", "setup", "extract", "postprocess")


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _fmt(value: Any, precision: int = 2, unit: str = "") -> str:
    """Format a numeric value, gracefully handling None."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.{precision}f}{unit}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_int(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_pct(value: Optional[float], precision: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{precision}f}%"


def _esc(text: Any) -> str:
    """HTML-escape a string."""
    return html.escape(str(text) if text is not None else "")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _short_label(label: str, max_len: int = 28) -> str:
    """Truncate a config label for axis ticks."""
    if len(label) <= max_len:
        return label
    return label[: max_len - 1] + "…"


def _run_label(payload: Dict[str, Any], idx: int) -> str:
    """Derive a short, human label for a whole JSON run file."""
    backend = payload.get("backend", "?")
    device = payload.get("device", "?")
    ts = payload.get("timestamp", "")
    # Compact timestamp: 20260718_145208 -> 07-18 14:52
    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        ts_short = dt.strftime("%m-%d %H:%M")
    except (ValueError, TypeError):
        ts_short = ts
    return f"{backend}/{device} {ts_short}" if ts_short else f"{backend}/{device} #{idx}"


# ──────────────────────────────────────────────────────────────────────
# SVG chart primitives
# ──────────────────────────────────────────────────────────────────────


def _svg_open(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" '
        f'style="max-width:100%;height:auto;font-family:ui-sans-serif,'
        f'system-ui,-apple-system,Segoe UI,Roboto,sans-serif;">'
    )


def _nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    """Return ~n "nice" tick values spanning [lo, hi]."""
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    raw_step = span / max(1, n)
    mag = 10 ** math.floor(math.log10(raw_step)) if raw_step > 0 else 1
    norm = raw_step / mag
    if norm < 1.5:
        step = 1 * mag
    elif norm < 3:
        step = 2 * mag
    elif norm < 7:
        step = 5 * mag
    else:
        step = 10 * mag
    start = math.floor(lo / step) * step
    ticks: List[float] = []
    v = start
    while v <= hi + 1e-9:
        if v >= lo - 1e-9:
            ticks.append(round(v, 10))
        v += step
    return ticks


def _fmt_tick(v: float, unit: str = "") -> str:
    """Compact tick label (e.g. 1.2k, 128k)."""
    av = abs(v)
    if av >= 1_000_000:
        s = f"{v / 1_000_000:.1f}M"
    elif av >= 1_000:
        s = f"{v / 1_000:.1f}k"
    elif av == int(av):
        s = f"{int(v)}"
    else:
        s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s + unit


def _bar_chart(
    series: List[Dict[str, Any]],
    categories: List[str],
    y_label: str,
    height: int = 320,
    y_unit: str = "",
    value_fmt: Optional[callable] = None,
    horizontal: bool = False,
) -> str:
    """Render a grouped/stacked bar chart as inline SVG.

    Args:
        series: List of ``{"name": str, "color": str, "values": List[float]}``.
            One entry per series; ``values`` aligns with ``categories``.
        categories: Category labels (x-axis ticks).
        y_label: Y-axis title.
        y_unit: Unit suffix for tick labels.
        value_fmt: Optional callable(value)->str for in-bar labels.
        horizontal: If True, draw horizontal bars (categories on y-axis).

    Returns:
        SVG string.
    """
    n_cats = len(categories)
    n_series = len(series)
    if n_cats == 0 or n_series == 0:
        return '<div class="chart-empty">No data.</div>'

    all_vals = [v for s in series for v in s["values"]]
    if not all_vals:
        return '<div class="chart-empty">No data.</div>'
    y_max = max(all_vals) if all_vals else 1.0
    y_min = min(0.0, min(all_vals))
    if y_max == y_min:
        y_max = y_min + 1.0
    # Pad top.
    y_max += (y_max - y_min) * 0.12

    if horizontal:
        width = 760
        plot_h = n_cats * (n_series * 18 + 10) + 40
        height = max(height, plot_h + 60)
        margin = {"l": 160, "r": 60, "t": 30, "b": 40}
    else:
        width = 820
        margin = {"l": 64, "r": 24, "t": 30, "b": 70}

    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]

    ticks = _nice_ticks(y_min, y_max, n=5)
    parts: List[str] = [_svg_open(width, height)]

    # Background plot area.
    parts.append(
        f'<rect x="{margin["l"]}" y="{margin["t"]}" width="{plot_w}" '
        f'height="{plot_h}" fill="{SURFACE_2}" rx="6" />'
    )

    # Y gridlines + tick labels.
    def y_to_px(v: float) -> float:
        return margin["t"] + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    def x_to_px(v: float) -> float:
        return margin["l"] + (v - y_min) / (y_max - y_min) * plot_w

    for t in ticks:
        if horizontal:
            px = x_to_px(t)
            parts.append(
                f'<line x1="{px:.1f}" y1="{margin["t"]}" x2="{px:.1f}" '
                f'y2="{margin["t"] + plot_h}" stroke="{BORDER}" '
                f'stroke-width="1" opacity="0.5" />'
            )
            parts.append(
                f'<text x="{px:.1f}" y="{margin["t"] + plot_h + 18}" '
                f'fill="{TEXT_DIM}" font-size="11" text-anchor="middle">'
                f'{_fmt_tick(t, y_unit)}</text>'
            )
        else:
            py = y_to_px(t)
            parts.append(
                f'<line x1="{margin["l"]}" y1="{py:.1f}" '
                f'x2="{margin["l"] + plot_w}" y2="{py:.1f}" '
                f'stroke="{BORDER}" stroke-width="1" opacity="0.5" />'
            )
            parts.append(
                f'<text x="{margin["l"] - 8}" y="{py + 3:.1f}" '
                f'fill="{TEXT_DIM}" font-size="11" text-anchor="end">'
                f'{_fmt_tick(t, y_unit)}</text>'
            )

    # Bars.
    if horizontal:
        cat_h = plot_h / n_cats
        bar_h = (cat_h - 6) / n_series
        for ci, cat in enumerate(categories):
            cy = margin["t"] + ci * cat_h + 3
            # Category label.
            parts.append(
                f'<text x="{margin["l"] - 8}" y="{cy + cat_h / 2:.1f}" '
                f'fill="{TEXT}" font-size="11" text-anchor="end">'
                f'{_esc(_short_label(cat, 22))}</text>'
            )
            for si, s in enumerate(series):
                v = s["values"][ci]
                by = cy + si * bar_h
                bx_end = x_to_px(v)
                bw = bx_end - margin["l"]
                parts.append(
                    f'<rect x="{margin["l"]}" y="{by:.1f}" width="{bw:.1f}" '
                    f'height="{bar_h - 2:.1f}" fill="{s["color"]}" rx="2" />'
                )
                if value_fmt and bw > 30:
                    parts.append(
                        f'<text x="{bx_end - 6:.1f}" y="{by + bar_h / 2 + 3:.1f}" '
                        f'fill="{TEXT}" font-size="10" text-anchor="end">'
                        f'{_esc(value_fmt(v))}</text>'
                    )
    else:
        cat_w = plot_w / n_cats
        group_w = cat_w * 0.7
        bar_w = group_w / n_series
        for ci, cat in enumerate(categories):
            gx = margin["l"] + ci * cat_w + (cat_w - group_w) / 2
            for si, s in enumerate(series):
                v = s["values"][ci]
                by = y_to_px(v)
                bh = (margin["t"] + plot_h) - by
                bx = gx + si * bar_w
                parts.append(
                    f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w - 2:.1f}" '
                    f'height="{max(0.0, bh):.1f}" fill="{s["color"]}" rx="2" />'
                )
                if value_fmt and bh > 14:
                    parts.append(
                        f'<text x="{bx + bar_w / 2:.1f}" y="{by - 4:.1f}" '
                        f'fill="{TEXT_DIM}" font-size="10" text-anchor="middle">'
                        f'{_esc(value_fmt(v))}</text>'
                    )
            # Category label.
            parts.append(
                f'<text x="{margin["l"] + ci * cat_w + cat_w / 2:.1f}" '
                f'y="{margin["t"] + plot_h + 18}" fill="{TEXT_DIM}" '
                f'font-size="11" text-anchor="middle">'
                f'{_esc(_short_label(cat, 18))}</text>'
            )

    # Axis titles.
    if horizontal:
        parts.append(
            f'<text x="{margin["l"] + plot_w / 2:.1f}" y="{height - 6}" '
            f'fill="{TEXT_DIM}" font-size="12" text-anchor="middle">'
            f'{_esc(y_label)}</text>'
        )
    else:
        parts.append(
            f'<text x="16" y="{margin["t"] + plot_h / 2:.1f}" fill="{TEXT_DIM}" '
            f'font-size="12" text-anchor="middle" transform="rotate(-90 16 '
            f'{margin["t"] + plot_h / 2})">{_esc(y_label)}</text>'
        )

    # Legend.
    lx = width - margin["r"] - 10
    ly = margin["t"] - 18
    parts.append(
        f'<g transform="translate({lx - n_series * 90}, {ly})">'
    )
    for si, s in enumerate(series):
        sx = si * 90
        parts.append(
            f'<rect x="{sx}" y="0" width="12" height="12" fill="{s["color"]}" rx="2" />'
        )
        parts.append(
            f'<text x="{sx + 18}" y="10" fill="{TEXT_DIM}" font-size="11">'
            f'{_esc(s["name"])}</text>'
        )
    parts.append("</g>")

    parts.append("</svg>")
    return "\n".join(parts)


def _line_chart(
    series: List[Dict[str, Any]],
    x_label: str,
    y_label: str,
    height: int = 320,
    x_unit: str = "",
    y_unit: str = "",
    x_is_numeric: bool = True,
) -> str:
    """Render a multi-series line chart as inline SVG.

    Args:
        series: List of ``{"name": str, "color": str, "x": List[float],
            "y": List[float]}``.
        x_label / y_label: Axis titles.
        x_is_numeric: If True, x-axis is numeric (ticks from data).
            If False, x values are treated as categorical labels.
    """
    width = 820
    margin = {"l": 64, "r": 24, "t": 40, "b": 70}
    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]

    all_x = [x for s in series for x in s["x"]]
    all_y = [y for s in series for y in s["y"]]
    if not all_x or not all_y:
        return '<div class="chart-empty">No data.</div>'

    x_min = min(all_x)
    x_max = max(all_x)
    if x_max == x_min:
        x_max = x_min + 1.0
    y_min = min(0.0, min(all_y))
    y_max = max(all_y)
    if y_max == y_min:
        y_max = y_min + 1.0
    y_max += (y_max - y_min) * 0.12
    x_pad = (x_max - x_min) * 0.05
    x_min -= x_pad
    x_max += x_pad

    def sx(v: float) -> float:
        return margin["l"] + (v - x_min) / (x_max - x_min) * plot_w

    def sy(v: float) -> float:
        return margin["t"] + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    parts: List[str] = [_svg_open(width, height)]
    parts.append(
        f'<rect x="{margin["l"]}" y="{margin["t"]}" width="{plot_w}" '
        f'height="{plot_h}" fill="{SURFACE_2}" rx="6" />'
    )

    # Y gridlines.
    y_ticks = _nice_ticks(y_min, y_max, n=5)
    for t in y_ticks:
        py = sy(t)
        parts.append(
            f'<line x1="{margin["l"]}" y1="{py:.1f}" x2="{margin["l"] + plot_w}" '
            f'y2="{py:.1f}" stroke="{BORDER}" stroke-width="1" opacity="0.5" />'
        )
        parts.append(
            f'<text x="{margin["l"] - 8}" y="{py + 3:.1f}" fill="{TEXT_DIM}" '
            f'font-size="11" text-anchor="end">{_fmt_tick(t, y_unit)}</text>'
        )

    # X ticks.
    if x_is_numeric:
        x_ticks = _nice_ticks(x_min, x_max, n=6)
        for t in x_ticks:
            px = sx(t)
            parts.append(
                f'<line x1="{px:.1f}" y1="{margin["t"]}" x2="{px:.1f}" '
                f'y2="{margin["t"] + plot_h}" stroke="{BORDER}" '
                f'stroke-width="1" opacity="0.25" />'
            )
            parts.append(
                f'<text x="{px:.1f}" y="{margin["t"] + plot_h + 18}" '
                f'fill="{TEXT_DIM}" font-size="11" text-anchor="middle">'
                f'{_fmt_tick(t, x_unit)}</text>'
            )
    else:
        # Categorical: use unique x values.
        uniq = sorted(set(all_x))
        for t in uniq:
            px = sx(t)
            parts.append(
                f'<text x="{px:.1f}" y="{margin["t"] + plot_h + 18}" '
                f'fill="{TEXT_DIM}" font-size="11" text-anchor="middle">'
                f'{_fmt_tick(t, x_unit)}</text>'
            )

    # Lines + points.
    for s in series:
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(s["x"], s["y"]))
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" '
            f'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />'
        )
        for x, y in zip(s["x"], s["y"]):
            parts.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" '
                f'fill="{s["color"]}" stroke="{BG}" stroke-width="1.5" />'
            )

    # Axis titles.
    parts.append(
        f'<text x="16" y="{margin["t"] + plot_h / 2:.1f}" fill="{TEXT_DIM}" '
        f'font-size="12" text-anchor="middle" transform="rotate(-90 16 '
        f'{margin["t"] + plot_h / 2})">{_esc(y_label)}</text>'
    )
    parts.append(
        f'<text x="{margin["l"] + plot_w / 2:.1f}" y="{height - 6}" '
        f'fill="{TEXT_DIM}" font-size="12" text-anchor="middle">'
        f'{_esc(x_label)}</text>'
    )

    # Legend.
    n = len(series)
    lx = margin["l"]
    ly = margin["t"] - 22
    parts.append(f'<g transform="translate({lx}, {ly})">')
    for si, s in enumerate(series):
        sx0 = si * 110
        parts.append(
            f'<line x1="{sx0}" y1="6" x2="{sx0 + 18}" y2="6" '
            f'stroke="{s["color"]}" stroke-width="2.5" />'
        )
        parts.append(
            f'<circle cx="{sx0 + 9}" cy="6" r="3.5" fill="{s["color"]}" />'
        )
        parts.append(
            f'<text x="{sx0 + 24}" y="10" fill="{TEXT_DIM}" font-size="11">'
            f'{_esc(s["name"])}</text>'
        )
    parts.append("</g>")

    parts.append("</svg>")
    return "\n".join(parts)


def _gantt_chart(
    events: List[Tuple[str, str, Optional[int], float]],
    title: str = "Stage Timeline",
    width: int = 820,
    height: int = 220,
) -> str:
    """Render a Gantt-style timeline of stage enter/exit events.

    Args:
        events: ``[(stage, event, mb_index, timestamp), ...]``.
        title: Chart title.

    Returns:
        SVG string.
    """
    # Pair enter/exit per (stage, mb_index).
    pairs: Dict[Tuple[str, Optional[int]], Dict[str, float]] = {}
    for stage, event, mb_idx, t in events:
        key = (stage, mb_idx)
        pairs.setdefault(key, {})[event] = t

    bars: List[Dict[str, Any]] = []
    for (stage, mb_idx), ev in pairs.items():
        if "enter" in ev and "exit" in ev:
            bars.append(
                {
                    "stage": stage,
                    "mb": mb_idx,
                    "start": ev["enter"],
                    "end": ev["exit"],
                }
            )
    if not bars:
        return '<div class="chart-empty">No stage events.</div>'

    t0 = min(b["start"] for b in bars)
    t1 = max(b["end"] for b in bars)
    span = t1 - t0
    if span <= 0:
        span = 1.0

    # One row per (stage, mb) — but group by stage for coloring.
    rows = sorted(set((b["stage"], b["mb"]) for b in bars), key=lambda r: (r[0] or "", r[1] or -1))
    row_idx = {r: i for i, r in enumerate(rows)}

    margin = {"l": 110, "r": 24, "t": 30, "b": 40}
    plot_w = width - margin["l"] - margin["r"]
    row_h = max(18, (height - margin["t"] - margin["b"]) // max(1, len(rows)))
    plot_h = row_h * len(rows)
    height = plot_h + margin["t"] + margin["b"]

    parts: List[str] = [_svg_open(width, height)]
    parts.append(
        f'<rect x="{margin["l"]}" y="{margin["t"]}" width="{plot_w}" '
        f'height="{plot_h}" fill="{SURFACE_2}" rx="6" />'
    )

    # X gridlines (seconds, relative).
    x_ticks = _nice_ticks(0.0, span, n=6)
    for t in x_ticks:
        px = margin["l"] + (t / span) * plot_w
        parts.append(
            f'<line x1="{px:.1f}" y1="{margin["t"]}" x2="{px:.1f}" '
            f'y2="{margin["t"] + plot_h}" stroke="{BORDER}" '
            f'stroke-width="1" opacity="0.4" />'
        )
        parts.append(
            f'<text x="{px:.1f}" y="{margin["t"] + plot_h + 18}" '
            f'fill="{TEXT_DIM}" font-size="11" text-anchor="middle">'
            f'{t:.2f}s</text>'
        )

    # Row labels + bars.
    for (stage, mb_idx), i in row_idx.items():
        ry = margin["t"] + i * row_h + 3
        label = stage
        if mb_idx is not None:
            label += f" [mb{mb_idx}]"
        parts.append(
            f'<text x="{margin["l"] - 8}" y="{ry + row_h / 2 + 3:.1f}" '
            f'fill="{TEXT_DIM}" font-size="11" text-anchor="end">'
            f'{_esc(label)}</text>'
        )
        # Faint row separator.
        parts.append(
            f'<line x1="{margin["l"]}" y1="{ry + row_h:.1f}" '
            f'x2="{margin["l"] + plot_w}" y2="{ry + row_h:.1f}" '
            f'stroke="{BORDER}" stroke-width="1" opacity="0.25" />'
        )

    for b in bars:
        i = row_idx[(b["stage"], b["mb"])]
        ry = margin["t"] + i * row_h + 4
        bx = margin["l"] + ((b["start"] - t0) / span) * plot_w
        bw = ((b["end"] - b["start"]) / span) * plot_w
        color = STAGE_COLORS.get(b["stage"], ACCENT)
        parts.append(
            f'<rect x="{bx:.1f}" y="{ry:.1f}" width="{max(1.0, bw):.1f}" '
            f'height="{row_h - 8:.1f}" fill="{color}" rx="3" opacity="0.9" />'
        )
        if bw > 28:
            dur = b["end"] - b["start"]
            parts.append(
                f'<text x="{bx + bw / 2:.1f}" y="{ry + (row_h - 8) / 2 + 3:.1f}" '
                f'fill="{BG}" font-size="10" text-anchor="middle" font-weight="600">'
                f'{dur:.2f}s</text>'
            )

    parts.append(
        f'<text x="{margin["l"] + plot_w / 2:.1f}" y="{height - 4}" '
        f'fill="{TEXT_DIM}" font-size="12" text-anchor="middle">'
        f'Time since run start (s)</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────
# Interpretation generators
# ──────────────────────────────────────────────────────────────────────


def _interp_stage_occupancy(metrics_list: List[Dict[str, Any]]) -> List[str]:
    """Bullets about where pipeline time is spent."""
    bullets: List[str] = []
    if not metrics_list:
        return bullets
    # Use the first config as representative (or the one with most repeats).
    m = metrics_list[0]
    occ = m.get("stage_occupancy_s", {})
    total = sum(occ.values()) or 1.0
    ranked = sorted(occ.items(), key=lambda kv: kv[1], reverse=True)
    if ranked:
        top_stage, top_t = ranked[0]
        pct = top_t / total * 100
        bullets.append(
            f"<strong>{top_stage}</strong> dominates at <strong>{pct:.0f}%</strong> "
            f"of total stage time ({top_t:.2f}s)."
        )
    # Extract vs prune ratio.
    ext = occ.get("extract", 0.0)
    prn = occ.get("prune", 0.0)
    if prn > 0 and ext > 0:
        ratio = ext / prn
        if ratio > 1.5:
            bullets.append(
                f"Extract takes {ratio:.1f}× longer than prune — the final LLM "
                f"call is the bottleneck, not pruning."
            )
        elif ratio < 0.67:
            bullets.append(
                f"Prune takes {1 / ratio:.1f}× longer than extract — per-chunk "
                f"pruner LLM calls dominate (many chunks per doc)."
            )
    post = occ.get("postprocess", 0.0)
    if post / total * 100 < 1.0:
        bullets.append(
            "Postprocess is negligible (<1% of stage time) — JSON repair / "
            "validation is not a bottleneck."
        )
    return bullets


def _interp_overlap(metrics_list: List[Dict[str, Any]]) -> List[str]:
    bullets: List[str] = []
    if not metrics_list:
        return bullets
    overlaps = [_safe_float(m.get("overlap_efficiency")) for m in metrics_list]
    best = max(overlaps)
    worst = min(overlaps)
    if best < 0:
        bullets.append(
            f"Overlap efficiency is negative across all configs (best {best * 100:.1f}%) "
            f"— stages ran <em>sequentially</em> with overhead; wall-clock exceeds "
            f"the sum of stage times. Pipelining is not engaged."
        )
    elif best < 0.1:
        bullets.append(
            f"Overlap efficiency is near zero (best {best * 100:.1f}%) — little "
            f"to no pipelining overlap. Increasing micro-batch size may help."
        )
    else:
        bullets.append(
            f"Best overlap efficiency {best * 100:.1f}% — pipelining is engaged "
            f"and saves wall-clock time vs. sequential execution."
        )
    if len(overlaps) > 1 and best - worst > 0.05:
        bullets.append(
            f"Spread of {(best - worst) * 100:.1f}pp across configs suggests "
            f"micro-batch size affects pipelining — sweep it further."
        )
    return bullets


def _interp_latency(metrics_list: List[Dict[str, Any]]) -> List[str]:
    bullets: List[str] = []
    if not metrics_list:
        return bullets
    # Tail behavior.
    m = metrics_list[0]
    p50 = _safe_float(m.get("latency_p50_s"))
    p99 = _safe_float(m.get("latency_p99_s"))
    repeats = m.get("repeats", 0)
    if repeats <= 1:
        bullets.append(
            f"<em>Only {repeats} repeat(s) — p90/p99 equal p50 and IQR is 0. "
            f"Tail percentiles are <strong>not reliable</strong>; rerun with "
            f"<code>--repeats 5+</code> for variance insight.</em>"
        )
    elif p50 > 0 and p99 / p50 > 1.5:
        bullets.append(
            f"p99 is {p99 / p50:.1f}× p50 — high tail-latency variance. "
            f"Investigate outliers (GC, scheduling, KV-cache misses)."
        )
    # Best config.
    best = min(metrics_list, key=lambda m: _safe_float(m.get("latency_p50_s")))
    bullets.append(
        f"Lowest p50 latency: <strong>{best.get('latency_p50_s', 0):.3f}s</strong> "
        f"at config <code>{best.get('batch_size')}×{best.get('micro_batch_size')}</code>."
    )
    return bullets


def _interp_throughput(metrics_list: List[Dict[str, Any]]) -> List[str]:
    bullets: List[str] = []
    if not metrics_list:
        return bullets
    best = max(metrics_list, key=lambda m: _safe_float(m.get("docs_per_s")))
    bullets.append(
        f"Peak throughput: <strong>{_safe_float(best.get('docs_per_s')):.2f} docs/s</strong> "
        f"(<strong>{_safe_float(best.get('tokens_per_s')):,.0f} tokens/s</strong>) "
        f"at batch={best.get('batch_size')}, mb={best.get('micro_batch_size')}."
    )
    # Scaling check across batch sizes (if multiple).
    by_bs: Dict[int, float] = {}
    for m in metrics_list:
        bs = m.get("batch_size")
        by_bs[bs] = max(by_bs.get(bs, 0.0), _safe_float(m.get("docs_per_s")))
    if len(by_bs) > 1:
        bss = sorted(by_bs)
        first, last = bss[0], bss[-1]
        ratio = by_bs[last] / by_bs[first] if by_bs[first] else 0
        ideal = last / first if first else 1
        if ratio < ideal * 0.7:
            bullets.append(
                f"Throughput scales <em>sub-linearly</em> from b={first}→b={last} "
                f"({by_bs[first]:.2f}→{by_bs[last]:.2f} docs/s, {ratio:.2f}× vs "
                f"ideal {ideal:.2f}×) — likely bottlenecked by the pruner's "
                f"per-chunk LLM calls."
            )
        else:
            bullets.append(
                f"Throughput scales well from b={first}→b={last} "
                f"({ratio:.2f}× vs ideal {ideal:.2f}×)."
            )
    return bullets


def _interp_resources(metrics_list: List[Dict[str, Any]]) -> List[str]:
    bullets: List[str] = []
    if not metrics_list:
        return bullets
    m = metrics_list[0]
    gpu_util = m.get("mean_gpu_util_pct")
    vram = m.get("peak_vram_mb")
    rss = m.get("peak_rss_mb")
    if gpu_util is not None:
        u = _safe_float(gpu_util)
        if u < 40:
            bullets.append(
                f"Mean GPU utilization is low ({u:.1f}%) — the GPU is starved; "
                f"the pipeline is CPU/IO-bound (preprocessing, chunking, or "
                f"pruner dispatch)."
            )
        elif u < 70:
            bullets.append(
                f"Mean GPU utilization is moderate ({u:.1f}%) — partial GPU "
                f"saturation; there's headroom for larger batches."
            )
        else:
            bullets.append(
                f"Mean GPU utilization is high ({u:.1f}%) — GPU is well utilized."
            )
    if vram is not None and vram > 0:
        bullets.append(f"Peak VRAM: <strong>{vram:,.0f} MB</strong>.")
    if rss is not None:
        bullets.append(f"Peak host RSS: <strong>{rss:,.0f} MB</strong>.")
    return bullets


def _interp_success(metrics_list: List[Dict[str, Any]]) -> List[str]:
    bullets: List[str] = []
    if not metrics_list:
        return bullets
    rates = [_safe_float(m.get("success_rate")) for m in metrics_list]
    if all(r >= 1.0 for r in rates):
        bullets.append("All configs achieved <strong>100% success rate</strong>.")
    else:
        worst = min(rates)
        bullets.append(
            f"Lowest success rate <strong>{worst * 100:.1f}%</strong> — some "
            f"extractions failed (JSON parse errors or LLM refusals)."
        )
    return bullets


# ──────────────────────────────────────────────────────────────────────
# Section renderers
# ──────────────────────────────────────────────────────────────────────


def _kpi_cards(metrics_list: List[Dict[str, Any]]) -> str:
    """Headline KPI cards row."""
    if not metrics_list:
        return ""
    best_p50 = min(_safe_float(m.get("latency_p50_s")) for m in metrics_list)
    best_dps = max(_safe_float(m.get("docs_per_s")) for m in metrics_list)
    best_tps = max(_safe_float(m.get("tokens_per_s")) for m in metrics_list)
    gpu_utils = [_safe_float(m.get("mean_gpu_util_pct")) for m in metrics_list if m.get("mean_gpu_util_pct") is not None]
    mean_gpu = sum(gpu_utils) / len(gpu_utils) if gpu_utils else None
    worst_success = min(_safe_float(m.get("success_rate")) for m in metrics_list)
    best_overlap = max(_safe_float(m.get("overlap_efficiency")) for m in metrics_list)
    vrams = [_safe_float(m.get("peak_vram_mb")) for m in metrics_list if m.get("peak_vram_mb")]
    peak_vram = max(vrams) if vrams else None

    cards = [
        ("Best p50 latency", f"{best_p50:.3f}s", "lower is better", ACCENT),
        ("Peak throughput", f"{best_dps:.2f} docs/s", "higher is better", GOOD),
        ("Peak token rate", f"{best_tps:,.0f} tok/s", "higher is better", ACCENT_2),
        ("Mean GPU util", _fmt_pct(mean_gpu) if mean_gpu is not None else "—", "higher is better", ACCENT_3),
        ("Worst success", f"{worst_success * 100:.1f}%", "higher is better", GOOD if worst_success >= 1 else BAD),
        ("Best overlap", f"{best_overlap * 100:.1f}%", "higher is better", ACCENT if best_overlap > 0 else BAD),
        ("Peak VRAM", f"{peak_vram:,.0f} MB" if peak_vram else "—", "lower is better", ACCENT_2),
    ]
    out = ['<div class="kpi-grid">']
    for title, value, hint, color in cards:
        out.append(
            f'<div class="kpi-card"><div class="kpi-value" style="color:{color}">'
            f'{_esc(value)}</div><div class="kpi-title">{_esc(title)}</div>'
            f'<div class="kpi-hint">{_esc(hint)}</div></div>'
        )
    out.append("</div>")
    return "\n".join(out)


def _summary_table(
    metrics_list: List[Dict[str, Any]],
    labels: List[str],
    run_labels: List[str],
) -> str:
    """Color-graded summary table across all configs/runs."""
    headers = [
        "Run", "Config", "Batch", "MB", "p50 (s)", "p90 (s)", "p99 (s)",
        "Mean (s)", "Docs/s", "Tokens/s", "Time/page (s)", "Overlap",
        "Success", "VRAM (MB)", "GPU util", "RSS (MB)",
    ]
    # For color grading we need min/max per numeric column.
    numeric_cols = {
        "latency_p50_s": ("latency_p50_s", True),   # lower better
        "latency_p90_s": ("latency_p90_s", True),
        "latency_p99_s": ("latency_p99_s", True),
        "latency_mean_s": ("latency_mean_s", True),
        "docs_per_s": ("docs_per_s", False),         # higher better
        "tokens_per_s": ("tokens_per_s", False),
        "time_per_page_s": ("time_per_page_s", True),
        "overlap_efficiency": ("overlap_efficiency", False),
        "success_rate": ("success_rate", False),
        "peak_vram_mb": ("peak_vram_mb", True),
        "mean_gpu_util_pct": ("mean_gpu_util_pct", False),
        "peak_rss_mb": ("peak_rss_mb", True),
    }
    col_ranges: Dict[str, Tuple[float, float]] = {}
    for key, (field, _) in numeric_cols.items():
        vals = [_safe_float(m.get(field)) for m in metrics_list if m.get(field) is not None]
        if vals:
            col_ranges[key] = (min(vals), max(vals))

    def cell_grade(key: str, value: Optional[float], lower_better: bool) -> str:
        if value is None or key not in col_ranges:
            return _fmt(value)
        lo, hi = col_ranges[key]
        if hi == lo:
            t = 0.5
        else:
            t = (value - lo) / (hi - lo)
        # 0 = best, 1 = worst (for lower_better). Invert for higher_better.
        norm = t if lower_better else (1 - t)
        # Map to a green→amber→red gradient via opacity on accent.
        if norm < 0.33:
            color = GOOD
        elif norm < 0.66:
            color = ACCENT_2
        else:
            color = BAD
        return f'<span style="color:{color};font-weight:600">{_fmt(value)}</span>'

    rows = []
    for i, m in enumerate(metrics_list):
        run_lbl = run_labels[i] if i < len(run_labels) else ""
        cfg_lbl = labels[i] if i < len(labels) else f"run-{i}"
        row = [
            f'<td>{_esc(run_lbl)}</td>',
            f'<td><code>{_esc(cfg_lbl)}</code></td>',
            f'<td>{m.get("batch_size", "—")}</td>',
            f'<td>{m.get("micro_batch_size", "—")}</td>',
            f'<td>{cell_grade("latency_p50_s", m.get("latency_p50_s"), True)}</td>',
            f'<td>{cell_grade("latency_p90_s", m.get("latency_p90_s"), True)}</td>',
            f'<td>{cell_grade("latency_p99_s", m.get("latency_p99_s"), True)}</td>',
            f'<td>{cell_grade("latency_mean_s", m.get("latency_mean_s"), True)}</td>',
            f'<td>{cell_grade("docs_per_s", m.get("docs_per_s"), False)}</td>',
            f'<td>{cell_grade("tokens_per_s", m.get("tokens_per_s"), False)}</td>',
            f'<td>{cell_grade("time_per_page_s", m.get("time_per_page_s"), True)}</td>',
            f'<td>{cell_grade("overlap_efficiency", m.get("overlap_efficiency"), False)}</td>',
            f'<td>{cell_grade("success_rate", m.get("success_rate"), False)}</td>',
            f'<td>{_fmt(m.get("peak_vram_mb"), precision=0)}</td>',
            f'<td>{_fmt_pct(m.get("mean_gpu_util_pct"))}</td>',
            f'<td>{_fmt(m.get("peak_rss_mb"), precision=0)}</td>',
        ]
        rows.append("<tr>" + "".join(row) + "</tr>")

    head = "".join(f"<th>{h}</th>" for h in headers)
    return (
        f'<div class="table-wrap"><table class="summary"><thead><tr>{head}</tr>'
        f'</thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def _bullets(items: Sequence[str]) -> str:
    if not items:
        return '<p class="muted">No notable observations.</p>'
    return "<ul>" + "".join(f"<li>{b}</li>" for b in items) + "</ul>"


def _section(title: str, chart_svg: str, interp: Sequence[str], chart_id: str = "") -> str:
    cid = f' id="{chart_id}"' if chart_id else ""
    return (
        f'<section class="chart-section"{cid}>'
        f'<h3>{_esc(title)}</h3>'
        f'<div class="chart">{chart_svg}</div>'
        f'<div class="interp"><span class="interp-label">Interpretation</span>'
        f'{_bullets(interp)}</div>'
        f"</section>"
    )


def _drilldown(
    metrics_list: List[Dict[str, Any]],
    labels: List[str],
    run_labels: List[str],
    raw_runs: List[Tuple[Dict[str, Any], Dict[str, Any], str, str]],
) -> str:
    """Collapsible per-config drill-down."""
    parts = ['<section class="drilldown"><h3>Per-Config Drill-Down</h3>']
    for i, (m, raw, run_lbl, cfg_lbl) in enumerate(raw_runs):
        occ = m.get("stage_occupancy_s", {})
        total_occ = sum(occ.values())
        events = []
        pr = raw.get("per_repeat", [])
        if pr:
            events = pr[0].get("stage_events", [])
        parts.append(
            f'<details><summary><code>{_esc(run_lbl)} · {_esc(cfg_lbl)}</code></summary>'
            f'<div class="drill-body">'
            f'<h4>Metrics</h4>'
            f'<table class="kv">'
        )
        kv = [
            ("Batch size", m.get("batch_size")),
            ("Micro-batch size", m.get("micro_batch_size")),
            ("Repeats", m.get("repeats")),
            ("p50 latency (s)", _fmt(m.get("latency_p50_s"), 4)),
            ("p90 latency (s)", _fmt(m.get("latency_p90_s"), 4)),
            ("p99 latency (s)", _fmt(m.get("latency_p99_s"), 4)),
            ("Mean latency (s)", _fmt(m.get("latency_mean_s"), 4)),
            ("Median latency (s)", _fmt(m.get("latency_median_s"), 4)),
            ("IQR (s)", _fmt(m.get("latency_iqr_s"), 4)),
            ("Docs/s", _fmt(m.get("docs_per_s"), 3)),
            ("Tokens/s", _fmt(m.get("tokens_per_s"), 0)),
            ("Time / page (s)", _fmt(m.get("time_per_page_s"), 4)),
            ("Time / 1k input tok (s)", _fmt(m.get("time_per_1k_input_tokens_s"), 5)),
            ("Success rate", _fmt_pct(m.get("success_rate") * 100 if m.get("success_rate") is not None else None)),
            ("Success / total", f'{m.get("success_count", "—")} / {m.get("total_count", "—")}'),
            ("Overlap efficiency", _fmt(_safe_float(m.get("overlap_efficiency")) * 100, 2, "%")),
            ("Warmup (s)", _fmt(m.get("warmup_s"), 3)),
            ("Peak VRAM (MB)", _fmt(m.get("peak_vram_mb"), 0)),
            ("Mean GPU util", _fmt_pct(m.get("mean_gpu_util_pct"))),
            ("Peak RSS (MB)", _fmt(m.get("peak_rss_mb"), 0)),
            ("Input tokens (total)", _fmt_int(m.get("input_tokens_total"))),
            ("Output tokens (total)", _fmt_int(m.get("output_tokens_total"))),
            ("Preprocess (s)", _fmt(occ.get("preprocess"), 4)),
            ("Prune (s)", _fmt(occ.get("prune"), 4)),
            ("Setup / GPU cache (s)", _fmt(occ.get("setup"), 4)),
            ("Extract (s)", _fmt(occ.get("extract"), 4)),
            ("Postprocess (s)", _fmt(occ.get("postprocess"), 4)),
            ("Total stage occupancy (s)", _fmt(total_occ, 4)),
        ]
        for k, v in kv:
            parts.append(f'<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>')
        parts.append("</table>")
        if events:
            parts.append("<h4>Stage Timeline (repeat 0)</h4>")
            parts.append(_gantt_chart(events))
        parts.append("</div></details>")
    parts.append("</section>")
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────
# PGF export (matplotlib — optional, requires a LaTeX installation)
# ──────────────────────────────────────────────────────────────────────


def _pgf_setup() -> Any:
    """Configure matplotlib for PGF export with the report's dark theme.

    Returns:
        The ``matplotlib.pyplot`` module.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    import matplotlib

    matplotlib.use("pgf")  # PGF backend (LaTeX vector graphics).
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": SURFACE_2,
            "savefig.facecolor": BG,
            "text.color": TEXT,
            "axes.labelcolor": TEXT_DIM,
            "axes.titlecolor": TEXT,
            "xtick.color": TEXT_DIM,
            "ytick.color": TEXT_DIM,
            "axes.edgecolor": BORDER,
            "grid.color": BORDER,
            "grid.alpha": 0.4,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "legend.facecolor": SURFACE,
            "legend.edgecolor": BORDER,
            "figure.dpi": 150,
            "pgf.texsystem": "xelatex",  # xelatex is the most portable.
            "pgf.preamble": (
                r"\usepackage{fontspec}"
                r"\setmainfont{DejaVu Sans}"
            ),
        }
    )
    return plt


def _pgf_bar(
    plt: Any,
    series: List[Dict[str, Any]],
    categories: List[str],
    y_label: str,
    out_path: Path,
    horizontal: bool = False,
) -> None:
    """Render a grouped/stacked bar chart to PGF."""
    import numpy as np

    n_cats = len(categories)
    n_series = len(series)
    if n_cats == 0 or n_series == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5 if not horizontal else max(4, 0.6 * n_cats)))
    x = np.arange(n_cats)
    width = 0.7 / n_series

    for si, s in enumerate(series):
        vals = s["values"]
        offset = (si - (n_series - 1) / 2) * width
        if horizontal:
            ax.barh(x + offset, vals, height=width, color=s["color"], label=s["name"])
        else:
            ax.bar(x + offset, vals, width=width, color=s["color"], label=s["name"])

    if horizontal:
        ax.set_yticks(x)
        ax.set_yticklabels([_short_label(c, 30) for c in categories])
        ax.invert_yaxis()
        ax.set_xlabel(y_label)
        ax.grid(axis="x")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([_short_label(c, 16) for c in categories], rotation=30, ha="right")
        ax.set_ylabel(y_label)
        ax.grid(axis="y")

    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def _pgf_line(
    plt: Any,
    series: List[Dict[str, Any]],
    x_label: str,
    y_label: str,
    out_path: Path,
) -> None:
    """Render a multi-series line chart to PGF."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in series:
        ax.plot(s["x"], s["y"], marker="o", color=s["color"], label=s["name"], linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def _pgf_gantt(
    plt: Any,
    events: List[Tuple[str, str, Optional[int], float]],
    out_path: Path,
) -> None:
    """Render a Gantt-style stage timeline to PGF."""
    pairs: Dict[Tuple[str, Optional[int]], Dict[str, float]] = {}
    for stage, event, mb_idx, t in events:
        key = (stage, mb_idx)
        pairs.setdefault(key, {})[event] = t

    bars: List[Dict[str, Any]] = []
    for (stage, mb_idx), ev in pairs.items():
        if "enter" in ev and "exit" in ev:
            bars.append({"stage": stage, "mb": mb_idx, "start": ev["enter"], "end": ev["exit"]})
    if not bars:
        return

    t0 = min(b["start"] for b in bars)
    rows = sorted(set((b["stage"], b["mb"]) for b in bars), key=lambda r: (r[0] or "", r[1] or -1))
    row_idx = {r: i for i, r in enumerate(rows)}

    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.5 * len(rows))))
    for b in bars:
        i = row_idx[(b["stage"], b["mb"])]
        color = STAGE_COLORS.get(b["stage"], ACCENT)
        ax.barh(
            i, b["end"] - b["start"], left=b["start"] - t0,
            height=0.6, color=color, edgecolor=BG, linewidth=0.5,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [f"{s}" + (f" [mb{m}]" if m is not None else "") for s, m in rows]
    )
    ax.invert_yaxis()
    ax.set_xlabel("Time since run start (s)")
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def _export_pgf(
    runs: List[Dict[str, Any]],
    run_labels: List[str],
    metrics_list: List[Dict[str, Any]],
    labels: List[str],
    flat_run_labels: List[str],
    raw_runs: List[Tuple[Dict[str, Any], Dict[str, Any], str, str]],
    pgf_dir: Path,
) -> List[Path]:
    """Export all charts as PGF files to ``pgf_dir``.

    Args:
        runs: Parsed JSON payloads.
        run_labels: Short label per run.
        metrics_list: Flattened metrics across all configs.
        labels: Config labels aligned with ``metrics_list``.
        flat_run_labels: Run labels aligned with ``metrics_list``.
        raw_runs: ``(metrics, raw, run_label, config_label)`` per config.
        pgf_dir: Destination directory for ``.pgf`` files.

    Returns:
        List[Path]: Paths to generated PGF files.
    """
    pgf_dir.mkdir(parents=True, exist_ok=True)
    plt = _pgf_setup()
    saved: List[Path] = []
    cats = [f"{rl} · {cl}" for rl, cl in zip(flat_run_labels, labels)]

    def _save(name: str) -> Path:
        p = pgf_dir / f"{name}.pgf"
        saved.append(p)
        return p

    # 1. Latency distribution.
    _pgf_bar(
        plt,
        series=[
            {"name": "p50", "color": ACCENT, "values": [_safe_float(m.get("latency_p50_s")) for m in metrics_list]},
            {"name": "p90", "color": ACCENT_2, "values": [_safe_float(m.get("latency_p90_s")) for m in metrics_list]},
            {"name": "p99", "color": ACCENT_3, "values": [_safe_float(m.get("latency_p99_s")) for m in metrics_list]},
        ],
        categories=cats,
        y_label="Latency (s)",
        out_path=_save("latency_distribution"),
    )

    # 2. Throughput vs batch size.
    dps_series: List[Dict[str, Any]] = []
    for ri, payload in enumerate(runs):
        mtrcs = payload.get("metrics") or []
        by_bs: Dict[int, float] = {}
        for m in mtrcs:
            bs = m.get("batch_size")
            by_bs[bs] = max(by_bs.get(bs, 0.0), _safe_float(m.get("docs_per_s")))
        if by_bs:
            bss = sorted(by_bs)
            color = ACCENT if ri == 0 else (ACCENT_2 if ri == 1 else ACCENT_3)
            dps_series.append({"name": f"{run_labels[ri]}", "color": color, "x": bss, "y": [by_bs[b] for b in bss]})
    if dps_series:
        _pgf_line(plt, dps_series, "Batch size", "Throughput (docs/s)", _save("throughput_vs_batch"))

    # 3. Per-stage occupancy.
    stage_series = []
    for stage in STAGE_ORDER:
        stage_series.append({
            "name": stage,
            "color": STAGE_COLORS[stage],
            "values": [_safe_float(m.get("stage_occupancy_s", {}).get(stage)) for m in metrics_list],
        })
    _pgf_bar(plt, stage_series, cats, "Occupancy (s)", _save("stage_occupancy"), horizontal=True)

    # 4. Gantt (first config of first run).
    if raw_runs:
        pr = raw_runs[0][1].get("per_repeat") or []
        if pr:
            events = pr[0].get("stage_events", [])
            if events:
                _pgf_gantt(plt, events, _save("stage_timeline"))

    # 5. Overlap efficiency vs micro-batch size.
    overlap_series: List[Dict[str, Any]] = []
    for ri, payload in enumerate(runs):
        mtrcs = payload.get("metrics") or []
        by_mb: Dict[int, float] = {}
        for m in mtrcs:
            mb = m.get("micro_batch_size")
            by_mb[mb] = max(by_mb.get(mb, -1e9), _safe_float(m.get("overlap_efficiency")) * 100)
        if by_mb:
            mbs = sorted(by_mb)
            color = ACCENT if ri == 0 else (ACCENT_2 if ri == 1 else ACCENT_3)
            overlap_series.append({"name": run_labels[ri], "color": color, "x": mbs, "y": [by_mb[b] for b in mbs]})
    if overlap_series:
        _pgf_line(plt, overlap_series, "Micro-batch size", "Overlap efficiency (%)", _save("overlap_vs_mb"))

    # 6. Resource usage.
    _pgf_bar(
        plt,
        [{"name": "GPU util %", "color": ACCENT, "values": [_safe_float(m.get("mean_gpu_util_pct")) for m in metrics_list]}],
        cats, "Mean GPU utilization (%)", _save("gpu_util"), horizontal=True,
    )
    _pgf_bar(
        plt,
        [{"name": "Peak VRAM (MB)", "color": ACCENT_2, "values": [_safe_float(m.get("peak_vram_mb")) for m in metrics_list]}],
        cats, "Peak VRAM (MB)", _save("peak_vram"), horizontal=True,
    )
    _pgf_bar(
        plt,
        [{"name": "Peak RSS (MB)", "color": ACCENT_3, "values": [_safe_float(m.get("peak_rss_mb")) for m in metrics_list]}],
        cats, "Peak host RSS (MB)", _save("peak_rss"), horizontal=True,
    )

    # 7. Cold start vs steady state.
    _pgf_bar(
        plt,
        [
            {"name": "Warmup (s)", "color": ACCENT_2, "values": [_safe_float(m.get("warmup_s")) for m in metrics_list]},
            {"name": "p50 (s)", "color": ACCENT, "values": [_safe_float(m.get("latency_p50_s")) for m in metrics_list]},
        ],
        cats, "Time (s)", _save("cold_start_vs_steady"),
    )

    # 8. Per-config Gantt charts from drill-down.
    for i, (m, raw, run_lbl, cfg_lbl) in enumerate(raw_runs):
        pr = raw.get("per_repeat") or []
        if pr:
            events = pr[0].get("stage_events", [])
            if events:
                safe_cfg = cfg_lbl.replace("/", "_").replace(" ", "_").replace("=", "")
                _pgf_gantt(plt, events, _save(f"timeline_{i}_{safe_cfg}"))

    return saved


# ──────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────


_CSS = f"""
:root {{
  --bg: {BG};
  --surface: {SURFACE};
  --surface-2: {SURFACE_2};
  --border: {BORDER};
  --text: {TEXT};
  --text-dim: {TEXT_DIM};
  --accent: {ACCENT};
  --accent-2: {ACCENT_2};
  --accent-3: {ACCENT_3};
  --good: {GOOD};
  --bad: {BAD};
}}
* {{ box-sizing: border-box; }}
html, body {{
  margin: 0; padding: 0;
  background: var(--bg); color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  line-height: 1.55; font-size: 15px;
}}
.container {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 80px; }}
header {{ border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 28px; }}
header h1 {{ font-size: 28px; margin: 0 0 6px; letter-spacing: -0.01em; }}
header h1 .accent {{ color: var(--accent); }}
header .sub {{ color: var(--text-dim); font-size: 14px; }}
.meta-grid {{
  display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 10px; margin-top: 18px;
}}
.meta-card {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 14px;
}}
.meta-card .k {{ color: var(--text-dim); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }}
.meta-card .v {{ font-size: 15px; margin-top: 2px; word-break: break-word; }}
h2 {{
  font-size: 20px; margin: 36px 0 14px; padding-bottom: 8px;
  border-bottom: 1px solid var(--border); color: var(--text);
}}
h3 {{ font-size: 16px; margin: 24px 0 10px; color: var(--text); }}
.kpi-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px; margin: 8px 0 8px;
}}
.kpi-card {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px 14px; text-align: center;
}}
.kpi-value {{ font-size: 24px; font-weight: 700; letter-spacing: -0.02em; }}
.kpi-title {{ color: var(--text-dim); font-size: 12px; margin-top: 4px; }}
.kpi-hint {{ color: var(--text-dim); font-size: 10px; margin-top: 2px; opacity: 0.8; }}
.table-wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; }}
table.summary, table.kv {{
  border-collapse: collapse; width: 100%; font-size: 13px;
}}
table.summary th, table.summary td, table.kv th, table.kv td {{
  padding: 8px 10px; text-align: right; border-bottom: 1px solid var(--border);
}}
table.summary th:first-child, table.summary td:first-child,
table.summary th:nth-child(2), table.summary td:nth-child(2),
table.kv th {{ text-align: left; }}
table.summary thead th {{
  background: var(--surface); color: var(--text-dim); font-weight: 600;
  position: sticky; top: 0; font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.03em;
}}
table.summary tbody tr:nth-child(even) {{ background: var(--surface); }}
table.kv {{ width: auto; min-width: 420px; }}
table.kv th {{ color: var(--text-dim); width: 50%; }}
table.kv td {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; }}
.chart-section {{ margin: 28px 0; }}
.chart {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px; overflow-x: auto;
}}
.chart svg {{ display: block; margin: 0 auto; }}
.chart-empty {{ color: var(--text-dim); padding: 40px; text-align: center; }}
.interp {{
  margin-top: 12px; padding: 12px 16px; background: var(--surface);
  border-left: 3px solid var(--accent); border-radius: 0 8px 8px 0;
}}
.interp-label {{
  display: block; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--accent); margin-bottom: 6px; font-weight: 600;
}}
.interp ul {{ margin: 0; padding-left: 18px; }}
.interp li {{ margin: 4px 0; color: var(--text); }}
.interp li code {{ background: var(--surface-2); padding: 1px 5px; border-radius: 3px; font-size: 12px; }}
.interp li em {{ color: var(--text-dim); }}
.muted {{ color: var(--text-dim); }}
.drilldown details {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  margin: 8px 0; padding: 0;
}}
.drilldown summary {{
  cursor: pointer; padding: 12px 16px; font-size: 14px; user-select: none;
}}
.drilldown summary:hover {{ background: var(--surface-2); border-radius: 8px; }}
.drilldown .drill-body {{ padding: 0 16px 16px; }}
.drilldown h4 {{ font-size: 13px; color: var(--text-dim); margin: 16px 0 6px; }}
footer {{
  margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--border);
  color: var(--text-dim); font-size: 12px; text-align: center;
}}
footer a {{ color: var(--accent); text-decoration: none; }}
"""


# ──────────────────────────────────────────────────────────────────────
# Top-level report assembly
# ──────────────────────────────────────────────────────────────────────


def build_report(
    runs: List[Dict[str, Any]],
    run_labels: List[str],
    out_path: Path,
    pgf_dir: Optional[Path] = None,
) -> List[Path]:
    """Assemble the full HTML report and write it to ``out_path``.

    Args:
        runs: List of parsed JSON payloads (one per input file).
        run_labels: Short label per run (for the "Run" column / legend).
        out_path: Destination ``.html`` path.
        pgf_dir: If provided, also export all charts as PGF files to this
            directory (requires matplotlib + a LaTeX installation).

    Returns:
        List[Path]: Paths to PGF files if ``pgf_dir`` was given, else ``[]``.
    """
    # Flatten: every (run, config) becomes one row in metrics_list.
    metrics_list: List[Dict[str, Any]] = []
    labels: List[str] = []
    flat_run_labels: List[str] = []
    raw_runs: List[Tuple[Dict[str, Any], Dict[str, Any], str, str]] = []

    for ri, (payload, run_lbl) in enumerate(zip(runs, run_labels)):
        cfgs = payload.get("configs") or []
        mtrcs = payload.get("metrics") or []
        raws = payload.get("raw") or []
        for ci, m in enumerate(mtrcs):
            metrics_list.append(m)
            label = cfgs[ci] if ci < len(cfgs) else f"b={m.get('batch_size')} mb={m.get('micro_batch_size')}"
            labels.append(label)
            flat_run_labels.append(run_lbl)
            raw = raws[ci] if ci < len(raws) else {}
            raw_runs.append((m, raw, run_lbl, label))

    # ── Header / metadata (use first run as representative) ──
    p0 = runs[0] if runs else {}
    ts = p0.get("timestamp", "")
    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        ts_pretty = dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        ts_pretty = ts

    meta_items = [
        ("Backend", p0.get("backend", "—")),
        ("Device", p0.get("device", "—")),
        ("Corpus", p0.get("corpus", "—")),
        ("Corpus size", p0.get("corpus_size", "—")),
        ("Query", p0.get("query", "—")),
        ("Repeats", p0.get("repeats", "—")),
        ("Warmup", p0.get("warmup", "—")),
        ("Batch sizes", ", ".join(str(b) for b in (p0.get("batch_sizes") or []))),
        ("Micro-batch sizes", ", ".join(str(b) for b in (p0.get("micro_batch_sizes") or []))),
        ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    ]
    meta_cards = "".join(
        f'<div class="meta-card"><div class="k">{_esc(k)}</div>'
        f'<div class="v">{_esc(v)}</div></div>'
        for k, v in meta_items
    )

    runs_badge = (
        f'<span class="accent">{len(runs)}</span> run(s) · '
        f'{len(metrics_list)} config(s)'
    )
    header = (
        f'<header><h1>Axetract <span class="accent">Benchmark Report</span></h1>'
        f'<div class="sub">{_esc(ts_pretty)} · {runs_badge}</div>'
        f'<div class="meta-grid">{meta_cards}</div></header>'
    )

    # ── KPI cards ──
    kpi = _kpi_cards(metrics_list)

    # ── Summary table ──
    table = _summary_table(metrics_list, labels, flat_run_labels)

    # ── Charts ──
    # 1. Latency distribution (grouped bars p50/p90/p99).
    latency_chart = _bar_chart(
        series=[
            {"name": "p50", "color": ACCENT, "values": [_safe_float(m.get("latency_p50_s")) for m in metrics_list]},
            {"name": "p90", "color": ACCENT_2, "values": [_safe_float(m.get("latency_p90_s")) for m in metrics_list]},
            {"name": "p99", "color": ACCENT_3, "values": [_safe_float(m.get("latency_p99_s")) for m in metrics_list]},
        ],
        categories=[f"{rl}\n{cl}" for rl, cl in zip(flat_run_labels, labels)],
        y_label="Latency (s)",
        y_unit="s",
        value_fmt=lambda v: f"{v:.2f}s",
    )
    latency_interp = _interp_latency(metrics_list)

    # 2. Throughput vs batch size (line). Group by run; x = batch size.
    tps_series: List[Dict[str, Any]] = []
    dps_series: List[Dict[str, Any]] = []
    # Build per-run series: x=batch_size, y=docs_per_s (max across mb for that bs).
    for ri, payload in enumerate(runs):
        mtrcs = payload.get("metrics") or []
        by_bs: Dict[int, float] = {}
        by_bs_tps: Dict[int, float] = {}
        for m in mtrcs:
            bs = m.get("batch_size")
            by_bs[bs] = max(by_bs.get(bs, 0.0), _safe_float(m.get("docs_per_s")))
            by_bs_tps[bs] = max(by_bs_tps.get(bs, 0.0), _safe_float(m.get("tokens_per_s")))
        if by_bs:
            bss = sorted(by_bs)
            color = ACCENT if ri == 0 else (ACCENT_2 if ri == 1 else ACCENT_3)
            dps_series.append({"name": f"{run_labels[ri]} docs/s", "color": color, "x": bss, "y": [by_bs[b] for b in bss]})
            tps_series.append({"name": f"{run_labels[ri]} tok/s", "color": color, "x": bss, "y": [by_bs_tps[b] for b in bss]})

    throughput_chart = _line_chart(
        series=dps_series or [{"name": "docs/s", "color": ACCENT, "x": [0], "y": [0]}],
        x_label="Batch size",
        y_label="Throughput (docs/s)",
        y_unit="",
    )
    throughput_interp = _interp_throughput(metrics_list)

    # 3. Per-stage occupancy stacked (horizontal bars per config).
    stage_series = []
    for stage in STAGE_ORDER:
        stage_series.append({
            "name": stage,
            "color": STAGE_COLORS[stage],
            "values": [_safe_float(m.get("stage_occupancy_s", {}).get(stage)) for m in metrics_list],
        })
    stage_chart = _bar_chart(
        series=stage_series,
        categories=[f"{rl} · {cl}" for rl, cl in zip(flat_run_labels, labels)],
        y_label="Occupancy (s)",
        y_unit="s",
        horizontal=True,
        value_fmt=lambda v: f"{v:.2f}s",
    )
    stage_interp = _interp_stage_occupancy(metrics_list)

    # 4. Stage timeline (Gantt) — first config of first run.
    gantt_svg = '<div class="chart-empty">No stage events available.</div>'
    if raw_runs:
        pr = raw_runs[0][1].get("per_repeat") or []
        if pr:
            events = pr[0].get("stage_events", [])
            if events:
                gantt_svg = _gantt_chart(events, title=f"Stage Timeline — {raw_runs[0][2]} · {raw_runs[0][3]}")
    gantt_interp = _interp_overlap(metrics_list)

    # 5. Overlap efficiency vs micro-batch size (line, per run).
    overlap_series: List[Dict[str, Any]] = []
    for ri, payload in enumerate(runs):
        mtrcs = payload.get("metrics") or []
        by_mb: Dict[int, float] = {}
        for m in mtrcs:
            mb = m.get("micro_batch_size")
            by_mb[mb] = max(by_mb.get(mb, -1e9), _safe_float(m.get("overlap_efficiency")) * 100)
        if by_mb:
            mbs = sorted(by_mb)
            color = ACCENT if ri == 0 else (ACCENT_2 if ri == 1 else ACCENT_3)
            overlap_series.append({"name": run_labels[ri], "color": color, "x": mbs, "y": [by_mb[b] for b in mbs]})
    overlap_chart = _line_chart(
        series=overlap_series or [{"name": "overlap", "color": ACCENT, "x": [0], "y": [0]}],
        x_label="Micro-batch size",
        y_label="Overlap efficiency (%)",
        y_unit="%",
    )

    # 6. Resource usage (grouped bars: GPU util, VRAM, RSS — normalized per config).
    # Use horizontal bars; three separate small charts to keep units sane.
    gpu_vals = [_safe_float(m.get("mean_gpu_util_pct")) for m in metrics_list]
    vram_vals = [_safe_float(m.get("peak_vram_mb")) for m in metrics_list]
    rss_vals = [_safe_float(m.get("peak_rss_mb")) for m in metrics_list]
    cats = [f"{rl} · {cl}" for rl, cl in zip(flat_run_labels, labels)]
    gpu_chart = _bar_chart(
        series=[{"name": "GPU util %", "color": ACCENT, "values": gpu_vals}],
        categories=cats, y_label="Mean GPU utilization (%)", y_unit="%",
        horizontal=True, value_fmt=lambda v: f"{v:.1f}%",
    )
    vram_chart = _bar_chart(
        series=[{"name": "Peak VRAM (MB)", "color": ACCENT_2, "values": vram_vals}],
        categories=cats, y_label="Peak VRAM (MB)", y_unit="MB",
        horizontal=True, value_fmt=lambda v: f"{v:,.0f}MB",
    )
    rss_chart = _bar_chart(
        series=[{"name": "Peak RSS (MB)", "color": ACCENT_3, "values": rss_vals}],
        categories=cats, y_label="Peak host RSS (MB)", y_unit="MB",
        horizontal=True, value_fmt=lambda v: f"{v:,.0f}MB",
    )
    resource_interp = _interp_resources(metrics_list)

    # 7. Cold start vs steady-state p50.
    cold_chart = _bar_chart(
        series=[
            {"name": "Warmup (s)", "color": ACCENT_2, "values": [_safe_float(m.get("warmup_s")) for m in metrics_list]},
            {"name": "p50 (s)", "color": ACCENT, "values": [_safe_float(m.get("latency_p50_s")) for m in metrics_list]},
        ],
        categories=cats, y_label="Time (s)", y_unit="s",
        value_fmt=lambda v: f"{v:.2f}s",
    )
    cold_interp: List[str] = []
    if metrics_list:
        w = _safe_float(metrics_list[0].get("warmup_s"))
        p = _safe_float(metrics_list[0].get("latency_p50_s"))
        if w > 0 and p > 0:
            cold_interp.append(
                f"Warmup is {w / p:.1f}× the steady-state p50 — first-request "
                f"latency is dominated by model/compile warmup."
            )

    # ── Success section ──
    success_interp = _interp_success(metrics_list)

    # ── Drill-down ──
    drilldown_html = _drilldown(metrics_list, labels, flat_run_labels, raw_runs)

    # ── Assemble ──
    body = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Axetract Benchmark Report — {_esc(ts_pretty)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
{header}
<h2>Headline Metrics</h2>
{kpi}
<h2>Summary Table</h2>
{table}
<h2>Latency Distribution</h2>
{_section("Latency percentiles per config (p50 / p90 / p99)", latency_chart, latency_interp, "latency")}
<h2>Throughput vs. Batch Size</h2>
{_section("Throughput (docs/s) vs. batch size, per run", throughput_chart, throughput_interp, "throughput")}
<h2>Per-Stage Occupancy</h2>
{_section("Where pipeline time is spent, per config", stage_chart, stage_interp, "stages")}
<h2>Pipeline Overlap</h2>
{_section("Stage timeline (Gantt) — first config of first run", gantt_svg, gantt_interp, "gantt")}
{_section("Overlap efficiency vs. micro-batch size", overlap_chart, [], "overlap")}
<h2>Resource Usage</h2>
{_section("Mean GPU utilization per config", gpu_chart, [], "gpu")}
{_section("Peak VRAM per config", vram_chart, [], "vram")}
{_section("Peak host RSS per config", rss_chart, resource_interp, "rss")}
<h2>Cold Start vs. Steady State</h2>
{_section("Warmup time vs. steady-state p50 latency", cold_chart, cold_interp, "cold")}
<h2>Reliability</h2>
<div class="interp"><span class="interp-label">Interpretation</span>{_bullets(success_interp)}</div>
{drilldown_html}
<footer>
Generated by <code>benchmarks.html_report</code> ·
<a href="https://github.com/abdo-Mansour/axetract">axetract</a>
</footer>
</div>
</body>
</html>
"""
    out_path.write_text(body, encoding="utf-8")

    # ── PGF export (optional) ──
    pgf_paths: List[Path] = []
    if pgf_dir is not None:
        try:
            pgf_paths = _export_pgf(
                runs, run_labels, metrics_list, labels, flat_run_labels,
                raw_runs, pgf_dir,
            )
            logger.info("Exported %d PGF charts to %s", len(pgf_paths), pgf_dir)
        except ImportError:
            logger.warning(
                "matplotlib not installed — skipping PGF export. "
                "Install it with: uv pip install matplotlib"
            )
        except Exception as e:
            logger.warning("PGF export failed: %s", e)
    return pgf_paths


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        int: Exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        prog="benchmarks.html_report",
        description="Render Axetract benchmark JSON to a self-contained HTML report.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more benchmark JSON files (glob patterns supported).",
    )
    parser.add_argument(
        "--out", "-o",
        type=str,
        default=None,
        help="Output HTML path (default: alongside the first input, .html extension).",
    )
    parser.add_argument(
        "--pgf",
        action="store_true",
        help="Also export all charts as PGF (LaTeX vector graphics) files. "
             "Requires matplotlib and a LaTeX installation (xelatex).",
    )
    parser.add_argument(
        "--pgf-dir",
        type=str,
        default=None,
        help="Directory for PGF files (default: <out>_pgf/).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Expand globs / dedupe while preserving order.
    in_paths: List[Path] = []
    seen: set = set()
    for pat in args.inputs:
        if any(c in pat for c in "*?["):
            matched = sorted(Path(".").glob(pat))
        else:
            matched = [Path(pat)]
        for p in matched:
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                in_paths.append(p)

    if not in_paths:
        logger.error("No input files found.")
        return 2

    runs: List[Dict[str, Any]] = []
    run_labels: List[str] = []
    for i, p in enumerate(in_paths):
        if not p.exists():
            logger.error("File not found: %s", p)
            return 2
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in %s: %s", p, e)
            return 2
        runs.append(payload)
        run_labels.append(_run_label(payload, i))
        logger.info("Loaded %s as '%s'", p, run_labels[-1])

    # Output path.
    if args.out:
        out_path = Path(args.out)
    else:
        first = in_paths[0]
        if len(in_paths) == 1:
            out_path = first.with_suffix(".html")
        else:
            out_path = first.parent / "benchmark_report.html"

    # PGF output directory.
    pgf_dir: Optional[Path] = None
    if args.pgf or args.pgf_dir:
        if args.pgf_dir:
            pgf_dir = Path(args.pgf_dir)
        else:
            pgf_dir = out_path.with_suffix("") if out_path.suffix else out_path
            pgf_dir = Path(str(pgf_dir) + "_pgf")

    build_report(runs, run_labels, out_path, pgf_dir=pgf_dir)
    logger.info("HTML report written to %s", out_path)
    print(f"\nReport: {out_path}")
    if pgf_dir:
        print(f"PGF:    {pgf_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
