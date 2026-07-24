"""Report rendering for Axetract benchmark results.

Converts aggregated metrics (from :mod:`benchmarks.metrics`) into a
human-readable Markdown summary table and optional matplotlib plots.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Markdown table
# ──────────────────────────────────────────────────────────────────────


def _fmt(value: Any, unit: str = "", precision: int = 2) -> str:
    """Format a numeric value for the Markdown table, handling None."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{precision}f}{unit}"
    return f"{value}{unit}"


def _fmt_pct(value: Optional[float]) -> str:
    """Format a percentage value."""
    if value is None:
        return "—"
    return f"{value:.1f}%"


def to_markdown(
    metrics_list: List[Dict[str, Any]],
    config_labels: Optional[List[str]] = None,
) -> str:
    """Render a list of run metrics as a Markdown summary table.

    Args:
        metrics_list: List of metric dicts from
            :func:`benchmarks.metrics.compute_run_metrics`.
        config_labels: Optional list of labels (e.g. ``"vllm/gpu"``) for each
            row.  If None, labels are derived from batch/micro-batch sizes.

    Returns:
        str: Markdown table string. When the run was performed with
        ``--pruner both``, a ``Pruner Δ`` column is appended that shows the
        per-config overhead of the pruner relative to its pruner-off twin.
    """
    if not metrics_list:
        return "_No benchmark results to display._"

    # Compute per-row pruner deltas (only meaningful for --pruner=both runs).
    from benchmarks.pruner_delta import (
        format_pruner_delta,
        pair_pruner_runs,
        pruner_enabled_for_label,
    )
    pairs = pair_pruner_runs(metrics_list, config_labels)
    # Lookup: on_idx / off_idx -> the other twin's metrics.
    delta_cells: Dict[int, str] = {}
    has_any_pair = bool(pairs)
    for pair in pairs.values():
        if pair["on_idx"] is not None and pair["off"] is not None:
            on = pair["on"]
            off = pair["off"]
            delta_cells[pair["on_idx"]] = format_pruner_delta(
                float(on.get("latency_mean_s") or 0.0),
                float(off.get("latency_mean_s") or 0.0),
                pruner_enabled=True,
            )
        if pair["off_idx"] is not None and pair["on"] is not None:
            on = pair["on"]
            off = pair["off"]
            delta_cells[pair["off_idx"]] = format_pruner_delta(
                float(on.get("latency_mean_s") or 0.0),
                float(off.get("latency_mean_s") or 0.0),
                pruner_enabled=False,
            )

    headers = [
        "Config",
        "Batch",
        "MB",
        "p50 (s)",
        "p90 (s)",
        "p99 (s)",
        "Mean (s)",
        "Docs/s",
        "Tokens/s",
        "Time/page (s)",
        "Overlap",
        "Success",
        "VRAM (MB)",
        "GPU util",
    ]
    if has_any_pair:
        headers.append("Pruner Δ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    for i, m in enumerate(metrics_list):
        label = config_labels[i] if config_labels else f"run-{i}"
        row = [
            label,
            str(m["batch_size"]),
            str(m["micro_batch_size"]),
            _fmt(m["latency_p50_s"]),
            _fmt(m["latency_p90_s"]),
            _fmt(m["latency_p99_s"]),
            _fmt(m["latency_mean_s"]),
            _fmt(m["docs_per_s"], precision=1),
            _fmt(m["tokens_per_s"], precision=0),
            _fmt(m["time_per_page_s"]),
            _fmt(m["overlap_efficiency"] * 100, unit="%", precision=1) if m["overlap_efficiency"] is not None else "—",
            _fmt(m["success_rate"] * 100, unit="%", precision=1),
            _fmt(m["peak_vram_mb"], precision=0) if m["peak_vram_mb"] is not None else "—",
            _fmt_pct(m["mean_gpu_util_pct"]),
        ]
        if has_any_pair:
            # Use the precomputed cell if the row is part of a pair;
            # otherwise show "—" for rows with no twin (e.g. --pruner=use
            # labels that lack a [-pruner] sibling).
            row.append(delta_cells.get(i, "—"))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def to_markdown_full(
    metrics_list: List[Dict[str, Any]],
    config_labels: Optional[List[str]] = None,
) -> str:
    """Render a full Markdown report including per-stage breakdown.

    Args:
        metrics_list: List of metric dicts.
        config_labels: Optional row labels.

    Returns:
        str: Full Markdown report.
    """
    parts: List[str] = ["# Axetract Benchmark Report\n"]

    # Summary table.
    parts.append("## Summary\n")
    parts.append(to_markdown(metrics_list, config_labels))
    parts.append("")

    # Per-stage occupancy breakdown.
    parts.append("## Per-Stage Occupancy (mean seconds)\n")
    stage_headers = ["Config", "Preprocess", "Prune", "Extract", "Postprocess", "Total"]
    lines = [
        "| " + " | ".join(stage_headers) + " |",
        "|" + "|".join(["---"] * len(stage_headers)) + "|",
    ]
    for i, m in enumerate(metrics_list):
        label = config_labels[i] if config_labels else f"run-{i}"
        occ = m["stage_occupancy_s"]
        total = sum(occ.values())
        row = [
            label,
            _fmt(occ.get("preprocess", 0.0), precision=3),
            _fmt(occ.get("prune", 0.0), precision=3),
            _fmt(occ.get("extract", 0.0), precision=3),
            _fmt(occ.get("postprocess", 0.0), precision=3),
            _fmt(total, precision=3),
        ]
        lines.append("| " + " | ".join(row) + " |")
    parts.append("\n".join(lines))
    parts.append("")

    # Cold start.
    parts.append("## Cold Start (warmup)\n")
    warmup_headers = ["Config", "Warmup (s)"]
    lines = [
        "| " + " | ".join(warmup_headers) + " |",
        "|" + "|".join(["---"] * len(warmup_headers)) + "|",
    ]
    for i, m in enumerate(metrics_list):
        label = config_labels[i] if config_labels else f"run-{i}"
        lines.append(f"| {label} | {_fmt(m['warmup_s'], precision=3)} |")
    parts.append("\n".join(lines))
    parts.append("")

    # ── Pruner overhead section (only when --pruner=both produced pairs) ──
    from benchmarks.pruner_delta import (
        aggregate_overhead,
        format_pruner_delta,
        pair_pruner_runs,
    )
    pairs = pair_pruner_runs(metrics_list, config_labels)
    agg = aggregate_overhead(pairs)
    if agg is not None:
        parts.append("## Pruner Overhead (--pruner both)\n")
        parts.append(
            f"Across **{agg['n_pairs']}** paired configurations, the pruner "
            f"adds **{agg['mean_delta_s']:.2f}s** of mean latency per run on "
            f"average (**×{agg['mean_delta_ratio']:.2f}** vs. the pruner-off "
            f"twin). The pruner stage itself accounts for "
            f"**{agg['mean_prune_pct']:.1f}%** of wall-clock time in the "
            f"pruner-on runs.\n"
        )
        # Per-config rows.
        overhead_headers = ["Config", "On mean (s)", "Off mean (s)", "Δ (s)", "× ratio", "Prune occ (s)"]
        lines = [
            "| " + " | ".join(overhead_headers) + " |",
            "|" + "|".join(["---"] * len(overhead_headers)) + "|",
        ]
        for key, pair in pairs.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        key,
                        f"{pair['on_mean_s']:.2f}",
                        f"{pair['off_mean_s']:.2f}",
                        f"{pair['delta_s']:+.2f}",
                        f"×{pair['delta_ratio']:.2f}",
                        f"{pair['prune_occupancy_s']:.2f}",
                    ]
                )
                + " |"
            )
        parts.append("\n".join(lines))
        parts.append("")

    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────
# Plots (matplotlib — optional)
# ──────────────────────────────────────────────────────────────────────


def to_plots(
    metrics_list: List[Dict[str, Any]],
    config_labels: Optional[List[str]] = None,
    outdir: str | Path = "benchmarks/results/plots",
) -> List[str]:
    """Generate matplotlib plots and return paths to saved PNGs.

    Plots produced:
    1. Throughput (docs/s) vs. batch size.
    2. Per-stage occupancy stacked bar.
    3. Overlap efficiency vs. micro-batch size.
    4. Latency (p50/p90/p99) vs. batch size.
    5. Pruner overhead ratio vs. batch size (only when ``--pruner=both``
       produced paired runs).

    When the run contains ``+pruner`` / ``-pruner`` twins, the line charts
    plot each variant as its own series (``+pruner`` solid, ``-pruner``
    dashed) so the latency/throughput contribution of the chunk-level
    pruner LoRA calls is visible at a glance. The per-stage occupancy bar
    chart gets an outline + annotation over the "prune" segment of any
    ``+pruner`` bar to call out the bottleneck.

    Args:
        metrics_list: List of metric dicts.
        config_labels: Optional row labels.
        outdir: Directory to write PNGs into.

    Returns:
        List[str]: Paths to generated PNG files.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend.
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plots.")
        return []

    from benchmarks.pruner_delta import (
        aggregate_overhead,
        canonical_key_for_label,
        pair_pruner_runs,
        pruner_enabled_for_label,
    )

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    labels = config_labels or [f"run-{i}" for i in range(len(metrics_list))]
    saved: List[str] = []

    # ── Pre-compute pruner-on/off series membership for plot styling ──
    # `pruner_state` is one of: True, False, None. None ⇒ single-variant
    # run (--pruner use / skip) where every label belongs to a single series.
    pruner_state: List[Optional[bool]] = [
        pruner_enabled_for_label(lbl) for lbl in labels
    ]
    has_pairs = any(s is not None for s in pruner_state) and not all(
        s is None for s in pruner_state
    )

    def _series_style(idx: int, *, default_marker: str = "o") -> Dict[str, Any]:
        """Return matplotlib kwargs for plotting a single config."""
        state = pruner_state[idx]
        if state is True:
            return {"linestyle": "-", "linewidth": 2.5, "marker": default_marker}
        if state is False:
            return {"linestyle": "--", "linewidth": 1.8, "marker": default_marker, "alpha": 0.75}
        return {"linestyle": "-", "linewidth": 2.0, "marker": default_marker}

    # ── 1. Throughput vs. batch size ──
    fig, ax = plt.subplots(figsize=(8, 5))
    batch_sizes = [m["batch_size"] for m in metrics_list]
    docs_per_s = [m["docs_per_s"] for m in metrics_list]
    if has_pairs:
        # Plot each variant as its own series so the pruner-on / pruner-off
        # distinction reads at a glance.
        for state, ls, lbl_prefix, color in (
            (True, "-", "+pruner", "#C44E52"),
            (False, "--", "-pruner", "#55A868"),
        ):
            xs = [batch_sizes[i] for i, s in enumerate(pruner_state) if s == state]
            ys = [docs_per_s[i] for i, s in enumerate(pruner_state) if s == state]
            if xs:
                ax.plot(xs, ys, linestyle=ls, marker="o", color=color,
                        label=f"docs/s ({lbl_prefix})")
    else:
        ax.plot(batch_sizes, docs_per_s, "o-", label="docs/s")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (docs/s)")
    ax.set_title("Throughput vs. Batch Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = out_path / "throughput_vs_batch.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # ── 2. Per-stage occupancy stacked bar ──
    fig, ax = plt.subplots(figsize=(10, 6))
    stages = ["preprocess", "prune", "extract", "postprocess"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    x = range(len(metrics_list))
    bottoms = [0.0] * len(metrics_list)
    bar_handles = []
    for stage, color in zip(stages, colors):
        vals = [m["stage_occupancy_s"].get(stage, 0.0) for m in metrics_list]
        bars = ax.bar(x, vals, bottom=bottoms, label=stage, color=color)
        bar_handles.append((stage, bars))
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    # Outline the prune segment of any +pruner bar to draw the eye.
    if has_pairs:
        prune_idx = stages.index("prune")
        for i, state in enumerate(pruner_state):
            if state is not True:
                continue
            prune_bars = bar_handles[prune_idx][1]
            bar = prune_bars[i]
            prune_height = bar.get_height()
            if prune_height <= 0:
                continue
            ax.annotate(
                "pruner LLM",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + prune_height / 2),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Occupancy (s)")
    ax.set_title("Per-Stage Occupancy")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    p = out_path / "stage_occupancy.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # ── 3. Overlap efficiency vs. micro-batch size ──
    fig, ax = plt.subplots(figsize=(8, 5))
    mb_sizes = [m["micro_batch_size"] for m in metrics_list]
    overlaps = [m["overlap_efficiency"] * 100 for m in metrics_list]
    if has_pairs:
        for state, ls, color, marker in (
            (True, "-", "#C44E52", "s"),
            (False, "--", "#55A868", "s"),
        ):
            xs = [mb_sizes[i] for i, s in enumerate(pruner_state) if s == state]
            ys = [overlaps[i] for i, s in enumerate(pruner_state) if s == state]
            if xs:
                ax.plot(xs, ys, linestyle=ls, marker=marker, color=color,
                        label="+pruner" if state is True else "−pruner")
    else:
        ax.plot(mb_sizes, overlaps, "s-", color="#DD8452", label="overlap %")
    ax.set_xlabel("Micro-Batch Size")
    ax.set_ylabel("Overlap Efficiency (%)")
    ax.set_title("Pipelining Overlap Efficiency vs. Micro-Batch Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = out_path / "overlap_vs_mb.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # ── 4. Latency vs. batch size ──
    fig, ax = plt.subplots(figsize=(8, 5))
    p50 = [m["latency_p50_s"] for m in metrics_list]
    p90 = [m["latency_p90_s"] for m in metrics_list]
    p99 = [m["latency_p99_s"] for m in metrics_list]
    if has_pairs:
        for state, ls, color in (
            (True, "-", "#C44E52"),
            (False, "--", "#55A868"),
        ):
            xs = [batch_sizes[i] for i, s in enumerate(pruner_state) if s == state]
            for vals, lbl, mk in ((p50, "p50", "o"), (p90, "p90", "s"), (p99, "p99", "^")):
                ys = [vals[i] for i, s in enumerate(pruner_state) if s == state]
                if xs:
                    tag = "+pruner" if state is True else "−pruner"
                    ax.plot(xs, ys, linestyle=ls, marker=mk, color=color,
                            label=f"{lbl} ({tag})")
    else:
        ax.plot(batch_sizes, p50, "o-", label="p50")
        ax.plot(batch_sizes, p90, "s-", label="p90")
        ax.plot(batch_sizes, p99, "^-", label="p99")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency vs. Batch Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = out_path / "latency_vs_batch.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # ── 5. Pruner overhead ratio vs. batch size (only when pairs exist) ──
    pairs = pair_pruner_runs(metrics_list, config_labels)
    agg = aggregate_overhead(pairs)
    if agg is not None and pairs:
        fig, ax = plt.subplots(figsize=(8, 5))
        # Sort by batch size for a clean line.
        sorted_keys = sorted(pairs.keys(), key=lambda k: (pairs[k]["batch_size"], pairs[k]["micro_batch_size"]))
        bs_x = [pairs[k]["batch_size"] for k in sorted_keys]
        ratios = [pairs[k]["delta_ratio"] for k in sorted_keys]
        deltas = [pairs[k]["delta_s"] for k in sorted_keys]
        ax.plot(bs_x, ratios, linestyle="-", marker="o", color="#C44E52", label="× ratio (latency)")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Pruner overhead multiplier (×)")
        ax.set_title(
            f"Pruner overhead vs. batch size (mean ×{agg['mean_delta_ratio']:.2f}, "
            f"{agg['mean_prune_pct']:.0f}% wall time)"
        )
        # Twin-axis: delta in seconds.
        ax2 = ax.twinx()
        ax2.plot(bs_x, deltas, linestyle="--", marker="s", color="#DD8452", label="Δ seconds")
        ax2.set_ylabel("Δ latency (s)")
        # Combined legend.
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax.grid(True, alpha=0.3)
        p = out_path / "pruner_overhead.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    logger.info("Saved %d plots to %s", len(saved), out_path)
    return saved
