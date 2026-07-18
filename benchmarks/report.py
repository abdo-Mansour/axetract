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
        str: Markdown table string.
    """
    if not metrics_list:
        return "_No benchmark results to display._"

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

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    labels = config_labels or [f"run-{i}" for i in range(len(metrics_list))]
    saved: List[str] = []

    # ── 1. Throughput vs. batch size ──
    fig, ax = plt.subplots(figsize=(8, 5))
    batch_sizes = [m["batch_size"] for m in metrics_list]
    docs_per_s = [m["docs_per_s"] for m in metrics_list]
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
    for stage, color in zip(stages, colors):
        vals = [m["stage_occupancy_s"].get(stage, 0.0) for m in metrics_list]
        ax.bar(x, vals, bottom=bottoms, label=stage, color=color)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
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
    ax.plot(mb_sizes, overlaps, "s-", color="#DD8452")
    ax.set_xlabel("Micro-Batch Size")
    ax.set_ylabel("Overlap Efficiency (%)")
    ax.set_title("Pipelining Overlap Efficiency vs. Micro-Batch Size")
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

    logger.info("Saved %d plots to %s", len(saved), out_path)
    return saved
