"""Pure metric-computation functions for the Axetract benchmark.

All functions here are side-effect-free and operate on the raw run-record
dictionaries produced by :func:`benchmarks.harness.run_config`.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Tuple

# Stage names in pipeline order.  ``setup`` accounts for the GPU memory
# reclaim (``gc.collect()`` + ``torch.cuda.empty_cache()``) that runs
# between ``prune`` and ``extract`` in sequential mode — without it, the
# per-stage occupancy totals under-report wall-clock and the Gantt chart
# has an unexplained gap.
STAGE_NAMES = ("preprocess", "prune", "setup", "extract", "postprocess")


def percentiles(values: List[float], ps: Tuple[float, ...] = (50, 90, 99)) -> Dict[str, float]:
    """Compute percentiles of a list of values.

    Uses linear interpolation between closest ranks (same method as numpy's
    default).

    Args:
        values (List[float]): Numeric values.
        ps (Tuple[float, ...]): Percentile points to compute (0-100).

    Returns:
        Dict[str, float]: Mapping like ``{"p50": ..., "p90": ...}``.
    """
    if not values:
        return {f"p{int(p)}": 0.0 for p in ps}

    s = sorted(values)
    n = len(s)
    result: Dict[str, float] = {}
    for p in ps:
        if n == 1:
            result[f"p{int(p)}"] = s[0]
            continue
        # Linear interpolation.
        rank = (p / 100.0) * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        frac = rank - lo
        result[f"p{int(p)}"] = s[lo] + (s[hi] - s[lo]) * frac
    return result


def _mean(values: List[float]) -> float:
    """Arithmetic mean, 0 for empty list."""
    return sum(values) / len(values) if values else 0.0


def _median(values: List[float]) -> float:
    """Median, 0 for empty list."""
    return statistics.median(values) if values else 0.0


def _iqr(values: List[float]) -> float:
    """Interquartile range (p75 - p25), 0 for empty list."""
    if not values:
        return 0.0
    p = percentiles(values, ps=(25, 75))
    return p["p75"] - p["p25"]


# ──────────────────────────────────────────────────────────────────────
# Per-stage occupancy from stage events
# ──────────────────────────────────────────────────────────────────────


def compute_stage_occupancy(
    stage_events: List[Tuple[str, str, Optional[int], float]],
) -> Dict[str, float]:
    """Compute total busy time (seconds) per stage from enter/exit events.

    Matches enter/exit pairs per (stage, mb_index).  In pipelined mode the
    sum across micro-batches gives the stage's total occupancy.  The sum of
    all stage occupancies minus the wall-clock is the overlap gain.

    Args:
        stage_events: List of ``(stage, event, mb_index, timestamp)`` tuples.

    Returns:
        Dict[str, float]: ``{stage: occupancy_seconds}`` for each stage.
    """
    # Group timestamps by (stage, mb_index) -> {enter: t, exit: t}
    pairs: Dict[Tuple[str, Optional[int]], Dict[str, float]] = {}
    for stage, event, mb_index, t in stage_events:
        key = (stage, mb_index)
        if key not in pairs:
            pairs[key] = {}
        pairs[key][event] = t

    occupancy: Dict[str, float] = {s: 0.0 for s in STAGE_NAMES}
    for (stage, _mb), evts in pairs.items():
        if "enter" in evts and "exit" in evts:
            dur = evts["exit"] - evts["enter"]
            if dur > 0:
                occupancy[stage] += dur
    return occupancy


def compute_overlap_efficiency(
    wall_s: float,
    stage_occupancy: Dict[str, float],
) -> float:
    """Compute pipelining overlap efficiency.

    .. math::
        \\text{overlap} = 1 - \\frac{t_{\\text{wall}}}{\\sum_i t_{\\text{stage}_i}}

    A value near 0 means no overlap (stages ran sequentially); higher values
    mean more overlap.  Can be negative if wall-clock exceeds the sum (e.g.,
    due to thread overhead or measurement noise).

    Args:
        wall_s (float): End-to-end wall-clock seconds.
        stage_occupancy (Dict[str, float]): Per-stage occupancy seconds.

    Returns:
        float: Overlap efficiency in [-inf, 1).
    """
    total_occupancy = sum(stage_occupancy.values())
    if total_occupancy <= 0:
        return 0.0
    return 1.0 - (wall_s / total_occupancy)


# ──────────────────────────────────────────────────────────────────────
# Run-level metrics
# ──────────────────────────────────────────────────────────────────────


def compute_run_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Compute aggregated metrics for a single benchmark configuration.

    Args:
        raw (dict): Raw run record from :func:`benchmarks.harness.run_config`.

    Returns:
        dict: Aggregated metrics including latency, throughput, per-stage
        occupancy, overlap efficiency, success rate, and resource usage.
    """
    per_repeat = raw["per_repeat"]
    batch_size = raw["batch_size"]
    repeats = raw["repeats"]

    wall_times = [r["wall_s"] for r in per_repeat]
    lat_pct = percentiles(wall_times, ps=(50, 90, 99))
    mean_wall = _mean(wall_times)
    median_wall = _median(wall_times)
    iqr_wall = _iqr(wall_times)

    # Throughput (based on mean wall time).
    docs_per_s = batch_size / mean_wall if mean_wall > 0 else 0.0
    input_tokens_total = raw["input_tokens_total"]
    output_tokens_total = raw["output_tokens_total"]
    tokens_per_s = (
        (input_tokens_total + output_tokens_total) / mean_wall
        if mean_wall > 0
        else 0.0
    )

    # Real LLM token usage (reported by the backend), summed across repeats.
    llm_prompt_tokens_total = raw.get("llm_prompt_tokens_total", 0)
    llm_completion_tokens_total = raw.get("llm_completion_tokens_total", 0)
    pruner_prompt_tokens_total = raw.get("pruner_prompt_tokens_total", 0)
    pruner_completion_tokens_total = raw.get("pruner_completion_tokens_total", 0)
    extractor_prompt_tokens_total = raw.get("extractor_prompt_tokens_total", 0)
    extractor_completion_tokens_total = raw.get("extractor_completion_tokens_total", 0)
    llm_tokens_total = llm_prompt_tokens_total + llm_completion_tokens_total

    # Real LLM token rates (tokens the model actually processed / mean wall).
    # These reflect post-preprocessing / post-pruning token counts, unlike
    # ``tokens_per_s`` which uses the char/4 heuristic over raw HTML.
    llm_tokens_per_s = llm_tokens_total / mean_wall if mean_wall > 0 else 0.0
    llm_prompt_tokens_per_s = (
        llm_prompt_tokens_total / mean_wall if mean_wall > 0 else 0.0
    )
    llm_completion_tokens_per_s = (
        llm_completion_tokens_total / mean_wall if mean_wall > 0 else 0.0
    )
    # Output-token throughput (completion tokens / wall) — the rate at which
    # the model generates tokens, useful for cost/latency estimation.
    llm_output_tokens_per_s = llm_completion_tokens_per_s

    # Total wall time across all repeats (for cost estimation).
    total_wall_s = sum(wall_times)

    # Time per page / per 1K input tokens.
    time_per_page = mean_wall / batch_size if batch_size > 0 else 0.0
    time_per_1k_input = (
        mean_wall / (input_tokens_total / 1000.0) if input_tokens_total > 0 else 0.0
    )

    # Success rate.
    success_count = raw["success_count"]
    total_count = raw["total_count"]
    success_rate = success_count / total_count if total_count > 0 else 0.0

    # Per-stage occupancy (averaged across repeats).
    stage_occupancies: List[Dict[str, float]] = []
    overlap_effs: List[float] = []
    for r in per_repeat:
        occ = compute_stage_occupancy(r["stage_events"])
        stage_occupancies.append(occ)
        overlap_effs.append(compute_overlap_efficiency(r["wall_s"], occ))

    avg_stage_occupancy: Dict[str, float] = {}
    for stage in STAGE_NAMES:
        vals = [occ.get(stage, 0.0) for occ in stage_occupancies]
        avg_stage_occupancy[stage] = _mean(vals)

    mean_overlap = _mean(overlap_effs) if overlap_effs else 0.0

    # Resources.
    gpu = raw.get("gpu", {})
    peak_rss_mb = raw.get("peak_rss_kb", 0) / 1024.0

    return {
        "batch_size": batch_size,
        "micro_batch_size": raw["micro_batch_size"],
        "repeats": repeats,
        # Latency (seconds)
        "latency_p50_s": lat_pct["p50"],
        "latency_p90_s": lat_pct["p90"],
        "latency_p99_s": lat_pct["p99"],
        "latency_mean_s": mean_wall,
        "latency_median_s": median_wall,
        "latency_iqr_s": iqr_wall,
        # Throughput
        "docs_per_s": docs_per_s,
        "tokens_per_s": tokens_per_s,
        "time_per_page_s": time_per_page,
        "time_per_1k_input_tokens_s": time_per_1k_input,
        # Success
        "success_rate": success_rate,
        "success_count": success_count,
        "total_count": total_count,
        # Per-stage
        "stage_occupancy_s": avg_stage_occupancy,
        "overlap_efficiency": mean_overlap,
        # Cold start
        "warmup_s": raw.get("warmup_s", 0.0),
        # Resources
        "peak_vram_mb": gpu.get("peak_vram_mb"),
        "mean_gpu_util_pct": gpu.get("mean_gpu_util_pct"),
        "peak_rss_mb": peak_rss_mb,
        # Raw token totals (for reference)
        "input_tokens_total": input_tokens_total,
        "output_tokens_total": output_tokens_total,
        # Real LLM token usage (reported by the backend)
        "llm_prompt_tokens_total": llm_prompt_tokens_total,
        "llm_completion_tokens_total": llm_completion_tokens_total,
        "llm_tokens_total": llm_tokens_total,
        "pruner_prompt_tokens_total": pruner_prompt_tokens_total,
        "pruner_completion_tokens_total": pruner_completion_tokens_total,
        "extractor_prompt_tokens_total": extractor_prompt_tokens_total,
        "extractor_completion_tokens_total": extractor_completion_tokens_total,
        # Real LLM token rates (post-preprocessing tokens / mean wall)
        "llm_tokens_per_s": llm_tokens_per_s,
        "llm_prompt_tokens_per_s": llm_prompt_tokens_per_s,
        "llm_completion_tokens_per_s": llm_completion_tokens_per_s,
        "llm_output_tokens_per_s": llm_output_tokens_per_s,
        # Totals (for cost estimation)
        "total_wall_s": total_wall_s,
        "total_repeats": repeats,
    }


# ──────────────────────────────────────────────────────────────────────
# Size-bucket metrics (for latency-vs-size analysis)
# ──────────────────────────────────────────────────────────────────────


def estimate_input_tokens(html: str) -> int:
    """Estimate input token count for an HTML string (char/4 heuristic)."""
    return max(1, len(html) // 4)


def bucket_for_tokens(tokens: int) -> str:
    """Map a token count to a size bucket label."""
    if tokens < 2000:
        return "small (<2K)"
    if tokens < 10000:
        return "medium (2K-10K)"
    return "large (>10K)"


def compute_size_bucket_metrics(
    samples_info: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Group per-sample timing info by input-size bucket.

    Args:
        samples_info: List of dicts, each with ``input_tokens`` and
            ``latency_s`` keys.

    Returns:
        Dict[str, dict]: ``{bucket: {count, mean_latency_s, p50, p90}}``.
    """
    buckets: Dict[str, List[float]] = {}
    for info in samples_info:
        bucket = bucket_for_tokens(info["input_tokens"])
        buckets.setdefault(bucket, []).append(info["latency_s"])

    result: Dict[str, Dict[str, Any]] = {}
    for bucket, lats in buckets.items():
        pct = percentiles(lats, ps=(50, 90))
        result[bucket] = {
            "count": len(lats),
            "mean_latency_s": _mean(lats),
            "p50_s": pct["p50"],
            "p90_s": pct["p90"],
        }
    return result
