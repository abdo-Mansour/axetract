"""CLI entry point for the Axetract speed benchmark.

Usage::

    # Smoke test (recommended first run)
    python -m benchmarks.run --backend vllm --batch-sizes 1 --repeats 1 --warmup 1 --no-mb-sweep

    # Default sweep (still substantial — each doc is multi-chunk LLM work)
    python -m benchmarks.run --backend vllm --device gpu --batch-sizes 1,2,4 --no-mb-sweep
    python -m benchmarks.run --backend hf --device cpu --batch-sizes 1 --repeats 1 --warmup 0 --no-mb-sweep
    python -m benchmarks.run --backend vllm --device gpu --no-mb-sweep --plots

The benchmark loads a corpus of local HTML files (as ``AXESample`` objects),
runs timed extraction passes across a sweep of batch sizes and micro-batch
sizes, collects per-stage timing events and resource statistics, and writes
timestamped JSON + Markdown results to ``benchmarks/results/``.

.. note::
   Defaults are intentionally modest.  Each document is cleaned, split into
   many chunks, and the **pruner runs one LLM call per chunk** before a final
   extraction call.  A full Cartesian sweep of large batch sizes × micro-batch
   sizes × repeats can mean tens of thousands of LLM calls and multi-hour runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path so ``benchmarks`` is importable
# even when invoked directly as ``python benchmarks/run.py``.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmarks.harness import (
    StageCollector,
    build_pipeline,
    load_corpus,
    run_config,
)
from benchmarks.metrics import compute_run_metrics
from benchmarks.report import to_markdown_full, to_plots

logger = logging.getLogger("benchmarks")


def _parse_int_list(s: str) -> List[int]:
    """Parse a comma-separated string of integers."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    """Run the benchmark CLI.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        int: Exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        prog="benchmarks.run",
        description="Axetract speed benchmark (latency, throughput, per-stage occupancy).",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "hf"],
        default="vllm",
        help="LLM backend to use (default: vllm).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Device to run on (default: gpu). CPU forces the HF backend.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4",
        help=(
            "Comma-separated input batch sizes to sweep (default: 1,2,4). "
            "Larger values multiply full pipeline work; each doc is multi-chunk."
        ),
    )
    parser.add_argument(
        "--micro-batch-sizes",
        type=str,
        default="4",
        help=(
            "Comma-separated micro-batch sizes to sweep (default: 4). "
            "Use --micro-batch-sizes 1,4,8 to measure pipelining overlap."
        ),
    )
    parser.add_argument(
        "--no-mb-sweep",
        action="store_true",
        help="Fix micro-batch size at 4 (same as the default single value).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repetitions per config (default: 3).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup samples before timing (default: 1).",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/benchmark",
        help="Directory containing benchmark HTML files (default: data/benchmark).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Extraction query for all samples (default: a generic product query).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="benchmarks/results",
        help="Output directory for results (default: benchmarks/results).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate matplotlib plots (requires matplotlib).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Dump cProfile stats (for deep-dive profiling).",
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

    # ── Parse sweep parameters ──
    batch_sizes = _parse_int_list(args.batch_sizes)
    if args.no_mb_sweep:
        mb_sizes = [4]
    else:
        mb_sizes = _parse_int_list(args.micro_batch_sizes)

    if not batch_sizes:
        logger.error("No batch sizes provided.")
        return 2
    if not mb_sizes:
        logger.error("No micro-batch sizes provided.")
        return 2

    # ── Load corpus ──
    from benchmarks.harness import DEFAULT_QUERY

    query = args.query or DEFAULT_QUERY
    logger.info("Loading corpus from %s ...", args.corpus)
    samples = load_corpus(args.corpus, query=query)
    logger.info("Corpus: %d samples", len(samples))

    # Rough work estimate so multi-hour sweeps are not a surprise.
    # Pruner issues ~1 LLM call per HTML chunk; Amazon-scale pages are ~8–15.
    n_configs = len(mb_sizes) * len(batch_sizes)
    est_docs = 0
    for bs in batch_sizes:
        est_docs += min(args.warmup, bs) + args.repeats * bs
    # Multiply by mb sweep (each config re-runs the same docs).
    est_docs *= len(mb_sizes)
    logger.info(
        "Sweep plan: %d config(s), batch_sizes=%s, micro_batch_sizes=%s, "
        "repeats=%d, warmup=%d → ~%d document-pass(es). "
        "Each document is multi-chunk (pruner LLM call per chunk + 1 extract). "
        "Start with --batch-sizes 1 --repeats 1 --warmup 1 --no-mb-sweep if unsure.",
        n_configs,
        batch_sizes,
        mb_sizes,
        args.repeats,
        args.warmup,
        est_docs,
    )

    # ── Output directory ──
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"bench_{args.backend}_{args.device}_{timestamp}"

    all_raw: List[dict] = []
    all_metrics: List[dict] = []
    config_labels: List[str] = []

    # ── Build pipeline once (the LLM client is expensive to construct) ──
    logger.info(
        "Building pipeline: backend=%s device=%s",
        args.backend, args.device,
    )
    pipeline = build_pipeline(
        backend=args.backend,
        device=args.device,
        micro_batch_size=mb_sizes[0],
    )
    collector = StageCollector()

    # ── Sweep ──
    config_idx = 0
    for mb_size in mb_sizes:
        # Only update the micro-batch size attribute — no need to rebuild
        # the LLM engine (vLLM / HF model) for each value.
        pipeline._micro_batch_size = mb_size

        for bs in batch_sizes:
            config_idx += 1
            label = f"{args.backend}/{args.device} b={bs} mb={mb_size}"
            logger.info(
                "Running config %d/%d: %s (repeats=%d, warmup=%d)",
                config_idx,
                n_configs,
                label,
                args.repeats,
                args.warmup,
            )

            if args.profile:
                import cProfile
                profiler = cProfile.Profile()
                profiler.enable()

            raw = run_config(
                pipeline=pipeline,
                samples=samples,
                batch_size=bs,
                repeats=args.repeats,
                warmup=args.warmup,
                collector=collector,
            )

            if args.profile:
                profiler.disable()
                prof_path = out_dir / f"{base_name}_b{bs}_mb{mb_size}.prof"
                profiler.dump_stats(str(prof_path))
                logger.info("Profile saved to %s", prof_path)

            metrics = compute_run_metrics(raw)

            all_raw.append(raw)
            all_metrics.append(metrics)
            config_labels.append(label)

            logger.info(
                "  -> p50=%.3fs  docs/s=%.1f  overlap=%.1f%%  success=%.1f%%",
                metrics["latency_p50_s"],
                metrics["docs_per_s"],
                metrics["overlap_efficiency"] * 100,
                metrics["success_rate"] * 100,
            )

    # ── Write JSON ──
    json_path = out_dir / f"{base_name}.json"
    json_payload = {
        "timestamp": timestamp,
        "backend": args.backend,
        "device": args.device,
        "corpus": args.corpus,
        "query": query,
        "corpus_size": len(samples),
        "batch_sizes": batch_sizes,
        "micro_batch_sizes": mb_sizes,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "configs": config_labels,
        "metrics": all_metrics,
        "raw": all_raw,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, default=str)
    logger.info("Raw JSON results written to %s", json_path)

    # ── Write Markdown ──
    md_path = out_dir / f"{base_name}.md"
    md_report = to_markdown_full(all_metrics, config_labels)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    logger.info("Markdown report written to %s", md_path)

    # ── Plots ──
    if args.plots:
        plot_dir = out_dir / f"{base_name}_plots"
        plot_paths = to_plots(all_metrics, config_labels, outdir=plot_dir)
        if plot_paths:
            logger.info("Plots written to %s", plot_dir)

    # ── Print summary to console ──
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(to_markdown_full(all_metrics, config_labels))
    print(f"\nResults: {json_path}")
    print(f"Report:  {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
