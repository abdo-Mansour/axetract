"""CLI entry point for the Axetract speed benchmark.

Usage::

    # Smoke test (recommended first run)
    python -m benchmarks.run --backend vllm --batch-sizes 1 --repeats 1 --warmup 1 --no-mb-sweep

    # Default sweep (still substantial — each doc is multi-chunk LLM work)
    python -m benchmarks.run --backend vllm --device gpu --batch-sizes 1,2,4 --no-mb-sweep
    python -m benchmarks.run --backend hf --device cpu --batch-sizes 1 --repeats 1 --warmup 0 --no-mb-sweep
    python -m benchmarks.run --backend vllm --device gpu --no-mb-sweep --plots

    # Skip the pruner entirely (no chunk-level LLM calls before extraction)
    python -m benchmarks.run --backend vllm --batch-sizes 1 --repeats 1 --warmup 1 --no-mb-sweep --pruner skip

    # Compare with/without pruner in a single run (each config is measured twice)
    python -m benchmarks.run --backend vllm --batch-sizes 1 --repeats 1 --warmup 1 --no-mb-sweep --pruner both

The benchmark loads a corpus of local HTML files (as ``AXESample`` objects),
runs timed extraction passes across a sweep of batch sizes and micro-batch
sizes, collects per-stage timing events and resource statistics, and writes
timestamped JSON + Markdown results to ``benchmarks/results/``.

With ``--pruner both``, every (batch_size, micro_batch_size) configuration
is measured twice — once with the pruner enabled and once with it skipped —
so the latency/throughput contribution of the chunk-level pruner LoRA calls
is directly visible in the same report.

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
from typing import List, Optional, Tuple

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


def _import_schema(dotted_path: str):
    """Import a pydantic ``BaseModel`` subclass from a dotted path.

    Args:
        dotted_path (str): e.g. ``"benchmarks.harness.Product"`` or
            ``"my_app.schemas:Product"`` (colon form also accepted).

    Returns:
        Type[BaseModel]: The resolved schema class.

    Raises:
        ValueError: If the path is malformed or the attribute is not a
            ``BaseModel`` subclass.
        ImportError: If the module cannot be imported.
        AttributeError: If the attribute is not found in the module.
    """
    if ":" in dotted_path:
        module_name, attr = dotted_path.split(":", 1)
    elif "." in dotted_path:
        module_name, attr = dotted_path.rsplit(".", 1)
    else:
        raise ValueError(
            f"Invalid schema path {dotted_path!r} — expected "
            "'module.attr' or 'module:attr'."
        )

    import importlib
    from pydantic import BaseModel

    module = importlib.import_module(module_name)
    schema_cls = getattr(module, attr)
    if not (isinstance(schema_cls, type) and issubclass(schema_cls, BaseModel)):
        raise ValueError(
            f"{dotted_path!r} does not resolve to a pydantic BaseModel subclass."
        )
    return schema_cls


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
        help="Use only the first micro-batch size (no sweep).",
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
        "--limit",
        type=int,
        default=None,
        help="Maximum number of pages to load from the corpus (default: all).",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help=(
            "Dotted path to a pydantic BaseModel class to use as the extraction "
            "schema (e.g. 'benchmarks.harness.Product'). "
            "Defaults to the built-in Product schema (name, price, rating)."
        ),
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
        "--html",
        action="store_true",
        help="Generate a self-contained HTML report (no extra dependencies).",
    )
    parser.add_argument(
        "--pgf",
        action="store_true",
        help="Also export HTML report charts as PGF files (requires matplotlib + LaTeX).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Dump cProfile stats (for deep-dive profiling).",
    )
    parser.add_argument(
        "--pruner",
        choices=["use", "skip", "both"],
        default="use",
        help=(
            "How to run the pruner stage. "
            "'use' (default): pruner is enabled normally (one LLM call per chunk). "
            "'skip': pruner stage is disabled (raw preprocessor HTML goes to extractor). "
            "'both': each batch/micro-batch configuration is measured twice — once with "
            "the pruner enabled and once with it skipped — so the pruner's contribution "
            "is visible in the same report."
        ),
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
        mb_sizes = _parse_int_list(args.micro_batch_sizes)[:1]
    else:
        mb_sizes = _parse_int_list(args.micro_batch_sizes)

    if not batch_sizes:
        logger.error("No batch sizes provided.")
        return 2
    if not mb_sizes:
        logger.error("No micro-batch sizes provided.")
        return 2

    # ── Load corpus ──
    from benchmarks.harness import DEFAULT_SCHEMA

    if args.schema:
        schema_model = _import_schema(args.schema)
    else:
        schema_model = DEFAULT_SCHEMA
    logger.info("Loading corpus from %s ...", args.corpus)
    samples = load_corpus(args.corpus, schema_model=schema_model)
    if args.limit is not None:
        samples = samples[: args.limit]
    logger.info("Corpus: %d samples", len(samples))

    # Rough work estimate so multi-hour sweeps are not a surprise.
    # Pruner issues ~1 LLM call per HTML chunk; Amazon-scale pages are ~8–15.
    n_configs = len(mb_sizes) * len(batch_sizes)
    n_pruner_variants = {"use": 1, "skip": 1, "both": 2}[args.pruner]
    est_docs = 0
    for bs in batch_sizes:
        est_docs += min(args.warmup, bs) + args.repeats * bs
    # Multiply by mb sweep (each config re-runs the same docs)
    # and by pruner variants ("both" doubles the work).
    est_docs *= len(mb_sizes) * n_pruner_variants
    logger.info(
        "Sweep plan: %d batch/mb config(s) × %d pruner variant(s) = %d total, "
        "batch_sizes=%s, micro_batch_sizes=%s, pruner=%s, repeats=%d, "
        "warmup=%d → ~%d document-pass(es). Each document is multi-chunk "
        "(pruner LLM call per chunk + 1 extract). "
        "Start with --batch-sizes 1 --repeats 1 --warmup 1 --no-mb-sweep if unsure.",
        n_configs,
        n_pruner_variants,
        n_configs * n_pruner_variants,
        batch_sizes,
        mb_sizes,
        args.pruner,
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
    # The pipeline is built with the pruner enabled (the default) so that
    # when ``--pruner both`` is requested we can simply flip ``skip`` on
    # the existing pruner instance instead of rebuilding the LLM client.
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

    # Determine which pruner variants to sweep.
    # "use"  -> only [(True,  "")]
    # "skip" -> only [(False, "")]
    # "both" -> [(True,  "+pruner"), (False, "-pruner")]
    if args.pruner == "use":
        pruner_variants: List[Tuple[bool, str]] = [(True, "")]
    elif args.pruner == "skip":
        pruner_variants = [(False, "")]
    else:  # "both"
        pruner_variants = [(True, " [+pruner]"), (False, " [-pruner]")]
    n_pruner_variants = len(pruner_variants)

    # ── Sweep ──
    config_idx = 0
    total_configs = n_configs * n_pruner_variants
    for mb_size in mb_sizes:
        # Only update the micro-batch size attribute — no need to rebuild
        # the LLM engine (vLLM / HF model) for each value.
        pipeline._micro_batch_size = mb_size

        for bs in batch_sizes:
            for pruner_enabled, pruner_tag in pruner_variants:
                # Toggle the pruner at runtime — no LLM rebuild needed.
                pipeline._pruner.skip = not pruner_enabled

                config_idx += 1
                label = (
                    f"{args.backend}/{args.device} b={bs} mb={mb_size}"
                    f"{pruner_tag}"
                )
                logger.info(
                    "Running config %d/%d: %s (pruner=%s, repeats=%d, warmup=%d)",
                    config_idx,
                    total_configs,
                    label,
                    "on" if pruner_enabled else "off",
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
                    prof_path = out_dir / (
                        f"{base_name}_b{bs}_mb{mb_size}"
                        f"{'_noprune' if not pruner_enabled else ''}.prof"
                    )
                    profiler.dump_stats(str(prof_path))
                    logger.info("Profile saved to %s", prof_path)

                # Record the pruner setting alongside the raw metrics.
                raw["pruner_enabled"] = pruner_enabled

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
        "schema": (
            args.schema
            if args.schema
            else f"{schema_model.__module__}.{schema_model.__name__}"
        ),
        "corpus_size": len(samples),
        "batch_sizes": batch_sizes,
        "micro_batch_sizes": mb_sizes,
        "pruner_mode": args.pruner,
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

    # ── HTML report ──
    html_path: Optional[Path] = None
    pgf_dir: Optional[Path] = (out_dir / f"{base_name}_pgf") if args.pgf else None
    pgf_written: List[Path] = []
    if args.html:
        from benchmarks.html_report import build_report

        html_path = out_dir / f"{base_name}.html"
        pgf_written = build_report([json_payload], [json_payload.get("timestamp", "")], html_path, pgf_dir=pgf_dir)
        logger.info("HTML report written to %s", html_path)
        # PGF files already exported by build_report when pgf_dir was given.
        if pgf_dir:
            if pgf_written:
                logger.info("PGF charts written to %s (%d files)", pgf_dir, len(pgf_written))
            else:
                logger.warning(
                    "PGF export produced no files — see warnings above "
                    "(common cause: missing TeX engine such as xelatex)."
                )
            pgf_dir = None  # Prevent double export below.
    # ── Standalone PGF export (--pgf without --html) ──
    if pgf_dir is not None:
        from benchmarks.html_report import build_report

        standalone_html = html_path or (out_dir / f"{base_name}.html")
        pgf_written = build_report([json_payload], [json_payload.get("timestamp", "")], standalone_html, pgf_dir=pgf_dir)
        if pgf_written:
            logger.info("PGF charts written to %s (%d files)", pgf_dir, len(pgf_written))
        else:
            logger.warning(
                "PGF export produced no files — see warnings above "
                "(common cause: missing TeX engine such as xelatex)."
            )

    # ── Print summary to console ──
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(to_markdown_full(all_metrics, config_labels))
    print(f"\nResults: {json_path}")
    print(f"Report:  {md_path}")
    if html_path:
        print(f"HTML:    {html_path}")
    if args.pgf:
        pgf_out_dir = out_dir / (base_name + "_pgf")
        if pgf_written:
            print(f"PGF:     {pgf_out_dir}/ ({len(pgf_written)} files)")
        else:
            print(f"PGF:     {pgf_out_dir}/ (no files — see warnings)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
