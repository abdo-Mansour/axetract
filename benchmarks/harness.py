"""Benchmark harness for the Axetract pipeline.

Loads a corpus of local HTML files as :class:`AXESample` objects, builds the
pipeline for a given backend/device, and runs timed extraction passes while
collecting per-stage timing events (via the ``on_stage`` callback) and resource
statistics (GPU utilization, peak VRAM, peak RSS).

This module is *speed-only*. It does not evaluate extraction quality — that is
a separate "evaluation" effort.
"""

from __future__ import annotations

import logging
import resource
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from axetract.data_types import AXESample, Status
from axetract.pipeline import AXEPipeline

logger = logging.getLogger(__name__)

# Default extraction query used when a corpus sample has no explicit query.
DEFAULT_QUERY = "Extract the product name, price, and key specifications."

# Approximate chars-per-token for the char/4 heuristic.
CHARS_PER_TOKEN = 4


# ──────────────────────────────────────────────────────────────────────
# Corpus loading
# ──────────────────────────────────────────────────────────────────────


def load_corpus(
    html_dir: str | Path = "data/benchmark",
    query: str = DEFAULT_QUERY,
) -> List[AXESample]:
    """Load every ``*.html`` file in *html_dir* as an :class:`AXESample`.

    Files with a ``:Zone.Identifier`` suffix (Windows download metadata) are
    skipped.  Each sample's ``content`` is the raw HTML text, ``is_content_url``
    is ``False``, and ``query`` is set to *query*.

    Args:
        html_dir (str | Path): Directory containing ``.html`` files.
        query (str): Extraction query to attach to every sample.

    Returns:
        List[AXESample]: One sample per HTML file, sorted by filename.
    """
    html_path = Path(html_dir)
    if not html_path.is_dir():
        raise FileNotFoundError(f"Benchmark corpus directory not found: {html_path}")

    samples: List[AXESample] = []
    for p in sorted(html_path.iterdir()):
        if not p.is_file():
            continue
        if p.name.endswith(":Zone.Identifier"):
            continue
        if p.suffix.lower() not in (".html", ".htm"):
            continue
        content = p.read_text(encoding="utf-8", errors="replace")
        samples.append(
            AXESample(
                id=p.stem,
                content=content,
                is_content_url=False,
                query=query,
            )
        )

    if not samples:
        raise FileNotFoundError(
            f"No .html files found in corpus directory: {html_path}"
        )

    logger.info("Loaded %d HTML samples from %s", len(samples), html_path)
    return samples


# ──────────────────────────────────────────────────────────────────────
# Pipeline construction
# ──────────────────────────────────────────────────────────────────────


def build_pipeline(
    backend: str = "vllm",
    device: str = "gpu",
    micro_batch_size: int = 4,
    llm_config: Optional[Dict[str, Any]] = None,
) -> AXEPipeline:
    """Build an :class:`AXEPipeline` for the requested backend and device.

    Args:
        backend (str): ``"vllm"`` or ``"hf"``.
        device (str): ``"gpu"`` or ``"cpu"``.  CPU runs force the HuggingFace
            backend with ``device_map="cpu"``.
        micro_batch_size (int): Micro-batch size for pipelined execution.
        llm_config (Optional[dict]): Override the default LLM config.

    Returns:
        AXEPipeline: A configured pipeline (without an ``on_stage`` hook —
        the caller attaches one via :meth:`AXEPipeline._on_stage` assignment
        or by constructing with ``on_stage=``).

    Raises:
        ValueError: If *backend* or *device* are invalid, or if CPU is
            requested with the vLLM backend (vLLM CPU support is experimental
            and not credible for benchmarking).
    """
    if backend not in ("vllm", "hf"):
        raise ValueError(f"Unknown backend: {backend!r} (expected 'vllm' or 'hf')")
    if device not in ("cpu", "gpu"):
        raise ValueError(f"Unknown device: {device!r} (expected 'cpu' or 'gpu')")

    if device == "cpu" and backend == "vllm":
        raise ValueError(
            "vLLM CPU backend is experimental and not supported for benchmarking. "
            "Use --backend hf --device cpu instead."
        )

    use_vllm = backend == "vllm"
    config = llm_config  # from_config applies its own default if None

    # For HF on CPU, override device_map to force CPU placement.
    if backend == "hf" and device == "cpu":
        config = dict(config) if config else {}
        model_kwargs = dict(config.get("model_kwargs", {}))
        model_kwargs["device_map"] = "cpu"
        config["model_kwargs"] = model_kwargs

    pipeline = AXEPipeline.from_config(llm_config=config, use_vllm=use_vllm)
    pipeline._micro_batch_size = micro_batch_size
    return pipeline


# ──────────────────────────────────────────────────────────────────────
# Stage timing collector
# ──────────────────────────────────────────────────────────────────────


class StageCollector:
    """Callable that accumulates ``on_stage`` events for later analysis.

    Designed to be passed as the ``on_stage`` callback to
    :class:`AXEPipeline`.  Thread-safe (pipelined mode emits from multiple
    threads).

    Attributes:
        records (List[Tuple[str, str, Optional[int], float]]): List of
            ``(stage, event, mb_index, timestamp)`` tuples.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.records: List[Tuple[str, str, Optional[int], float]] = []

    def __call__(
        self, stage: str, event: str, mb_index: Optional[int], t: float
    ) -> None:
        with self._lock:
            self.records.append((stage, event, mb_index, t))

    def reset(self) -> None:
        """Clear all accumulated records."""
        with self._lock:
            self.records.clear()


# ──────────────────────────────────────────────────────────────────────
# GPU resource sampler
# ──────────────────────────────────────────────────────────────────────


class GPUResourceSampler:
    """Polls GPU utilization in a background thread during a benchmark run.

    Only active when CUDA is available.  Otherwise it's a no-op.

    Attributes:
        util_samples (List[float]): Sampled utilization percentages.
        peak_vram_bytes (int): Peak allocated VRAM across the run.
    """

    def __init__(self) -> None:
        self.util_samples: List[float] = []
        self.peak_vram_bytes: int = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._torch = None
        try:
            import torch

            if torch.cuda.is_available():
                self._torch = torch
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        """Whether GPU monitoring is active."""
        return self._torch is not None

    def start(self) -> None:
        """Begin sampling GPU utilization and tracking peak VRAM."""
        if not self.available:
            return
        self._stop.clear()
        self.util_samples.clear()
        self.peak_vram_bytes = 0
        self._torch.cuda.reset_peak_memory_stats()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and record final peak VRAM."""
        if not self.available:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.peak_vram_bytes = self._torch.cuda.max_memory_allocated()

    def _poll(self) -> None:
        """Background loop sampling utilization at 10ms intervals."""
        assert self._torch is not None
        while not self._stop.is_set():
            try:
                util = self._torch.cuda.utilization()
                if util is not None:
                    self.util_samples.append(float(util))
            except Exception:
                pass
            time.sleep(0.01)


# ──────────────────────────────────────────────────────────────────────
# Timed run
# ──────────────────────────────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """Rough token estimate using the char/4 heuristic.

    Args:
        text (str): Input or output text.

    Returns:
        int: Approximate token count.
    """
    return max(1, len(text) // CHARS_PER_TOKEN)


def _peak_rss_kb() -> int:
    """Return peak resident set size in kilobytes for the current process."""
    # ru_maxrss is in KB on Linux, in bytes on macOS — we assume Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def run_config(
    pipeline: AXEPipeline,
    samples: List[AXESample],
    batch_size: int,
    repeats: int = 5,
    warmup: int = 3,
    collector: Optional[StageCollector] = None,
) -> Dict[str, Any]:
    """Run a single benchmark configuration and return raw timing records.

    Args:
        pipeline (AXEPipeline): A pipeline instance.  If *collector* is given,
            ``pipeline._on_stage`` is set to it before running.
        samples (List[AXESample]): Corpus to extract from.
        batch_size (int): Number of samples per timed batch.
        repeats (int): Number of timed repetitions.
        warmup (int): Number of warmup samples (discarded).
        collector (Optional[StageCollector]): Stage-event collector.  If
            provided, it is attached to the pipeline and reset before each
            repeat.

    Returns:
        dict: Raw run record with keys:
            ``backend``, ``device``, ``batch_size``, ``micro_batch_size``,
            ``repeats``, ``warmup_s``, ``per_repeat`` (list of per-repeat
            dicts), ``input_tokens_total``, ``output_tokens_total``,
            ``success_count``, ``total_count``, ``peak_rss_kb``,
            ``gpu`` (dict with ``peak_vram_mb``, ``mean_gpu_util_pct`` or
            ``None``).
    """
    # Attach collector if provided.
    if collector is not None:
        pipeline._on_stage = collector

    # Build the batch (cycled to fill batch_size if corpus is smaller).
    if len(samples) >= batch_size:
        batch = samples[:batch_size]
    else:
        # Cycle to fill the batch.
        batch = list(samples) * (batch_size // len(samples))
        remainder = batch_size % len(samples)
        batch.extend(samples[:remainder])

    # Input token estimate (char/4) — computed once, constant across repeats.
    input_tokens_total = sum(_estimate_tokens(s.content) for s in batch)

    # ── Warmup ──
    warmup_batch = batch[: min(warmup, len(batch))] if warmup > 0 else []
    warmup_s = 0.0
    if warmup_batch:
        t0 = time.perf_counter()
        pipeline.extract_batch(list(warmup_batch))
        warmup_s = time.perf_counter() - t0

    # ── GPU sampler ──
    gpu_sampler = GPUResourceSampler()

    per_repeat: List[Dict[str, Any]] = []
    success_count = 0
    total_count = 0
    output_tokens_total = 0

    for rep in range(repeats):
        if collector is not None:
            collector.reset()

        gpu_sampler.start()
        t_start = time.perf_counter()
        results = pipeline.extract_batch(list(batch))
        t_end = time.perf_counter()
        gpu_sampler.stop()

        wall_s = t_end - t_start
        stage_events = list(collector.records) if collector else []

        # Count successes and estimate output tokens.
        rep_success = 0
        rep_output_tokens = 0
        for r in results:
            total_count += 1
            if r.status == Status.SUCCESS:
                success_count += 1
                rep_success += 1
            pred_str = str(r.prediction) if r.prediction else ""
            rep_output_tokens += _estimate_tokens(pred_str)
        output_tokens_total += rep_output_tokens

        per_repeat.append(
            {
                "repeat": rep,
                "wall_s": wall_s,
                "success_count": rep_success,
                "total_count": len(results),
                "output_tokens": rep_output_tokens,
                "stage_events": stage_events,
            }
        )

    # ── Resource summary ──
    peak_rss_kb = _peak_rss_kb()

    gpu_info: Dict[str, Any]
    if gpu_sampler.available:
        peak_vram_mb = gpu_sampler.peak_vram_bytes / (1024 * 1024)
        mean_util = (
            sum(gpu_sampler.util_samples) / len(gpu_sampler.util_samples)
            if gpu_sampler.util_samples
            else 0.0
        )
        gpu_info = {
            "peak_vram_mb": peak_vram_mb,
            "mean_gpu_util_pct": mean_util,
        }
    else:
        gpu_info = {
            "peak_vram_mb": None,
            "mean_gpu_util_pct": None,
        }

    return {
        "batch_size": batch_size,
        "micro_batch_size": pipeline._micro_batch_size,
        "repeats": repeats,
        "warmup_s": warmup_s,
        "per_repeat": per_repeat,
        "input_tokens_total": input_tokens_total,
        "output_tokens_total": output_tokens_total,
        "success_count": success_count,
        "total_count": total_count,
        "peak_rss_kb": peak_rss_kb,
        "gpu": gpu_info,
    }
