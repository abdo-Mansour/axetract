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
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from axetract.data_types import AXESample, Status
from axetract.pipeline import AXEPipeline

logger = logging.getLogger(__name__)


# Default extraction schema used when a corpus sample has no explicit schema.
# Mirrors the canonical "Product" example from the README so the benchmark
# exercises the same structured-extraction path that real users hit.
class Product(BaseModel):
    """Canonical benchmark extraction schema (product page)."""

    name: str
    price: str
    rating: float


DEFAULT_SCHEMA: Type[BaseModel] = Product

# Approximate chars-per-token for the char/4 heuristic.
CHARS_PER_TOKEN = 4


# ──────────────────────────────────────────────────────────────────────
# Corpus loading
# ──────────────────────────────────────────────────────────────────────


def load_corpus(
    html_dir: str | Path = "data/benchmark",
    schema_model: Optional[Type[BaseModel]] = DEFAULT_SCHEMA,
) -> List[AXESample]:
    """Load every ``*.html`` file in *html_dir* as an :class:`AXESample`.

    Files with a ``:Zone.Identifier`` suffix (Windows download metadata) are
    skipped.  Each sample's ``content`` is the raw HTML text, ``is_content_url``
    is ``False``, and ``schema_model`` is set to *schema_model*.

    Args:
        html_dir (str | Path): Directory containing ``.html`` files.
        schema_model (Optional[Type[BaseModel]]): Extraction schema to attach
            to every sample.  Defaults to :data:`DEFAULT_SCHEMA` (``Product``).

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
                schema_model=schema_model,
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

    VRAM tracking uses the **driver-level** view (``cudaMemGetInfo`` /
    ``torch.cuda.mem_get_info``) rather than ``torch.cuda.max_memory_allocated``
    so that allocations made **outside** of PyTorch's caching allocator — for
    example, by vLLM, which reserves its KV-cache blocks with raw CUDA
    calls — are still measured.  Without this fallback, vLLM-backed runs
    report ``peak_vram_mb=0`` even when the GPU is fully occupied.

    Attributes:
        util_samples (List[float]): Sampled utilization percentages.
        vram_samples (List[int]): Sampled used-VRAM byte counts.
        peak_vram_bytes (int): Peak allocated VRAM across the run.
    """

    def __init__(self) -> None:
        self.util_samples: List[float] = []
        self.vram_samples: List[int] = []
        self.peak_vram_bytes: int = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._torch = None
        self._device_index: int = 0
        try:
            import torch

            if torch.cuda.is_available():
                self._torch = torch
                # Pin to the current device so mem_get_info queries the
                # GPU vLLM (or HF) is actually using.
                try:
                    self._device_index = torch.cuda.current_device()
                except Exception:
                    self._device_index = 0
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
        self.vram_samples.clear()
        self.peak_vram_bytes = 0
        # Reset the PyTorch-allocator tracker too — useful for the HF
        # backend where the model lives inside the caching allocator.
        try:
            self._torch.cuda.reset_peak_memory_stats(self._device_index)
        except Exception:
            pass
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and record final peak VRAM.

        Prefers the driver-level peak (works for vLLM); falls back to
        PyTorch's caching-allocator tracker (works for HF) if the driver
        query produced no samples.
        """
        if not self.available:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        driver_peak = max(self.vram_samples) if self.vram_samples else 0
        try:
            torch_peak = self._torch.cuda.max_memory_allocated(self._device_index)
        except Exception:
            torch_peak = 0
        # Take the larger of the two so we never under-report.
        self.peak_vram_bytes = max(driver_peak, int(torch_peak))

    def _poll(self) -> None:
        """Background loop sampling utilization and VRAM at 10ms intervals."""
        assert self._torch is not None
        while not self._stop.is_set():
            try:
                util = self._torch.cuda.utilization()
                if util is not None:
                    self.util_samples.append(float(util))
            except Exception:
                pass
            # Driver-level used VRAM = total - free.  This reports
            # **all** allocations on the device (vLLM, other processes,
            # framework overhead), not just PyTorch's view.
            try:
                free_b, total_b = self._torch.cuda.mem_get_info(self._device_index)
                used_b = int(total_b) - int(free_b)
                if used_b > 0:
                    self.vram_samples.append(used_b)
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


def build_batch(samples: List[AXESample], batch_size: int) -> List[AXESample]:
    """Build an independent batch of *batch_size* samples from *samples*.

    Cycles through the corpus when it is smaller than *batch_size*.  Every
    returned sample is a **deep copy** with a unique ``id`` so concurrent
    pipeline stages never mutate the same object (shared references break
    pipelined execution and corrupt per-sample state).

    Args:
        samples (List[AXESample]): Source corpus (must be non-empty).
        batch_size (int): Desired number of samples in the batch.

    Returns:
        List[AXESample]: Deep-copied samples of length *batch_size*.

    Raises:
        ValueError: If *samples* is empty or *batch_size* < 1.
    """
    if not samples:
        raise ValueError("Cannot build a batch from an empty corpus.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    batch: List[AXESample] = []
    for i in range(batch_size):
        src = samples[i % len(samples)]
        # Deep copy so each slot is an independent pipeline item.
        cloned = src.model_copy(deep=True)
        # Keep ids unique and free of '-' so pruner chunkid parsing
        # (``chunkid.split("-")[0]`` → batch index) is unaffected; the
        # preprocessor assigns chunk ids from batch position, not sample.id,
        # but unique ids still help debugging and result attribution.
        cloned.id = f"{src.id}__{i}"
        # Ensure mutable stage fields start clean for a fair timed run.
        cloned.chunks = []
        cloned.original_html = ""
        cloned.current_html = ""
        cloned.prediction = None
        cloned.xpaths = None
        cloned.status = Status.PENDING
        batch.append(cloned)
    return batch


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

    # Template batch (independent deep copies).  Each timed call rebuilds
    # from this template so prior runs cannot leak mutated HTML/predictions.
    batch_template = build_batch(samples, batch_size)

    # Input token estimate (char/4) — computed once, constant across repeats.
    input_tokens_total = sum(_estimate_tokens(s.content) for s in batch_template)

    logger.info(
        "Config start: batch_size=%d micro_batch_size=%d repeats=%d warmup=%d "
        "corpus=%d (~%d input tokens/batch)",
        batch_size,
        pipeline._micro_batch_size,
        repeats,
        warmup,
        len(samples),
        input_tokens_total,
    )

    # ── Warmup ──
    warmup_n = min(warmup, batch_size) if warmup > 0 else 0
    warmup_s = 0.0
    if warmup_n:
        warmup_batch = build_batch(samples, warmup_n)
        logger.info("Warmup: extracting %d sample(s)...", warmup_n)
        t0 = time.perf_counter()
        pipeline.extract_batch(warmup_batch)
        warmup_s = time.perf_counter() - t0
        logger.info("Warmup done in %.2fs", warmup_s)

    # ── GPU sampler ──
    gpu_sampler = GPUResourceSampler()

    per_repeat: List[Dict[str, Any]] = []
    success_count = 0
    total_count = 0
    output_tokens_total = 0

    for rep in range(repeats):
        if collector is not None:
            collector.reset()

        # Fresh copies every repeat — pipeline stages mutate samples in place.
        batch = build_batch(samples, batch_size)

        logger.info(
            "Repeat %d/%d: extract_batch(n=%d) starting...",
            rep + 1,
            repeats,
            batch_size,
        )
        gpu_sampler.start()
        t_start = time.perf_counter()
        results = pipeline.extract_batch(batch)
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

        logger.info(
            "Repeat %d/%d done in %.2fs (success=%d/%d)",
            rep + 1,
            repeats,
            wall_s,
            rep_success,
            len(results),
        )

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
