# Benchmarking

AXEtract includes a speed benchmark suite for measuring latency, throughput,
per-stage occupancy, and resource usage across CPU and GPU backends. This is
a **speed-only** benchmark — extraction quality is evaluated separately.

## Quick Start

```bash
# GPU benchmark with vLLM (default)
python -m benchmarks.run --backend vllm --device gpu

# CPU benchmark with HuggingFace
python -m benchmarks.run --backend hf --device cpu

# Quick smoke test (1 batch size, 1 repeat)
python -m benchmarks.run --backend vllm --batch-sizes 1 --repeats 1 --warmup 1

# With plots (requires matplotlib)
python -m benchmarks.run --backend vllm --plots
```

## Corpus

The benchmark reads HTML files from `data/benchmark/*.html`. Each file becomes
an `AXESample` with a default extraction query. Add your own `.html` files to
this directory to expand the corpus — the harness handles any number of files.

```bash
data/benchmark/
├── product_page_1.html
├── article_2.html
└── ...
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--backend` | `vllm` | LLM backend: `vllm` or `hf` |
| `--device` | `gpu` | Device: `cpu` or `gpu` (CPU forces HF backend) |
| `--batch-sizes` | `1,4,16,64` | Comma-separated input batch sizes to sweep |
| `--micro-batch-sizes` | `1,4,8,16` | Comma-separated micro-batch sizes to sweep |
| `--no-mb-sweep` | off | Fix micro-batch size at 4 instead of sweeping |
| `--repeats` | `5` | Timed repetitions per config (for p50/p90/p99) |
| `--warmup` | `3` | Warmup samples before timing (cold-start data) |
| `--corpus` | `data/benchmark` | Directory containing HTML files |
| `--query` | *(generic)* | Extraction query for all samples |
| `--out` | `benchmarks/results` | Output directory for results |
| `--plots` | off | Generate matplotlib plots |
| `--profile` | off | Dump cProfile stats for deep-dive profiling |
| `-v` | off | Enable debug logging |

## Metrics

### Latency

| Metric | Description |
|---|---|
| `latency_p50_s` | Median end-to-end latency per batch |
| `latency_p90_s` | 90th percentile latency |
| `latency_p99_s` | 99th percentile latency |
| `latency_mean_s` | Mean latency |
| `latency_iqr_s` | Interquartile range (p75 - p25) |

### Throughput

| Metric | Description |
|---|---|
| `docs_per_s` | Documents processed per second |
| `tokens_per_s` | Input + output tokens per second (char/4 heuristic) |
| `time_per_page_s` | Mean wall-clock per document |
| `time_per_1k_input_tokens_s` | Mean wall-clock per 1K input tokens |

### Per-Stage & Pipelining

| Metric | Description |
|---|---|
| `stage_occupancy_s` | Total busy time per stage (preprocess, prune, extract, postprocess) |
| `overlap_efficiency` | Pipelining gain: `1 - wall_clock / sum(stage_occupancy)` |

**Overlap efficiency** measures how much the pipelined execution overlaps
stages. A value of 0% means stages ran sequentially (no overlap); higher
values mean more overlap. It's computed as:

$$\text{overlap} = 1 - \frac{t_{\text{wall}}}{\sum_i t_{\text{stage}_i}}$$

### Resources

| Metric | Description |
|---|---|
| `peak_vram_mb` | Peak GPU VRAM allocated (null on CPU) |
| `mean_gpu_util_pct` | Mean GPU utilization % (null on CPU) |
| `peak_rss_mb` | Peak resident set size (CPU memory) |

### Success

| Metric | Description |
|---|---|
| `success_rate` | Fraction of samples with `Status.SUCCESS` |

## Output

Each run produces timestamped files in `benchmarks/results/`:

```
benchmarks/results/
├── bench_vllm_gpu_20260711_120000.json   # Raw + computed metrics
├── bench_vllm_gpu_20260711_120000.md     # Markdown summary table
└── bench_vllm_gpu_20260711_120000_plots/  # Optional PNG plots
    ├── throughput_vs_batch.png
    ├── stage_occupancy.png
    ├── overlap_vs_mb.png
    └── latency_vs_batch.png
```

## CPU vs GPU

The benchmark supports both CPU and GPU runs:

- **GPU**: `--backend vllm --device gpu` — the primary production path.
- **CPU**: `--backend hf --device cpu` — baseline showing the cost of no GPU.

CPU runs use the HuggingFace backend with `device_map="cpu"`. GPU metrics
(VRAM, utilization) are omitted for CPU runs.

## Adding Samples Programmatically

The harness accepts `AXESample` objects directly. To use custom samples
instead of loading from the corpus directory:

```python
from benchmarks.harness import build_pipeline, run_config, StageCollector
from benchmarks.metrics import compute_run_metrics
from axetract.data_types import AXESample

samples = [
    AXESample(id="doc1", content="<html>...</html>", is_content_url=False, query="..."),
    AXESample(id="doc2", content="<html>...</html>", is_content_url=False, query="..."),
]

pipeline = build_pipeline(backend="vllm", device="gpu", micro_batch_size=4)
collector = StageCollector()
raw = run_config(pipeline, samples, batch_size=2, repeats=5, warmup=3, collector=collector)
metrics = compute_run_metrics(raw)
```
