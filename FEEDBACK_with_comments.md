# AXEtract — Comprehensive Package Feedback & Improvement Roadmap

> This document is a structured critique of the `axetract` package from the perspective of an outside developer encountering it for the first time. It covers the developer experience (DX), API design, documentation quality, missing features, deployment story, and community readiness. Items are grouped by theme and prioritized within each section.

---

## Table of Contents

1. [Critical Blockers — Fix Before Any Public Announcement](#1-critical-blockers)
2. [API & Interface Design](#2-api--interface-design)
3. [Documentation Gaps](#3-documentation-gaps)
4. [Missing Features & Capabilities](#4-missing-features--capabilities)
5. [Error Handling & Observability](#5-error-handling--observability)
6. [Performance & Scalability](#6-performance--scalability)
7. [Testing & Quality Assurance](#7-testing--quality-assurance)
8. [Community & Ecosystem Readiness](#8-community--ecosystem-readiness)
9. [Deployment & Production Story](#9-deployment--production-story)
10. [Small But High-Impact Polish](#10-small-but-high-impact-polish)

---

## 1. Critical Blockers

These are issues that will make a first-time user give up within minutes.

### 1.1 — README Installation Command is Wrong

The README shows:

```bash
pip install -e .
```

But `getting-started.md` says:

```bash
pip install axetract
# or
uv add axetract
```

If the package is not yet on PyPI, the README *must* say so explicitly, and the `getting-started.md` must not claim a `pip install axetract` works. If it **is** on PyPI, remove the clone-based installation from the top of the README. A confused first install is a silent project killer.

> We will use UV everywhere and add requirements.txt as well for those who need it

### 1.2 — `pipeline.process()` vs `pipeline.extract()` — The API is Inconsistent

In the README quick-start:

```python
result = pipeline.process(url, query=query)
```

In `getting-started.md`, the examples, and the actual source code, the correct method is:

```python
result = pipeline.extract(url, query=query)
```

`process()` does not exist on `AXEPipeline`. Someone copying the README will get an `AttributeError` on line 1. This is arguably the most damaging bug in the entire project right now — it is the first thing every new user will try.

> We should use extract instead of process

### 1.3 — No `__version__` available at the module level

```python
import axetract
print(axetract.__version__)  # works
```

But there is no `axetract.__version__` exported in `__all__` and it is hardcoded in two places: `__init__.py` and `pyproject.toml`. These will inevitably drift. Use `importlib.metadata` to derive the version at runtime:

```python
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("axetract")
except PackageNotFoundError:
    __version__ = "unknown"
```

> Sure, good thing

### 1.4 — The `axe_server` Cannot be Installed via the Package

The server lives in `axe_server/` but has no entry point, no separate install, and requires manually `cd`-ing into the folder. A user who installs `pip install axetract` has no idea the server exists or how to run it. Add a CLI entry point:

```toml
[project.scripts]
axe-server = "axetract.server:main"  # or similar
```

> Good solution, needs testing though
---

## 2. API & Interface Design

### 2.1 — No Context Manager / Resource Management

Loading a 0.6B model takes time and memory. There is no `__enter__`/`__exit__` or explicit `.close()` / `.shutdown()` on `AXEPipeline`. Users running pipelines inside loops will have no clean way to release GPU memory. This is especially painful with vLLM. Implement:

```python
with AXEPipeline.from_config() as pipeline:
    result = pipeline.extract(url, query="...")
# Model is unloaded here automatically
```

> This is low priority for now, no need to do it

### 2.2 — `from_config()` Hides Too Much Magic

`AXEPipeline.from_config()` is a good pattern, but the default `llm_config` is a 20-line nested dict buried inside the method body. There is no way for a user to inspect what the defaults are, extend them partially, or know what keys are valid.

**Recommendation**: Create a `AXEConfig` Pydantic model:

```python
from axetract import AXEConfig, AXEPipeline

config = AXEConfig(
    model_name="Qwen/Qwen3-0.6B",
    chunk_size=1500,
    gpu_memory_utilization=0.6,
)
pipeline = AXEPipeline.from_config(config)
print(config.model_json_schema())  # Self-documenting
```

This also makes config serializable (save/load extraction configs as JSON).

> We should have AXEPipeline() work with reasonable defaults without the need for .from_config()
> Currently, the way the pipeline handels LLMs is bad, it has an LLMClient interface but it takes a dictionary named config as input, this is bad as the interface user (or the user of the interface children) has no way of knowing which paramters they should provide, this needs a good refactor where the base client (named BaseClient) should be renamed to BaseLLMClient and have the same interface functions but with no constructor parameters, children can define their constructor parameters based on their backend.

### 2.3 — `AXEResult` Has No Convenience Methods

Users receive an `AXEResult` and must manually check `.status`, handle `.error`, and cast `.prediction`. Add helper methods:

```python
result.raise_for_status()   # raise ExtractionError if status != SUCCESS
result.to_model(MySchema)   # return MySchema(**result.prediction)
result.as_dataframe()       # pd.DataFrame for list predictions
```
> Sure thing, but more detailes are needed

### 2.4 — No `async` Support

The entire pipeline is synchronous. In a FastAPI server this means each `/process` request blocks an entire thread for the duration of inference. There is no `async def extract(...)` variant. For the server use-case, this will become a bottleneck immediately even with a single user.

**Recommendation**: Offer `await pipeline.aextract(url, query=...)` using `asyncio.to_thread` as a minimal solution.

> Generally using AsyncIO is a good idea, but it will be an overhaul refactor so it will need some good time, great idea but not immediate priority.

### 2.5 — `extract_batch_same_query` is an Awkward Name

Discoverability suffers. Python convention would be `extract_many` or simply overloading `extract` to accept a list. The current names:

- `extract()` — single item
- `extract_batch()` — mixed batch
- `extract_batch_same_query()` — list with shared query

This is three methods where one or two would suffice. Consider:

```python
pipeline.extract(url, query=...)           # single
pipeline.extract([url1, url2], query=...)  # auto-detect list → batch with same query
pipeline.extract(batch)                    # AXESample list → heterogeneous batch
```
> Good idea, let's do this

### 2.6 — The `AXESample` Public Surface is Awkward

`AXESample` is both an internal state container (with `chunks`, `current_html`, `status`) and the user-facing input object. A user building a batch must know about `is_content_url`, `content`, etc. — internal fields that should not be part of the public API.

**Recommendation**: Separate concerns.

- `AXEInput` — the public input model (`content`, optional `query`, optional `schema`)
- `AXESample` — the internal mutable state object (not exported)

> Ok, good idea

### 2.7 — No Streaming / Progressive Results

For large pages or slow networks, there is no signal while the pipeline is running. Users see nothing until the entire pipeline finishes. Even a simple progress callback would help:

```python
pipeline.extract(
    url,
    query="...",
    on_progress=lambda stage, pct: print(f"{stage}: {pct:.0%}")
)
```
> Good idea, but not a priority currently

---

## 3. Documentation Gaps

### 3.1 — GXR (Grounded XPath Resolution) is Mentioned But Never Explained Clearly

GXR is one of the most unique features of AXEtract. Yet:

- The architecture doc says the Extractor "provides XPaths for every extracted value" but the actual XPath resolution happens in the **Postprocessor**, not the Extractor. This is misleading.
- There is **no example** showing what `result.xpaths` actually looks like.
- There is no explanation of *why* XPaths are useful (verification, re-scraping, DOM interaction).

Add a dedicated page: **`docs/user-guide/gxr.md`** with:
- What GXR is and why it exists
- A concrete example of `result.xpaths` output
- How to use XPaths with `lxml` or `selenium`
- Its limitations (fuzzy matching, dynamic pages)

> Great idea, let's do this.


### 3.2 — No API Reference for Key Public Classes

`docs/api/pipeline.md` exists but contains only:

```markdown
::: axetract.pipeline.AXEPipeline
```

There is no narrative — no explanation of when to use `from_config()` vs manual construction, no parameter tables rendered in the final docs. Verify the MkDocs auto-generation actually works and produces readable output.

> Ok, good idea too.

### 3.3 — `AXEChunk`, `Status`, and `AXEResult` are Undocumented from a User Perspective

A user receiving a `FAILED` status has no idea what caused it, where to look, or how to retry. Document the `Status` enum values and what each failure mode implies.

> Ok, but this needs more research on how we would do this, we need to understand the problem more and the different ways to solve it.

### 3.4 — The Benchmark Page is Empty

`benchmarks/basic.py` exists but its results appear nowhere in the docs. The `docs/index.md` table shows:

```
| Accuracy (SWDE F1) | **88.1%** |
```

But there is no explanation of what SWDE is, how the benchmark was run, or how to reproduce it. To researchers (the primary audience from the paper), this is a red flag.

Add a dedicated `docs/benchmarks.md` page:
- SWDE dataset description and link
- Reproduction commands
- Hardware used (GPU, VRAM)
- Comparison table against baseline methods from the paper

> benchmarks folder is in progress and should have much work done in it, to pick the benchmarking strategy, write the code, benchmark it on a dataset, choosing a subset of the dataset and add it as part of the process to merge PRs (like tests but for speed & accuracy of the package)

### 3.5 — No Explanation of What Happens When Both `query` and `schema` Are Passed

The code in `AXEExtractor._generate_output()` treats `schema_model` as the `query` if no `query` is set. But the docs never say what happens if a user passes both — does one take precedence? Which?

> Good point, we need to determine what to do in this case, this is a valid issue

### 3.6 — The `components.md` Postprocessor Section is Wrong

The components doc says:
> **Schema Validation**: Forcing the output to match your Pydantic model.

But the `AXEPostprocessor` does **not** validate against Pydantic. It uses `json-repair` and fuzzy matching. It cannot "force" the output to match a schema. This will create false expectations and confusion.

> fair enough, enforcement should be put, maybe with a loose mode to return as is without forcing the output.

### 3.7 — Missing: How to Extend the Pipeline

There are `BasePruner`, `BaseExtractor`, `BasePreprocessor`, and `BasePostprocessor` abstract classes — a clear extension point. But there is **no documentation or example** showing how to write a custom component. This is a major missed opportunity. Add `docs/user-guide/extending.md`.

> Good point, we need to add examples on how to extend both in code and docs.

---

## 4. Missing Features & Capabilities

### 4.1 — No Caching Layer

Every call to `pipeline.extract(url)` re-fetches the URL, re-processes the full HTML, and re-runs inference. For development and debugging this is expensive. A simple caching mechanism would be highly valued:

```python
pipeline = AXEPipeline.from_config(cache_dir=".axe_cache")
result = pipeline.extract(url, query="...")  # Cached on disk after first run
```

This could use `diskcache` or even just pickle files keyed by URL hash.

> good enhancement but not a priority currently.

### 4.2 — No Retry Logic on URL Fetch Failures

`fetch_content()` exists but the pipeline only wraps it in a bare `try/except` that silently replaces the content with `"[Fetch ERROR] ..."`. The string then gets processed by the model which will produce garbage. The pipeline should:
1. Retry with exponential backoff on transient HTTP errors (429, 503)
2. Mark the `AXEResult` with `Status.FAILED` and a meaningful error message
3. Optionally accept a `requests.Session` for custom headers / cookie injection

> Great thinking, retrying mechanisms shall be implemented.

### 4.3 — No Support for JavaScript-Rendered Pages

Modern web pages require JavaScript execution for meaningful content. AXEtract has no integration with `playwright` or `selenium`. Even a simple note in the FAQ would help, but ideally:

```python
from axetract.fetchers import PlaywrightFetcher

pipeline = AXEPipeline.from_config(fetcher=PlaywrightFetcher())
```

> we will add it as an issue but won't fix it anytime soon since it needs research, not just coding.

### 4.4 — No Output Validation Against Pydantic Schema

When a user passes `schema=MyModel`, the final `result.prediction` is a raw `dict`. There are no guarantees it matches `MyModel`. Users have to do `MyModel(**result.prediction)` themselves and handle `ValidationError`. The pipeline should:

```python
result = pipeline.extract(url, schema=Product)
# result.prediction is already a validated Product instance
# or result.status is FAILED with a validation error message
```

> Correct, good catch

### 4.5 — No `dry_run` / `skip_pruner` Flag

During development, users want to test extraction prompts without waiting for the pruner. A `skip_pruner=True` flag on `extract()` would be invaluable for iteration speed.

```python
result = pipeline.extract(url, query="...", skip_pruner=True)  # fast debug mode
```

> Sure, skipping pruner will be helpful for some users

### 4.6 — No Way to Inspect Intermediate Outputs

There is no supported way to see what the pruner decided to keep, or what HTML the extractor actually saw. This makes debugging poor extractions very difficult. Expose:

```python
result = pipeline.extract(url, query="...", return_intermediates=True)
result.pruned_html     # What the pruner kept
result.extractor_input # The exact string sent to the LLM
result.raw_llm_output  # The raw text before JSON repair
```

> good idea

### 4.7 — No LiteLLM / OpenAI Compatibility in `from_config()`

`LiteLLMClient` exists in the source but is not exposed through `from_config()` at all. A user who wants to use `gpt-4o` or `claude-3-5-sonnet` as their extractor has to manually instantiate four pipeline components. Add:

```python
pipeline = AXEPipeline.from_config(
    provider="openai",
    model="gpt-4o-mini",
    api_key="sk-..."
)
```

> This is cruical, but not in this way, just as I mentioned earlier with the LLM client, I want to have a child for litellm that inherits from the BaseLLM class. these two issues can be linked together.

### 4.8 — No Sitemap / Multi-Page Crawling Helper

The batch API works per-URL, but there is no built-in utility to crawl a site or process a sitemap. A `from_sitemap(url)` helper or a `crawl(seed_url, depth=2)` mode would significantly expand the addressable use-cases.

> Mmmm, no, a crawler is out of scope of this package, developers can use firecrawl or crawl4ai package.

---

## 5. Error Handling & Observability

### 5.1 — Pipeline Swallows Errors Silently in Pipelined Mode

In `_process_pipelined()`, exceptions are caught and appended to an `errors` list:

```python
errors.append(e)
q_preprocessed.put((mb_idx, mb))  # Pass through on error
```

But the sample is silently passed through to the next stage with potentially corrupt state. The final `AXEResult` may have `Status.SUCCESS` even when an error occurred mid-pipeline. The `errors` list is logged as a warning but never exposed to the caller.

> Yes, good catch

### 5.2 — `Status.FAILED` Provides No Root Cause

When `result.status == Status.FAILED`, the `result.error` field says:
> `"Encountered error/pending status: Status.FAILED"`

This is useless. The error field should contain the actual exception message or stage name where failure occurred.

> Yes, we shall use logging to log the error and the error field should contain meaningful values.

### 5.3 — Logging is All-Or-Nothing

The package uses `logging.DEBUG` for all internal messages. There is no structured way for users to get a concise summary of what happened (e.g., how many chunks were pruned, how long the LLM took) without drowning in debug noise. Consider a pipeline-level metrics object:

```python
result.metrics = {
    "chunks_before_prune": 42,
    "chunks_after_prune": 3,
    "prune_time_s": 0.41,
    "extract_time_s": 1.22,
    "total_time_s": 1.63
}
```

> Great, having logging.exception or error, and having some metrics is very useful for benchmarking too.

### 5.4 — The Schema Template Prompt Repetition is a Code Smell

In `schema_prompt.py`:

```
STICK TO THE TARGET SCHEMA STRUCTURE
STICK TO THE TARGET SCHEMA STRUCTURE
STICK TO THE TARGET SCHEMA STRUCTURE
STICK TO THE TARGET SCHEMA STRUCTURE
STICK TO THE TARGET SCHEMA STRUCTURE
STICK TO THE TARGET SCHEMA STRUCTURE
STICK TO THE TARGET SCHEMA STRUCTURE
```

This suggests schema adherence is a known pain point. This is a workaround, not a solution. Consider:

- Structured outputs / JSON mode (vLLM supports `guided_json`)
- Post-extraction Pydantic validation (see §4.4)
- A dedicated fine-tuning note for users who need stricter compliance

At minimum, this repeated line is embarrassing in a public library and should be addressed before any community exposure.

> Mmmm, no, this empirically enhanced our results for the model, add this as an issue just because we need to add better default prompts in case a larger, stronger LLM was used, where this repetition is unnecessary.

---

## 6. Performance & Scalability

### 6.1 — `from_config(use_vllm=False)` Loads the Model on First Call, with No Warning

When using `HuggingFaceClient`, the model is downloaded and loaded into memory the first time `extract()` is called. There is no progress bar, no log message visible at INFO level, and no way for the user to pre-warm the pipeline. Add a `pipeline.warmup()` method and log at INFO when the model starts loading.

> Ok, no problems

### 6.2 — The Sequential Threshold is Hardcoded and Undocumented

In `extract_batch()`:

```python
if len(batch) <= self._micro_batch_size:
    return self._process_sequential(batch)
```

The cutoff between sequential and pipelined execution is `micro_batch_size` (default 4). This is not documented anywhere. Users with batch sizes of 3 get sequential processing without knowing it. Make this configurable and document the trade-off.

> Ok, needs to be added to docs as well as logging it in code as a warning.

### 6.3 — `ProcessPoolExecutor` in the Preprocessor Can Cause Issues in Certain Environments

The preprocessor uses `ProcessPoolExecutor` for CPU-bound chunking. This breaks when:
- Run inside a Jupyter notebook (spawn context issues on macOS/Windows)
- Run inside another multiprocessing context (e.g., a Celery worker)
- Run on systems where `fork` is not available

Add a graceful fallback to `ThreadPoolExecutor` when process spawning fails, and document this limitation.

> Good idea.

### 6.4 — No Token Usage Tracking

For users paying for API access (via `LiteLLMClient`), there is no tracking of tokens consumed. This makes cost estimation impossible. Add an optional `token_usage` field to `AXEResult`.

> Good idea.

---

## 7. Testing & Quality Assurance

### 7.1 — Zero End-to-End Tests Against Real URLs

`tests/e2e/test_e2e.py` exists but its content is not visible in a typical CI run. If E2E tests are skipped/mocked, then there is no guarantee the actual HTML fetching + model inference works as advertised. Add at least one live integration test (gated behind an environment variable `RUN_E2E=1`) against a stable, publicly accessible URL.

> Good idea, no problems with this.

### 7.2 — No Tests for the `LiteLLMClient`

`tests/llm/` has tests but there are no tests for `LiteLLMClient`. Given that LiteLLM is the bridge to commercial providers (OpenAI, Anthropic, etc.), this is a gap.

> Yes, this can be combined with the issue of adding liteLLM client for ease of use mentioned above.

### 7.3 — No Property-Based / Fuzz Tests for JSON Repair

`json-repair` is used in a critical path. The kinds of malformed JSON an LLM produces are highly varied. Property-based tests using `hypothesis` would provide significantly better coverage than hand-crafted unit tests.

> Good idea

### 7.4 — Test Coverage is Not Reported

There is no coverage badge on the README, no `.coveragerc`, and no coverage reporting in the CI. Even a basic `pytest --cov=axetract` invocation in CI would surface untested code paths.

> Good idea

---

## 8. Community & Ecosystem Readiness

### 8.1 — No GitHub Issue Templates

Without issue templates, bug reports will be low-quality and missing essential context (Python version, OS, model backend, HTML snippet that caused the issue). Add `.github/ISSUE_TEMPLATE/bug_report.md` and `feature_request.md`.

> Great stuff

### 8.2 — No `SECURITY.md` is Meaningful

The `SECURITY.md` exists, which is good. But check that it has actual contact information and a clear disclosure process, not just a placeholder.

> yes, it has, no need to turn this into an issue.

### 8.3 — No Discussion Forum / Discord

The community is pointed only to GitHub Issues. For a research-adjacent package, a GitHub Discussions tab or a Discord server would allow pre-issue questions, adapter sharing, and community benchmarking.

> Hmmmm, our contact information exist, they cand send us emails, no need to turn this into an issue.

### 8.4 — `pyproject.toml` Only Declares Python 3.12

```toml
classifiers = [
    "Programming Language :: Python :: 3.12",
]
```

Python 3.12 is recent. Does it actually fail on 3.11? If not, declare `>=3.11` in both `requires-python` and classifiers. Narrower Python requirements reduce the addressable audience for no clear reason.

> We need to have github actions / local ways to test different versions of python on the package to ensure compatability. good idea.

### 8.5 — No `pip install axetract[server]` Extra

The server requires `fastapi`, `uvicorn`, and `python-dotenv`, but these are not in any optional dependency group. A user running the server will get `ImportError` on `fastapi`. Add:

```toml
[project.optional-dependencies]
server = ["fastapi>=0.115", "uvicorn>=0.34", "python-dotenv>=1.0"]
litellm = ["litellm>=1.67"]
```

> good idea.

### 8.6 — Adapter Versions are Pinned to Implicit "Latest"

```python
"pruner": {
    "path": "abdo-Mansour/Pruner_Adaptor_Qwen_3_FINAL_EXTRA",
    ...
}
```

The HuggingFace model ID has `FINAL_EXTRA` in the name, which signals iteration but not versioning. When you update the adapter (e.g., to fix a regression), existing users' pipelines will silently start using a different model. Use HuggingFace model **revision** pinning:

```python
"pruner": {
    "path": "abdo-Mansour/Pruner_Adaptor_Qwen3",
    "revision": "v1.0.0",  # Pin to a git tag
}
```
> Ok, good idea

---

## 9. Deployment & Production Story

### 9.1 — No Docker / Container Support

There is no `Dockerfile`, no `docker-compose.yml`. For a package that bundles a web server, the absence of a container setup is a significant gap. Users deploying to cloud environments (AWS, GCP, etc.) expect a container. Provide:

```
docker/
  Dockerfile          # server + model
  docker-compose.yml  # server + optional GPU setup
```

> great idea

### 9.2 — The FastAPI Server has No Authentication

`/process` and `/process_batch` endpoints have no authentication, rate limiting, or input size validation. The `input_data` field accepts arbitrary strings with no max length. In a public deployment this is both a security and resource exhaustion risk.

> This needs to be addressed in the server cli parameters, it should be supported via both `uv tool install axetract[server]` for example, and docker for people to run the image with correct parameters.

### 9.3 — The Server Startup Silently Fails

If `AXEPipeline.from_config()` throws during startup, the `pipeline` global stays `None` and all subsequent requests return `503`. There is no crash — the server starts "successfully" and then fails every request silently. The startup error is only visible in logs. This should be a hard-fail: the server should refuse to start if the pipeline cannot be initialized.

> Great idea

### 9.4 — No Health Check Returns Pipeline Details

The `/health` endpoint only reports `pipeline_initialized: bool`. In production, this should include model name, adapter versions, GPU memory available, and uptime — standard observability for an ML serving system.

> good idea

### 9.5 — No Kubernetes / Helm Chart Guidance

For teams deploying this at scale, there is no guidance on horizontal scaling, resource requests (`nvidia.com/gpu: 1`), or liveness/readiness probe configuration. A brief `docs/deployment/kubernetes.md` would go a long way.

> Hmmm, a bit too much, the server is made for basic stuff and local deployment, serious scaleable software teams will use the package with their own backend servers. But sure add it as an issue and will see about it later on.

---

## 10. Small But High-Impact Polish

| # | Issue | Effort | Impact |
|---|-------|--------|--------|
| 10.1 | `AXEResult` should implement `__repr__` for readable REPL output | Low | High |
| 10.2 | Add `result.prediction` type narrowing — currently `Union[str, dict, Any]` which defeats static analysis | Low | Medium |
| 10.3 | The `AXESample.id` is set to `str(uuid.uuid4())` but `AXESample` also accepts `id` as a constructor param — the schema says `str` but `_format_batch` passes `str(item.get("id", uuid.uuid4()))` which can silently generate a UUID string from a UUID object | Low | Medium |
| 10.4 | `AXEPreprocessor` takes `mp.cpu_count()` as the default for `fetch_workers` — for URL fetching, IO-bound tasks benefit from many more workers (e.g., `min(32, cpu_count * 4)`) | Low | Medium |
| 10.5 | The quickstart in `getting-started.md` accesses `result.prediction['articles']` with a hardcoded key — this will throw `KeyError` for most users trying the example | Low | High |
| 10.6 | The Mermaid diagram in `architecture.md` labels the Pruner and Extractor as one "Small LLM (0.6B)" subgraph but does not show the LoRA switch — the most interesting implementation detail is invisible | Low | Medium |
| 10.7 | The `benchmarks/basic.py` exists but is not run in CI and its results are never shown anywhere | Low | High |
| 10.8 | Adding a `py.typed` marker file to `src/axetract/` would signal to type checkers that the package supports PEP 561 — without it, mypy/pyright ignores type annotations | Low | Medium |
| 10.9 | The `CHANGELOG.md` entry is dated `2026-03-04` which is in the future relative to most readers — if the paper is under review, consider marking this clearly | Low | Low |
| 10.10 | `is_content_url` is determined by checking if the string starts with `http://` or `https://` — this will misclassify raw HTML that begins with a comment like `<!-- https://... -->`. Use a proper URL validator (`urllib.parse.urlparse`) | Low | Medium |

> Great points, we need an issue for each of these.
> 10.7 can be combined with previous issue earlier
---

## Summary of Highest Priority Actions

If you can only act on five things, act on these:

1. **Fix `pipeline.process()` → `pipeline.extract()` in the README** — every new user hits this immediately.
2. **Add a working `pip install` command or clearly mark it as not-on-PyPI** — the installation story is broken.
3. **Expose `LiteLLMClient` in `from_config()`** — this unlocks OpenAI/Anthropic backends and dramatically increases the addressable audience.
4. **Add a GXR explanation page with a real `result.xpaths` example** — the most unique feature is also the most invisible.
5. **Fix the repeated `STICK TO THE TARGET SCHEMA STRUCTURE` prompt** — this is the first internal implementation detail most contributors will see, and it sends the wrong signal about code quality.
