# The Pipeline

The `AXEPipeline` is the main entry point for Axetract.

## Configuration

You can create a pipeline using the `from_config` factory method, which sets up the default adapters.

```python
from axetract import AXEPipeline

# Default configuration (Hugging Face)
pipeline = AXEPipeline.from_config()

# High-throughput configuration (vLLM)
pipeline_vllm = AXEPipeline.from_config(use_vllm=True)
```

## Processing Methods

### `process`
Processes a single input.

```python
result = pipeline.process(input_data="<html>...</html>", query="...")
```

### `process_many`
Processes a list of inputs using the **same** query/schema.

```python
results = pipeline.process_many(
    inputs=["url1", "url2", "url3"],
    query="Extract prices"
)
```

### `process_batch`
Processes a heterogenous batch of `AXESample` objects.

```python
from axetract.data_types import AXESample

batch = [
    AXESample(content="...", query="..."),
    AXESample(content="...", schema=MySchema)
]
results = pipeline.process_batch(batch)
```

## The `AXEResult` Object

Every processing method returns `AXEResult` objects containing:
- `prediction`: The actual extracted data (dict).
- `xpaths`: A mapping of extracted fields to their source XPaths.
- `status`: Execution status (SUCCESS, FAILURE, etc.).
- `error`: Error message if status is not SUCCESS.
