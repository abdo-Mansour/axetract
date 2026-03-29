# The Pipeline

The `AXEPipeline` is the main entry point for AXEtract.

## Configuration

Use the `from_config` factory method to create a ready-to-run pipeline with default LoRA adapters.

```python
from axetract import AXEPipeline

# Default: HuggingFace local inference
pipeline = AXEPipeline.from_config()

# High-throughput: vLLM serving
pipeline = AXEPipeline.from_config(use_vllm=True)
```

## Processing Methods

### `extract`

The primary method. Accepts a **single** input or a **list** of inputs — URLs, raw HTML strings, or `Path` objects pointing to `.html`/`.htm` files.

```python
from pydantic import BaseModel
from axetract import AXEPipeline

class Product(BaseModel):
    name: str
    price: float

pipeline = AXEPipeline.from_config()

# --- Single input, natural language query ---
result = pipeline.extract(
    input_data="https://example.com/item",
    query="Extract the product name and price",
)

# --- Single input, typed schema ---
result = pipeline.extract(
    input_data="https://example.com/item",
    schema=Product,
)

# --- Multiple inputs, same query/schema ---
results = pipeline.extract(
    input_data=[
        "https://example.com/item1",
        "https://example.com/item2",
    ],
    schema=Product,
)
```

### `extract_batch`

For heterogeneous batches where each item has its own query or schema, pass a list of `AXESample` objects.

```python
import uuid
from axetract import AXEPipeline
from axetract.data_types import AXESample
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float

pipeline = AXEPipeline.from_config()

batch = [
    AXESample(
        id=str(uuid.uuid4()),
        content="https://site-a.com/article",
        is_content_url=True,
        query="Get the article abstract",
    ),
    AXESample(
        id=str(uuid.uuid4()),
        content="https://site-b.com/product",
        is_content_url=True,
        schema_model=Product,
    ),
]

results = pipeline.extract_batch(batch)
```

## Execution Strategy

Execution mode is chosen automatically based on batch size:

| Batch size | Mode | Description |
|---|---|---|
| ≤ `micro_batch_size` (default: 4) | **Sequential** | Simple loop — preprocess → prune → extract → postprocess |
| > `micro_batch_size` | **Pipelined** | Concurrent threads with bounded queues for CPU/GPU overlap |

## The `AXEResult` Object

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Sample identifier |
| `prediction` | `dict` | Extracted structured data |
| `xpaths` | `dict \| None` | Field → source XPath mapping |
| `status` | `Status` | `Status.SUCCESS`, `Status.FAILED`, or `Status.PENDING` |
| `error` | `str \| None` | Error message if status is not `SUCCESS` |

```python
from axetract.data_types import Status

result = pipeline.extract("https://example.com", query="Get the price")

if result.status == Status.SUCCESS:
    print(result.prediction)
    print(result.xpaths)
else:
    print(result.error)
```
