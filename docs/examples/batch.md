# Batch Processing

AXEtract is optimized for processing multiple documents efficiently. Use `extract()` for a single query across multiple URLs, or `extract_batch()` for heterogeneous tasks with different queries or schemas.

## Using `extract()` with Multiple URLs

Use this when you want to extract the same type of information from several similar pages. The pipeline automatically uses pipelined (multi-threaded) execution for large batches.

```python
from axetract import AXEPipeline

pipeline = AXEPipeline.from_config()

urls = [
    "https://example.com/item1",
    "https://example.com/item2",
    "https://example.com/item3",
]

results = pipeline.extract(
    input_data=urls,
    query="Extract the title and price"
)

for url, res in zip(urls, results):
    print(f"Results for {url}: {res.prediction}")
```

## Using `extract_batch()`

Use this when each item requires a different query or schema. You must provide `AXESample` objects with all required fields (`id`, `content`, `is_content_url`).

```python
import uuid
from axetract import AXEPipeline
from axetract.data_types import AXESample, Status
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float

pipeline = AXEPipeline.from_config()

batch = [
    AXESample(
        id=str(uuid.uuid4()),
        content="https://site1.com/article",
        is_content_url=True,
        query="Get the article abstract",
    ),
    AXESample(
        id=str(uuid.uuid4()),
        content="https://site2.com/product",
        is_content_url=True,
        schema_model=Product,
    ),
]

results = pipeline.extract_batch(batch)

for res in results:
    if res.status == Status.SUCCESS:
        print(res.prediction)
    else:
        print(f"Failed: {res.error}")
```
