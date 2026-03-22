# Batch Processing

Axetract is optimized for processing multiple documents efficiently. You can use `process_many` for a single query across multiple URLs, or `process_batch` for different queries.

## Using `process_many`

Use this when you want to extract the same type of information from several similar pages.

```python
from axetract import AXEPipeline

pipeline = AXEPipeline.from_config()

urls = [
    "https://example.com/item1",
    "https://example.com/item2",
    "https://example.com/item3",
]

results = pipeline.process_many(
    inputs=urls,
    query="Extract the title and price"
)

for url, res in zip(urls, results):
    print(f"Results for {url}: {res.prediction}")
```

## Using `process_batch`

Use this for heterogenous processing.

```python
from axetract import AXEPipeline
from axetract.data_types import AXESample
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float

pipeline = AXEPipeline.from_config()

batch = [
    AXESample(content="https://site1.com", query="Get the abstract"),
    AXESample(content="https://site2.com", schema_model=Product),
]

results = pipeline.process_batch(batch)

for res in results:
    print(res.prediction)
```
