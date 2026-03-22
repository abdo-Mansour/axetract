# Getting Started

## Installation

Axetract requires Python 3.12 or higher.

### Using `pip`

```bash
pip install axetract
```

### Using `uv` (Recommended)

```bash
uv add axetract
```

## Basic Usage

The core of Axetract is the `AXEPipeline`. It orchestrates the flow from raw HTML to structured data.

### 1. Simple Extraction

Pass a URL and a natural language query:

```python
from axetract import AXEPipeline

pipeline = AXEPipeline.from_config()

result = pipeline.process(
    "https://news.ycombinator.com",
    query="List the top 5 articles with their titles and points"
)

for article in result.prediction['articles']:
    print(f"{article['title']} ({article['points']} pts)")
```

### 2. Using Pydantic Schemas

For production environments, we recommend defining a Pydantic schema to ensure data consistency.

```python
from pydantic import BaseModel
from typing import List
from axetract import AXEPipeline

class Product(BaseModel):
    name: str
    price: float
    availability: bool

pipeline = AXEPipeline.from_config()

result = pipeline.process(
    "https://example.com/item",
    schema=Product
)

product = Product(**result.prediction)
print(product.name)
```

## Next Steps

- Explore the [Architecture](user-guide/architecture.md) to understand how it works.
- Check the [API Reference](api/pipeline.md) for detailed class documentation.
- See more [Examples](examples/basic.md).
