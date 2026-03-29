<p align="center">
  <img src="assets/logo-black-mode.svg" alt="AXEtract Logo" width="120" style="margin-bottom: 1rem;">
</p>

# AXEtract

**High-performance, LoRA-powered web data extraction. Based on the Paper [AXE: Low-Cost Cross-Domain Web Structured Information Extraction](https://arxiv.org/abs/2602.01838)**

[Getting Started](getting-started.md){ .md-button .md-button--primary }
[GitHub](https://github.com/abdo-Mansour/axetract){ .md-button }

---

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Extreme Efficiency**
    ---
    Achieve state-of-the-art extraction accuracy with 0.6B parameter models.

-   :material-brain:{ .lg .middle } **LoRA Switching**
    ---
    Dynamically switch between pruning and extraction adapters in a single VRAM footprint.

-   :material-target:{ .lg .middle } **Grounded XPath (GXR)**
    ---
    Automatically map extracted data back to the original DOM XPaths.

-   :material-lightning-bolt:{ .lg .middle } **vLLM Support**
    ---
    Built-in support for high-throughput batch processing with vLLM.

</div>

## Why Axetract?

Traditional web extractors are often a trade-off between brittle manual heuristics and the prohibitive cost of Large Language Models. Axetract provides a solution: the intelligence and flexibility of an LLM with the efficiency of a local 0.6B model via intelligent DOM pruning.

| Feature | Axetract (0.6B) |
|---------|-----------------|
| Accuracy (SWDE F1) | **88.1%** |
| Compute Required | **Low (0.6B)** |
| Cost | **Free (Local)** |
| Privacy | **100% On-Prem** |

---

## Quick Start

```python
from pydantic import BaseModel
from axetract import AXEPipeline

class Product(BaseModel):
    name: str
    price: float
    currency: str

pipeline = AXEPipeline.from_config()

result = pipeline.extract(
    input_data="https://example.com/product",
    schema=Product,
)

print(result.prediction)
# Output: {'name': 'Smartphone X', 'price': 999.0, 'currency': 'USD'}
```

## Get Involved

- [GitHub Repository](https://github.com/abdo-Mansour/axetract)
- [Hugging Face Adapters](https://huggingface.co/abdo-Mansour)
- [Issue Tracker](https://github.com/abdo-Mansour/axetract/issues)
