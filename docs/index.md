<p align="center">
  <img src="assets/logo.png" alt="Axetract Logo" width="200">
</p>

# Axetract

**High-performance, LoRA-powered web data extraction.**

[Getting Started](getting-started.md){ .md-button .md-button--primary }
[GitHub](https://github.com/abdo-Mansour/axetract){ .md-button }

---

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Extreme Efficiency**
    ---
    Achieve GPT-4 level extraction accuracy with 0.6B parameter models.

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

Traditional web scrapers are either brittle (CSS selectors) or expensive (GPT-4). Axetract provides a middle ground: the intelligence of an LLM with the speed and cost of a local 0.6B model.

| Feature | Axetract (0.6B) | GPT-4o / Claude 3.5 |
|---------|-----------------|---------------------|
| Accuracy (SWDE F1) | **88.1%** | ~90% |
| Latency | **< 1s** | 5s - 15s |
| Cost | **Free (Local)** | High (Per Token) |
| Privacy | **100% On-Prem** | Cloud-dependent |

---

## Quick Start

```python
from axetract import AXEPipeline

# Create a pipeline with default LoRA adapters
pipeline = AXEPipeline.from_config()

# Extract data from a URL
result = pipeline.process(
    input_data="https://example.com/product",
    query="Extract the product name, price, and currency"
)

print(result.prediction)
# Output: {'name': 'Smartphone X', 'price': 999.0, 'currency': 'USD'}
```

## Get Involved

- [GitHub Repository](https://github.com/abdo-Mansour/axetract)
- [Hugging Face Adapters](https://huggingface.co/abdo-Mansour)
- [Issue Tracker](https://github.com/abdo-Mansour/axetract/issues)
