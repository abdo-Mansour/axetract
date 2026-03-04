<div align="center">
  <img src="docs/assets/logo.png" alt="AXEtract Logo" width="400">
  <h1>AXEtract</h1>
  <h3>Low-Cost Cross-Domain Web Structured Information Extraction</h3>

  [![Documentation](https://img.shields.io/badge/docs-latest-teal)](https://abdo-mansour.github.io/axetract/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![GitHub](https://img.shields.io/github/stars/abdo-Mansour/axetract?style=social)](https://github.com/abdo-Mansour/axetract)
</div>

---

**AXEtract** is a high-performance, low-cost framework for extracting structured data from web pages. Based on the paper **"AXE: Low-Cost Cross-Domain Web Structured Information Extraction"**, it optimizes the extraction pipeline by using specialized LoRA adapters for pruning and query-specific extraction, enabling state-of-the-art results with small models (e.g., Qwen3-0.6B).

## 🚀 Key Features

- **🎯 Specialized LoRA Adapters**: Uses task-specific adapters for DOM pruning and structured extraction, achieving high accuracy with minimal token overhead.
- **✂️ Smart DOM Pruning**: Classifies and prunes irrelevant HTML nodes before passing them to the extractor, significantly reducing context window usage and costs.
- **📍 Grounded XPath Resolution (GXR)**: Automatically maps extracted JSON fields back to their original source XPaths in the DOM for verification and grounding.
- **⚡ High-Throughput Pipeline**: Built-in support for multiple LLM engines, including **vLLM** for production-grade serving and **HuggingFace** for local research.
- **🌐 Cross-Domain Versatility**: Designed to generalize across various web domains (e-commerce, real estate, listings) without needing domain-specific rules.

## 🛠️ Architecture

AXEtract follows a three-part decoupled pipeline for maximum efficiency:

1.  **Preprocessor**: Fetches raw HTML and chunks it into manageable, token-aware fragments.
2.  **AI Extractor**: Divided into two stages:
    *   **Pruner**: A lightweight LLM (LoRA-powered) filters out noise and selects only relevant HTML chunks.
    *   **Extractor**: A task-specific LLM maps the pruned HTML content directly to a structured JSON schema or natural language answer.
3.  **Postprocessor**: Validates the output and resolves source XPaths via Grounded XPath Resolution (GXR).

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/abdo-Mansour/axetract.git
cd axetract

# Install with dependencies using uv or pip
pip install -e .
```

## 🚥 Quick Start

```python
from axetract.pipeline import AXEPipeline

# 1. Initialize the pipeline with default LoRA adapters
# (Automatically downloads adapters from HuggingFace Hub)
pipeline = AXEPipeline.from_config(use_vllm=False)

# 2. Extract from a URL or raw HTML
url = "https://example.com/item/12345"
query = "Extract the product name, price, and current rating."

result = pipeline.process(url, query=query)

# 3. Access your structured data
print(f"Status: {result.status}")
print(f"Prediction: {result.prediction}")
print(f"Source XPaths: {result.xpaths}")
```

### Using a Pydantic Schema

```python
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: str
    rating: float

result = pipeline.process(url, schema=Product)
```

## 🌐 API Server

AXEtract includes a built-in FastAPI server for high-throughput serving:

```bash
# Start the server (on port 8000 by default)
cd axe_server
python main.py
```

Check the `axe_server/client_example.py` for examples of interacting with the API via `requests`.

## 📝 Citation

If you use AXEtract in your research, please cite our paper:

```bibtex
@article{mansour2026axe,
  title={AXE: Low-Cost Cross-Domain Web Structured Information Extraction},
  author={Abdelrahman Mansour and Khaled W. Alshaer and Motaz El-Saban},
  journal={arXiv preprint},
  year={2026}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
