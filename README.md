# axe-extract

 Here is your highly intricate, copy-pasteable checklist. You can drop this directly into a GitHub Issue, Notion, or Linear. It is explicitly divided by day, by person, and sequenced so that nobody is blocked.

### **Pre-Sprint: The 30-Minute Sync (Both Members)**
- [ ] **Define the API Contract:** Agree on the exact input/output format of the `AXEPipeline` class. (e.g., `extractor.extract(url, schema)` returns `{"data": dict, "xpaths": dict}`).
- [ ] **Setup GitHub Repo:** Initialize the repository, branch protection rules (require 1 review), and give both members admin access.
- [ ] **Create Branches:**
  - Person A creates `feature/core-engine`.
  - Person B creates `feature/dx-and-tests`.

---

### **Day 1: Core Engine & Developer Experience (DX)**

#### **Person A (Core ML & Packaging Engineer)**
**Morning: Project Scaffolding & Core Modules**
- [ ] **Package Structure:** Create the `axe/` directory and empty `__init__.py`, `pipeline.py`, `pruner.py`, `llm.py`, and `gxr.py` files.
- [ ] **Dependency Management:** Write the `pyproject.toml`. Include dependencies: `transformers`, `peft`, `accelerate`, `htmlrag`, `lxml`, `beautifulsoup4`, `pydantic`.
- [ ] **Implement `pruner.py`:** Port the HTMLRAG DOM pruning research code. Ensure it safely handles malformed HTML without crashing.
- [ ] **Implement `gxr.py`:** Port the Grounded XPath Resolution logic. Ensure it maps LLM text outputs reliably back to the original DOM XPaths.

**Afternoon: Hugging Face & Multi-LoRA Logic**
- [ ] **Implement `llm.py`:** Write the base model loading logic using `AutoModelForCausalLM.from_pretrained("Qwen/Qwen-0.6B")`.
- [ ] **Multi-LoRA Support:** Add `PeftModel` loading. Implement the ability to load the base model once, and use `model.load_adapter()` to load your task-specific adapters into the same VRAM footprint.
- [ ] **Dynamic Switching:** Implement the logic that detects the task type and uses `model.set_adapter("target_adapter")` before inference.
- [ ] **Device Routing:** Add a utility to automatically detect and route to `cuda` (Nvidia), `mps` (Apple Silicon), or `cpu`.

**Evening: The Pipeline Wrapper**
- [ ] **Implement `pipeline.py`:** Stitch `pruner`, `llm`, and `gxr` together into the main `AXEPipeline` class.
- [ ] **Local Sanity Check:** Run a local Python script to ensure a raw URL can be passed in, and valid JSON + XPaths come out. Push branch.

#### **Person B (DX, Testing & Integrations Engineer)**
**Morning: Testing Framework & Mocks**
- [ ] **Setup `pytest`:** Add `pytest` to dev dependencies. Create a `tests/` directory.
- [ ] **Create Mock Data:** Save 3 snippet HTML files locally (`tests/mocks/table.html`, `tests/mocks/list.html`, etc.) so tests don't require internet access.
- [ ] **Write Unit Tests:** Write tests for `gxr.py` checking if specific strings correctly resolve to XPaths in the mock HTML.
- [ ] **Write Schema Validation Tests:** Ensure the pipeline fails gracefully if a user provides an invalid JSON schema.

**Afternoon: Documentation Foundation**
- [ ] **Initialize `README.md`:** Create the hero section. Add a clean architecture diagram (can use Excalidraw or similar) showing: HTML -> Pruning -> Qwen3-0.6B -> JSON + XPath.
- [ ] **Add Performance Table:** Highlight the 88.1% F1 SWDE score. Compare it to GPT-4o or Claude 3.5 to show cost/speed superiority.
- [ ] **Write Quickstart:** Write the 5-line standard Python `pip install` and inference code block.
- [ ] **Write vLLM Guide:** Add the "High-Performance Serving" section explaining how enterprise users can use `vLLM` for concurrent mixed-batch LoRA routing.

**Evening: The Colab Playground**
- [ ] **Create `notebooks/axe_quickstart.ipynb`:** Set up the notebook to run on a free T4 GPU.
- [ ] **Add Setup Blocks:** Write the `!pip install` commands.
- [ ] **Add Examples:** Create 2 real-world extraction examples (e.g., scraping an Amazon product, or a Wikipedia table). Test it using Person A's branch. Push branch.

---

### **Day 2: Integrations, CI/CD & The Launch**

#### **Person A (Deployment & MLOps)**
**Morning: CI/CD & Automation**
- [ ] **Code Formatting:** Run `ruff` or `black` over the codebase. Add a `make format` or simple script for it.
- [ ] **GitHub Actions:** Create `.github/workflows/python-app.yml`. Configure it to run `pytest` and linting on every push to `main`.
- [ ] **Hugging Face Model Cards:** Go to HF Hub. Update all adapter model cards with the same README, clearly linking to the new GitHub repo.

**Afternoon: PyPI Deployment & Bug Fixes**
- [ ] **Build Package:** Run `python -m build` to generate the `.whl` and `.tar.gz` files.
- [ ] **Test Wheel Locally:** Install the built `.whl` file in a completely fresh virtual environment to catch missing dependencies.
- [ ] **Publish to TestPyPI:** Run `twine upload --repository testpypi dist/*` to ensure the name `axe-extractor` is available and uploads correctly.
- [ ] **Publish to PyPI:** Execute the final `twine upload dist/*`.
- [ ] **Bug Support:** Sit on standby to fix any edge-case bugs Person B finds during end-to-end integration testing.

#### **Person B (Integrations & Marketing)**
**Morning: Agentic Workflows & MCP**
- [ ] **Setup MCP SDK:** Install the Model Context Protocol python SDK.
- [ ] **Write `examples/mcp_server.py`:** Expose AXE as an MCP tool (e.g., `extract_web_data(url: str, schema: dict)`).
- [ ] **Test MCP:** Connect the local MCP server to Claude Desktop and ask Claude: *"Go to this URL and extract the pricing plans using the AXE tool."* Ensure it works.
- [ ] **Write LangChain Example:** Create `examples/langchain_agent.py` showing how to wrap the pipeline in a standard `@tool` decorator.

**Afternoon: End-to-End QA & Launch Prep**
- [ ] **The "Blind User" Test:** Create a fresh Python environment. Run `pip install axe-extractor`. Copy-paste the exact code from the README. If it fails, open a high-priority issue for Person A.
- [ ] **Finalize Colab:** Ensure the Colab "Open in Colab" badge is properly linked in the README and works for public users.
- [ ] **Draft Twitter/X Thread:**
  - Post 1: The Hook (Beat massive models with 0.6B).
  - Post 2: Video/GIF of the MCP server working inside Claude Desktop.
  - Post 3: Explain Grounded XPath Resolution (GXR).
  - Post 4: Links to GitHub and Colab.
- [ ] **Draft Reddit/HN Posts:** Write the Hacker News title and the `r/LocalLLaMA` text post focusing heavily on the local execution and open weights.

**Evening: The Launch (Both Members)**
- [ ] **Merge to Main:** Merge both branches into `main`.
- [ ] **Create GitHub Release:** Tag version `v0.1.0` on GitHub and publish the release notes.
- [ ] **Hit Post:** Publish the Twitter thread, Hacker News submission, and Reddit posts.
- [ ] **Monitor:** Hang out in the comments for the next 2 hours answering questions and handling the initial surge of issues.
