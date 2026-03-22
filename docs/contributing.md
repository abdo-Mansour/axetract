# Contributing

First off, thank you for considering contributing to Axetract! It's people like you who make Axetract such a great tool.

## Development Setup

Axetract uses `uv` for dependency management and project orchestration.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abdo-Mansour/axetract.git
   cd axetract
   ```

2. **Install dependencies**:
   ```bash
   uv sync --all-groups
   ```

3. **Run tests**:
   ```bash
   uv run pytest
   ```

## Code Style

- We use `ruff` for linting and formatting.
- Ensure all new functions have descriptive docstrings (we use the Google style).
- Type hints are mandatory for all public APIs.

## Pull Request Process

1. Create a new branch for your feature or bugfix.
2. Add tests for any new functionality.
3. Ensure the documentation reflects your changes.
4. Submit a Pull Request to the `main` branch.

## Training Custom Adapters

If you are interested in contributing training scripts or new LoRA adapters, please reach out via GitHub Issues. We are looking for adapters trained on:
- E-commerce product pages
- Real Estate listings
- Academic papers
- Financial reports
