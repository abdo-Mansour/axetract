# Contributing to AXEtract

Thank you for your interest in contributing to AXEtract! We welcome contributions in many forms, including bug reports, feature requests, documentation improvements, and code contributions.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/your-username/axetract.git
    cd axetract
    ```
3.  **Install development dependencies** using `uv`:
    ```bash
    uv sync --dev
    ```
4.  **Install pre-commit hooks**:
    ```bash
    uv run pre-commit install
    ```

## Development Workflow

### Coding Standards

We use `ruff` to enforce coding standards. Please ensure your code passes linting and formatting before submitting a PR.
```bash
uv run ruff check .
uv run ruff format .
```

### Docstrings

We follow the **Google Style** for docstrings. All new public functions, classes, and modules should have clear documentation.

### Testing

Please add unit tests for any new features or bug fixes. Run tests using `pytest`:
```bash
uv run pytest tests/
```

## Submitting a Pull Request

1.  Create a new branch for your changes:
    ```bash
    git checkout -b feature/my-new-feature
    ```
2.  Commit your changes following the [Conventional Commits](https://www.conventionalcommits.org/) specification (recommended).
3.  Push your branch to your fork.
4.  Submit a Pull Request to the `main` branch of the original repository.

## Feedback

If you have questions or feedback, please open an issue on GitHub.
