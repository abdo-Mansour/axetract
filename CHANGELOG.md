# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-04

### Added
- Initial release of AXEtract.
- Four-stage decoupled pipeline: Preprocessing, Pruning, Extraction, Postprocessing.
- Grounded XPath Resolution (GXR) for JSON field verification.
- Specialized LoRA-powered adapters for task-specific models.
- Support for vLLM and HuggingFace LLM engines.
- FastAPI server implementation for high-throughput serving.
- Structural logging system replacing print statements.
- Comprehensive CI/CD with GitHub Actions.
- Enhanced API surface with top-level exports in `axetract` package.
- Professional package metadata and documentation (Contributing, Security, Changelog).
- Pre-commit hooks for automated code quality.
- Custom exception hierarchy for graceful error handling.
- Optimized dependencies with optional `vllm` extra.
