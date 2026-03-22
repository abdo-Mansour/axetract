# Security Policy

## Supported Versions

Users of AXEtract are encouraged to upgrade to the latest version as soon as possible. Only the latest version is actively supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| v0.1.0  | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in AXEtract, please do NOT open a public issue. Instead, report it privately to the maintainers:

- **Abdelrahman Mansour**: abdelrahman.f.mansour@gmail.com
- **Khaled Alshaer**: khaled.w.alshaer@gmail.com

We will acknowledge your report and work with you to resolve the issue as quickly as possible. Please include as much detail as possible, including steps to reproduce the vulnerability.

## Secure Usage

- **API Keys**: Avoid hardcoding any API keys or secrets in your extraction scripts or configuration. Use environment variables (e.g., `os.getenv`).
- **Data Privacy**: AXEtract processes web content. Ensure you have the right to scrape or extract data from any URL you provide to the pipeline.
- **Model Credentials**: When using remote LLM engines, protect your authentication tokens.
