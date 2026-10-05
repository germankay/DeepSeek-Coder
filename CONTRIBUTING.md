# Contributing to CyberCode Studio DeepSeek-Coder V1

Thank you for your interest in contributing to CyberCode Studio!

## Guiding Principles
1. **Strict Lock on DeepSeek-Coder V1**: All changes must exclusively target DeepSeek-Coder V1 model weights (`1.3B`, `6.7B`, `33B`). No references or migrations to V2, V3, or R1 are permitted.
2. **Backward Compatibility**: Existing scripts, training workflows, and evaluation benchmarks must continue to function without disruption.
3. **Quality & Security**: All contributions must pass static security scanning (`bandit`), linting (`ruff`), and unit test suites (`pytest`).

## Development Workflow
1. Fork the repository and create a feature branch (`git checkout -b feature/my-feature`).
2. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
3. Run code quality tools:
   ```bash
   ruff check .
   black --check .
   PYTHONPATH=. python3 -m pytest
   ```
4. Submit a Pull Request targeting `main` using the Pull Request template.
