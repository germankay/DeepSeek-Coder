# Changelog - CyberCode Studio DeepSeek-Coder V1

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2025-01-01

### Added
- **API Server (`serve/api_server.py`)**: OpenAI-compatible REST API endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/models`).
- **Security Audit Engine (`serve/security_audit.py`)**: Dedicated endpoints for security audit (`/v1/security/audit`), PR code review (`/v1/code/review`), and OWASP-compliant code generation (`/v1/code/generate`).
- **PEFT / LoRA Support**: Integrated LoRA/QLoRA 4-bit/8-bit fine-tuning and standalone adapter merger (`finetune/merge_peft_adapters.py`).
- **Data Anonymization Pipeline**: Data cleansing and PII/secret scrubbing module (`finetune/finetune_cybercode.py`).
- **Autonomous Demo (`demo/app.py`)**: Standalone Gradio user interface with direct security audit tab.
- **Docker & CI/CD**: Multi-stage `Dockerfile`, `docker-compose.yml`, `docker-compose.prod.yml`, and GitHub Actions workflow `.github/workflows/ci.yml`.
- **Governance & Compliance Docs**: Privacy policy, terms, compliance matrix, Model Card, Datasheet, and SBOM.

### Preserved
- Full backward compatibility with original DeepSeek-Coder V1 baseline training, evaluation benchmarks (HumanEval, MBPP, LeetCode, DS-1000, PAL-Math), and CLI tools.
