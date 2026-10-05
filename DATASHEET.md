# Datasheet - CyberCode Studio Dataset & Adapters

## Dataset Provenance
- **Base Pre-training Corpus**: DeepSeek-Coder V1 pre-training dataset consisting of 2 Trillion tokens across 87 programming languages.
- **CyberCode Studio Fine-Tuning Corpus**: Curated security-focused code review datasets, OWASP vulnerability fix examples, and clean software engineering samples.

## Data Cleansing & Anonymization
- **Secret Removal**: High-entropy strings, AWS keys, JWT tokens, and private RSA keys automatically stripped.
- **PII Scrubbing**: Names, emails, IP addresses, and custom domain endpoints sanitized.
