# Compliance & Governance Standard - CyberCode Studio

## Regulatory Compliance Matrix

### 1. GDPR (General Data Protection Regulation)
- Data minimization and strict anonymization filters (`finetune/finetune_cybercode.py`).
- Right to be forgotten supported for custom fine-tuning datasets.

### 2. ISO/IEC 27001 & SOC 2 Type II Alignment
- **Access Control**: Role-based access control and Bearer token authentication.
- **Secrets Management**: Integration with HashiCorp Vault.
- **Audit Logging**: Structured JSON logging for API requests without sensitive code exposure.

### 3. Software Supply Chain Security (SBOM)
- Automated Software Bill of Materials generation provided in `SBOM.json` compliant with CycloneDX standards.
