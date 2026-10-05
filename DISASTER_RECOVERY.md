# Disaster Recovery & Backup Plan

## Backup Protocols
1. **LoRA Adapters & Configs**: Daily automated backup to encrypted offsite object storage.
2. **Environment & Keys**: Vault key backups managed via encrypted enterprise snapshots.

## Recovery Time Objective (RTO) & Recovery Point Objective (RPO)
- **RTO**: < 30 minutes for API service restoration via Docker Compose / Helm.
- **RPO**: < 1 hour for custom fine-tuned LoRA model weights.
