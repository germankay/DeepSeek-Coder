# GUIDES D'UTILISATION ET ARCHITECTURE CYBERCODE STUDIO

## 1. MIGRATION ET COMPATIBILITÉ DEEPSEEK-CODER V1

Cette version personnalisée conserve l'intégralité des fonctionnalités baseline du dépôt officiel DeepSeek-Coder V1 (`deepseek-ai/deepseek-coder-1.3b-instruct`, `6.7b-instruct`, `33b-instruct`).
- **Verrouillage V1** : L'utilisation de modèles V2, V3, R1 ou dérivés est strictement bloquée dans le code (`DEFAULT_MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"`).
- **Rétrocompatibilité** : Tous les scripts d'évaluation originaux (`Evaluation/`) et le fine-tuning Full ZeRO-3 fonctionnent sans aucune modification de leurs commandes CLI.

---

## 2. NOUVELLES FONCTIONNALITÉS ET EXTENSIONS

### 2.1. Fine-Tuning LoRA / QLoRA
- Script étendu : `finetune/finetune_deepseekcoder.py`
  ```bash
  python finetune/finetune_deepseekcoder.py \
    --model_name_or_path deepseek-ai/deepseek-coder-6.7b-instruct \
    --data_path ./data/dataset.json \
    --output_dir ./models/cybercode-lora \
    --use_peft --load_in_4bit --lora_r 16 --lora_alpha 32
  ```
- Anonymisation et Fine-tuning Continu : `finetune/finetune_cybercode.py`
  - Supprime automatiquement les adresses IP, jetons Bearer, clés API et e-mails des données clients avant l'entraînement.
- Fusion des adaptateurs LoRA : `finetune/merge_peft_adapters.py`
  ```bash
  python finetune/merge_peft_adapters.py \
    --base_model_path deepseek-ai/deepseek-coder-6.7b-instruct \
    --adapter_path ./models/cybercode-lora \
    --output_dir ./models/merged-model
  ```

### 2.2. API Serveur & Cybersécurité (`serve/`)
- Endpoints OpenAI v1 :
  - `POST /v1/chat/completions`
  - `POST /v1/completions` (support FIM avec `<｜fim begin｜>`, `<｜fim hole｜>`, `<｜fim end｜>`)
  - `GET /v1/models`
- Endpoints CyberCode Studio :
  - `POST /v1/security/audit` : Audit de sécurité du code.
  - `POST /v1/code/review` : Revue automatisée de PR.
  - `POST /v1/code/generate` : Génération de code sécurisé respectant OWASP Top 10.

---

## 3. COMPARATIF ET ESTIMATION DES COÛTS

### 3.1. Estimation des Coûts d'Infrastructure
| Élément | Spécification | Coût Mensuel Estimé |
| :--- | :--- | :--- |
| **VPS / Cloud GPU** | NVIDIA A10G / RTX 4090 (24 Go VRAM) | 150 € - 300 € |
| **Stockage Sécurisé** | SSD NVMe 500 Go (Chiffrement repos AES-256) | ~30 € |
| **Bande Passante & Trafic API** | Reverse Proxy TLS + CDN | ~20 € |
| **Total Estimé** | Environnement de production complet | **200 € - 350 € / mois** |

### 3.2. Comparatif Inférence : vLLM vs Ollama
| Critère | vLLM (Choix Production) | Ollama (Usage Dev) |
| :--- | :--- | :--- |
| **Débit (Tokens/sec)** | Très élevé (PagedAttention) | Moyen |
| **Contrôle API / FIM** | Total (FastAPI / native) | Simplifié |
| **Support LoRA Dynamique** | Oui | Limité |
| **Usage Recommandé** | Production VPS / Cluster | Tests locaux développeurs |

---

## 4. PLAN DE ROLLBACK ET STRATÉGIE DE REPLI

1. **Déploiement Blue-Green / Canary** :
   - Deux conteneurs API/Inférence tournent en parallèle (`api-blue` et `api-green`).
   - Le reverse proxy (Nginx / Traefik) redirige 10 % du trafic sur la nouvelle version (Canary) avant bascule complète.
2. **Healthcheck Automatique & Healthz** :
   - Endpoint `/healthz` interrogé toutes les 15 secondes.
   - Si la nouvelle version renvoie un échec ou ne répond pas dans les 60 secondes, déclenchement immédiat du rollback vers la version stable précédente.
