# Master Prompt d'Implémentation Globale - CyberCode Studio

> **Instructif** : Ce prompt est le guide maître pour tout développeur ou agent IA chargé d'exécuter la feuille de route d'amélioration de CyberCode Studio en s'appuyant sur les fichiers `analyse-obiey/analyse.md` et `analyse-obiey/amelioration.md`.

---

## Directives & Master Prompt

```markdown
Tu es un Ingénieur Expert en Cybersécurité, MLOps et Architecte Backend FastAPI/vLLM Senior.

### Contexte & Contraintes Absolues :
1. **Modèle Unique & Verrouillé** : Le projet repose EXCLUSIVEMENT sur les modèles **DeepSeek-Coder V1** (`1.3b-instruct`, `6.7b-instruct`, `33b-instruct`). Il est STRICTEMENT INTERDIT d'utiliser, d'importer ou de faire référence aux modèles DeepSeek V2, V3 ou R1. Les garde-fous existants (exceptions `ValueError` sur `v2`, `v3`, `r1`) doivent être rigoureusement maintenus.
2. **Rétrocompatibilité** : Toutes les fonctionnalités existantes (API OpenAI V1, fine-tuning LoRA, anonymisation RGPD, démo Gradio, scripts d'évaluation) doivent continuer à fonctionner sans régression.
3. **Qualité & Tests** : Chaque modification doit s'accompagner de tests unitaires ou d'intégration automatisés.

---

### Objectif : Implémenter l'Ensemble des 9 Axes d'Amélioration de CyberCode Studio

Réalise l'implémentation progressive et modulaire des améliorations suivantes :

#### 1. Moteur d'Audit Hybride (Regex + AST + LLM + CVE/CWE)
- Étends `SecurityAuditor` dans `serve/security_audit.py` avec un analyseur AST Python (`ast.parse`) pour vérifier les nœuds syntaxiques et éliminer les faux positifs causés par les commentaires ou les littéraux inoffensifs.
- Intègre la classification CWE (ex: CWE-89, CWE-79, CWE-78) pour chaque vulnérabilité identifiée.

#### 2. Rapport d'Audit Enrichi (`SecurityAuditReport`)
- Ajoute les champs Pydantic suivants à `VulnerabilityItem` :
  - `confidence_score` : `high` | `medium` | `low`
  - `cwe_id` : identifiant CWE officiel
  - `code_before` / `code_after` : extrait de code avant et après correctif recommandé.

#### 3. Inférence, File d'Attente & Gestion VRAM
- Ajoute la gestion des retries et le support asynchrone des requêtes lourdes dans `serve/api_server.py`.
- Configure un garde-fou VRAM pour vLLM (`--gpu-memory-utilization 0.90`) dans `serve/vllm_server.py`.

#### 4. Observabilité & Métriques Prometheus (`/metrics`)
- Expose le point de terminaison `/metrics` dans `serve/api_server.py` fournissant les compteurs de requêtes par endpoint, les latences d'exécution et le score moyen de sécurité des audits.

#### 5. Tests de Charge Locust
- Crée le fichier `tests/locustfile.py` simulant les appels concurrents sur `/v1/chat/completions` et `/v1/security/audit` avec mesure des temps de réponse (TTFT, TPOT).

#### 6. Erreurs RFC 7807 & Validation Utilisateur
- Implémenter un middleware de gestion globale des exceptions dans FastAPI retournant des réponses d'erreur structurées selon la spécification RFC 7807 (Problem Details).
- Limiter la taille maximale des payloads d'audit à 5 Mo (413 Payload Too Large).

#### 7. Versioning LoRA & MLflow Registry
- Dans `finetune/finetune_cybercode.py`, ajouter le logging optionnel des hyperparamètres et métriques de loss via MLflow ou Weights & Biases.

#### 8. Scalabilité Multi-GPU & Conteneurisation
- Mettre à jour le `Dockerfile` pour s'exécuter en tant qu'utilisateur non-root (`USER 10001`).
- Fournir les manifestes Kubernetes de base dans `deploy/k8s/` pour le déploiement multi-instances.

#### 9. Abstraction Model Engine & Plan de Secours (Exit Strategy)
- Créer la classe abstraite `BaseModelProvider` dans `serve/providers/` permettant de basculer de manière transparente sur un modèle alternatif si le modèle principal est indisponible.
```
