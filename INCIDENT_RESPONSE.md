# PLAN DE RÉPONSE AUX INCIDENTS - CYBERCODE STUDIO

Ce document définit les procédures d'urgence (*Incident Response*) pour l'infrastructure DeepSeek-Coder V1 de CyberCode Studio.

---

## 1. FUITE DE DONNÉES OU SECRETS EXPOSÉS

**Symptômes** : Détection d'une clé API, d'un jeton JWT ou d'une donnée confidentielle dans les logs ou un dépôt public.

### Procédure d'Urgence :
1. **Révocation immédiate des secrets** :
   - Invalider la clé API compromise via HashiCorp Vault ou AWS Secrets Manager.
   - Mettre à jour la variable `CYBERCODE_API_KEY` sur le serveur de production.
2. **Rotation des jetons** :
   - Générer une nouvelle clé API et redistribuer aux services autorisés via canal sécurisé.
3. **Purge et Isolation** :
   - Purger les caches de requêtes et supprimer les fichiers de logs temporaires non anonymisés.
4. **Notification** :
   - Informer le Responsable de la Sécurité des Systèmes d'Information (RSSI) et l'équipe conformité RGPD.

---

## 2. ATTAQUE OU SURCHARGE DoS / DDoS

**Symptômes** : Pic anormal de trafic, taux d'erreur API > 5%, latence TTFT > 2000 ms.

### Procédure d'Urgence :
1. **Activation du Rate Limiting Strict** :
   - Diminuer la limite globale (`RATE_LIMIT_PER_MINUTE=20`) dans `serve/api_server.py` ou la passerelle Traefik/Nginx.
2. **Blocage d'Adresses IP** :
   - Identifier les IP clientes malveillantes via les logs anonymisés et ajouter une règle de blocage `iptables` ou Cloudflare.
3. **Bascule en Mode Dégradé** :
   - Rediriger les requêtes vers un modèle plus léger ou désactiver les requêtes d'audit intensives temporairement.

---

## 3. MODÈLE OU ADAPTATEUR COMPROMIS

**Symptômes** : Comportement anormal du modèle, injection d'instructions (Prompt Injection), ou altération des poids LoRA.

### Procédure d'Urgence :
1. **Isolation du Conteneur** :
   - Stopper le service compromis : `docker compose stop api demo`.
2. **Rollback Immédiat** :
   - Restaurer l'adaptateur LoRA ou le modèle baseline V1 précédent depuis la sauvegarde saine.
3. **Audit de Sécurité des Poids** :
   - Re-vérifier l'intégrité SHA-256 des fichiers de poids par rapport au registre officiel.
4. **Relance sécurisée** :
   - Redémarrer les services et valider via la suite de tests `/healthz` et `pytest tests/`.
