# 🧠 SentimentAI — Pipeline MLOps End-to-End

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Nginx](https://img.shields.io/badge/Nginx-009639?style=for-the-badge&logo=nginx)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![LightGBM](https://img.shields.io/badge/LightGBM-Machine_Learning-yellow?style=for-the-badge)

Application complète d'**analyse de sentiment** (Positif / Neutre / Négatif) sur des avis clients Trustpilot.
Ce n'est pas un simple notebook Data Science — c'est une **architecture MLOps de production** : API REST sécurisée, reverse proxy Nginx, interface graphique Streamlit, et monitoring de drift du modèle.

---

## 📋 Table des matières

1. [Fonctionnalités](#-fonctionnalités)
2. [Architecture](#-architecture)
3. [Structure du projet](#-structure-du-projet)
4. [Prérequis & Installation](#-prérequis--installation)
5. [Démarrage rapide](#-démarrage-rapide)
6. [Développement local (sans Docker)](#-développement-local-sans-docker)
7. [Gestion des comptes](#-gestion-des-comptes)
8. [API Reference](#-api-reference)
9. [Monitoring de drift](#-monitoring-de-drift)
10. [Sécurité](#-sécurité)
11. [Frontend — Onglets](#-frontend--onglets)
12. [Configuration](#-configuration)
13. [Réentraîner le modèle & MLflow](#-réentraîner-le-modèle)
14. [Résolution de problèmes](#-résolution-de-problèmes)
15. [Technologies](#-technologies)
16. [Licence](#-licence)

---

## ✨ Fonctionnalités

| Catégorie | Détail |
|---|---|
| 🤖 **ML** | Modèle LightGBM + vectoriseur TF-IDF, 3 classes : Négatif / Neutre / Positif |
| 🔍 **Explicabilité** | SHAP TreeExplainer — les mots qui influencent chaque prédiction |
| 🚀 **API REST** | FastAPI avec documentation interactive `/docs` |
| 🛡️ **Sécurité** | Nginx `auth_request`, rate limiting (2 req/s), clés API SHA256 |
| 🚦 **Quotas RBAC** | Utilisateurs : 5 prédictions/jour · Admins : illimité |
| 📊 **Monitoring drift** | Divergence KL entre distribution récente et distribution d'entraînement |
| 🎨 **Frontend** | Interface Streamlit dark glassmorphism, 5 onglets |
| 🐳 **Docker** | 3 services orchestrés (api + nginx + frontend), démarrage en une commande |
| 📝 **Historique** | Log JSONL de toutes les prédictions + feedback utilisateur |

---

## 🏗️ Architecture

```
Utilisateur (navigateur)
        │
        ▼ :8501
┌─────────────────────┐
│      Streamlit      │  Frontend — interface graphique
│     (port 8501)     │  Appels HTTP vers /predict, /token_API, etc.
└──────────┬──────────┘
           │
           ▼ :8080
┌─────────────────────┐
│        Nginx        │  Reverse proxy (expose le port 8080)
│     (port 8080)     │  • Rate limiting : 2 req/s par IP, burst 5
│                     │  • auth_request → /verify_admin avant /predict
└──────────┬──────────┘
           │
           ▼ :80 (interne Docker)
┌─────────────────────┐
│       FastAPI       │  Backend (port 80 interne, 8000 exposé pour /docs)
│  (port 8000 exposé) │  • Auth, prédiction, monitoring, quotas
│                     │  • Charge les modèles ML au démarrage (singleton)
└─────────────────────┘
```

### Flux d'une prédiction

```
Client ──POST /predict──▶ Nginx (8080)
                              │
                        Rate limit check (2 req/s)
                              │
                        auth_request ──▶ FastAPI /verify_admin
                                              │
                                   Vérifie token + quota
                                              │
                              ┌───────────────┴───────────────┐
                           200 OK                          403 Forbidden
                              │                               │
                    Nginx forward /predict            Nginx retourne 403
                              │
                         FastAPI /predict
                              │
                    TF-IDF vectorize ──▶ LightGBM predict
                              │
                    Incrément quota + log JSONL
                              │
                    Retourne {sentiment, score, class_id}
```

> **Note ports :** Nginx écoute en interne sur le port 80 et est exposé sur `8080`. Le backend FastAPI écoute en interne sur le port 80 et est aussi exposé sur `8000` pour accéder à la doc Swagger directement (sans passer par Nginx).

---

## 📁 Structure du projet

```
Test-api/
│
├── docker-compose.yml              # Orchestration des 3 services Docker
├── requirements.txt                # Dépendances Python
├── .env                            # Variables d'environnement (API_KEY) — ne pas committer
├── Procfile                        # Déploiement Heroku (optionnel)
├── readme.md                       # Ce fichier
│
├── backend/                        # Service FastAPI
│   ├── Dockerfile                  # Image Python 3.12-slim, installe requirements.txt
│   ├── __init__.py
│   └── app/
│       ├── main.py                 # Point d'entrée : crée l'app FastAPI, inclut les routers
│       ├── core/
│       │   ├── config.py           # Chemins, constantes (DAILY_QUOTA=5, seuils KL, etc.)
│       │   └── security.py         # hash_password (bcrypt), verify_password, generate_token, require_admin
│       ├── routes/
│       │   ├── auth.py             # POST /login, POST /token_API, GET /verify_admin
│       │   ├── predict.py          # POST /predict (protégé par Nginx auth_request)
│       │   └── monitoring.py       # GET /quota/status, /predictions/history, POST /feedback, GET /monitor/stats
│       ├── schemas/
│       │   └── models.py           # Modèles Pydantic : UserCreate, UserLogin, Review, FeedbackPayload
│       └── services/
│           ├── ml_service.py       # Chargement LightGBM + TF-IDF (singleton), pipeline prédiction
│           ├── monitor_service.py  # Append log JSONL, calcul KL divergence, get_monitoring_stats
│           └── users.py            # load_users() / save_users() — lecture/écriture users.json
│
├── frontend/                       # Service Streamlit
│   ├── Dockerfile                  # Image Python 3.12-slim, installe requirements.txt
│   ├── main.py                     # Point d'entrée Streamlit : routing onglets, gestion session
│   ├── .streamlit/
│   │   └── config.toml             # Thème dark (#0D1117, accent #6366F1)
│   ├── components/
│   │   ├── styles.py               # CSS glassmorphism injecté via st.markdown
│   │   ├── sidebar.py              # Formulaires connexion/inscription, barre quota, infos modèle
│   │   └── tabs/
│   │       ├── demo_tab.py         # ⚡ Analyse Live : saisie texte, score, SHAP, CSV bulk (admin)
│   │       ├── history_tab.py      # 🕑 Historique : 50 dernières prédictions, feedback, donut chart
│   │       ├── dataset_tab.py      # 📊 Infos sur le dataset d'entraînement
│   │       ├── performance_tab.py  # 🤖 Métriques modèle : accuracy 71.8%, F1 0.72, matrice confusion
│   │       └── monitor_tab.py      # 📡 Monitoring drift — admin uniquement
│   └── utils/
│       ├── api_client.py           # Toutes les fonctions HTTP vers le backend (login, predict, feedback…)
│       ├── nlp_pipeline.py         # Préprocessing texte local (NLTK) + chargement modèle pour SHAP
│       └── constants.py            # API_URL, couleurs/emojis des sentiments
│
├── nginx/
│   └── nginx.conf                  # Rate limiting zone, auth_request, proxy_pass vers api
│
├── data/                           # Données persistantes — montées en volumes Docker
│   ├── users.json                  # Base utilisateurs (hash SHA-256, rôles, tokens, quotas)
│   └── predictions_log.jsonl       # Log append-only de chaque prédiction (1 JSON par ligne)
│
├── models/                         # Modèles ML — montés en volume Docker (read-only)
│   ├── trustpilot_lgbm_model.pkl   # Modèle LightGBM entraîné (3 classes)
│   └── tfidf_vectorizer.pkl        # Vectoriseur TF-IDF fitté (5 000 features)
│
└── tests/                          # Tests automatisés (pytest)
    ├── conftest.py                 # Fixtures partagées : client TestClient, mocks modèle ML, fichiers tmp
    ├── test_auth.py                # Tests /login, /token_API, /verify_admin
    └── test_predict.py             # Tests /predict, /quota/status, /predictions/history, /feedback, /health
```

---

## 🛠️ Prérequis & Installation

### Ce qu'il faut avoir installé

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (inclut Docker Compose) — **version 24+**
- Git

### Ce qu'il faut avoir en local

Les fichiers modèles **ne sont pas dans le repo Git** (trop lourds). Il faut les placer manuellement :

```
models/
├── trustpilot_lgbm_model.pkl    ← récupérer depuis le stockage partagé
└── tfidf_vectorizer.pkl         ← récupérer depuis le stockage partagé
```

### Initialiser les fichiers de données

Si le dossier `data/` est vide ou inexistant, créer les fichiers suivants :

**`data/users.json`** — base utilisateurs vide :
```json
{}
```

**`data/predictions_log.jsonl`** — log vide :
```
(fichier vide)
```

> Ces deux fichiers sont montés en volumes Docker. S'ils n'existent pas, Docker les crée comme des dossiers et le backend crashe au démarrage.

### Fichier `.env`

Créer un fichier `.env` à la racine (ne **jamais** committer ce fichier) :

```env
API_KEY=votre_cle_api_secrete
```

> Ce fichier est déjà dans `.gitignore`.

---

## 🚀 Démarrage rapide

```bash
# 1. Cloner le repo
git clone https://github.com/VOTRE_NOM/VOTRE_REPO.git
cd VOTRE_REPO

# 2. Vérifier que les modèles sont présents
ls models/
# → trustpilot_lgbm_model.pkl  tfidf_vectorizer.pkl

# 3. Vérifier que data/ existe avec les bons fichiers
ls data/
# → users.json  predictions_log.jsonl

# 4. Construire et démarrer tous les services
docker compose up --build
```

**Premier démarrage :** 2 à 5 minutes (téléchargement images Docker, installation dépendances, téléchargement ressources NLTK).

### Accéder à l'application

| Service | URL | Description |
|---|---|---|
| Frontend Streamlit | http://localhost:8501 | Interface graphique principale |
| API via Nginx | http://localhost:8080 | Point d'entrée sécurisé de l'API |
| API directe (Swagger) | http://localhost:8000/docs | Documentation interactive, bypass Nginx |

> **Important :** En production, ne pas exposer le port `8000` (FastAPI direct). Tout doit passer par Nginx (`8080`).

### Arrêter les services

```bash
docker compose down
```

### Voir les logs en temps réel

```bash
docker compose logs -f              # Tous les services
docker compose logs -f api          # Backend FastAPI uniquement
docker compose logs -f nginx        # Nginx uniquement
docker compose logs -f frontend     # Streamlit uniquement
```

### Reconstruire un seul service

```bash
docker compose up --build api       # Reconstruire seulement le backend
```

---

## 🔧 Développement local (sans Docker)

Utile pour développer rapidement sans reconstruire les images.

```bash
# Installer les dépendances
pip install -r requirements.txt

# Terminal 1 — Backend FastAPI
APP_BASE_DIR="." uvicorn backend.app.main:app --reload --port 8000

# Terminal 2 — Frontend Streamlit
MODELS_DIR="./models" streamlit run frontend/main.py --server.port 8501
```

> Sans Docker, le frontend appelle directement le backend sur `localhost:8000` (pas via Nginx).
> Modifier `frontend/utils/constants.py` pour changer l'URL de l'API si nécessaire.

---

## 👤 Gestion des comptes

### Créer un compte utilisateur (via l'interface)

1. Ouvrir `http://localhost:8501`
2. Dans la sidebar → **"Créer un compte"**
3. Entrer un nom d'utilisateur et un mot de passe
4. Se connecter → un token API est généré automatiquement

### Créer un compte **admin**

Il n'y a pas de formulaire admin dans l'interface (par sécurité). Trois méthodes :

**Option 1 — Via Swagger UI :**
```
http://localhost:8000/docs
POST /login
{
  "username": "admin",
  "password": "votre_mot_de_passe",
  "role": "admin"
}
```

**Option 2 — Via curl :**
```bash
curl -X POST http://localhost:8000/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "monmdp", "role": "admin"}'
```

**Option 3 — Éditer `data/users.json` directement :**
```json
{
  "admin": {
    "password": "<hash_sha256_du_mot_de_passe>",
    "role": "admin",
    "api_key": null
  }
}
```
> Redémarrer le container `api` après modification manuelle : `docker compose restart api`

### Structure de `data/users.json`

```json
{
  "alice": {
    "password": "$2b$12$...bcrypt_hash...",
    "role": "user",
    "api_key": "64charactershextoken...",
    "token_expires_at": "2026-04-13",
    "daily_count": 3,
    "last_request_date": "2026-03-14"
  },
  "bob": {
    "password": "$2b$12$...bcrypt_hash...",
    "role": "admin",
    "api_key": "64charactershextoken...",
    "token_expires_at": "2026-04-13"
  }
}
```

> - `daily_count` et `last_request_date` n'existent que pour `role: "user"`. Le quota se réinitialise automatiquement chaque jour.
> - `token_expires_at` : date ISO d'expiration du token (30 jours après connexion). Après cette date, le token est refusé.
> - Les mots de passe sont hashés en **bcrypt** (les anciens comptes SHA-256 migrent automatiquement à la prochaine connexion).

---

## 🔌 API Reference

L'API est accessible via Swagger UI sur `http://localhost:8000/docs`.
En production, toutes les requêtes passent par Nginx (`http://localhost:8080`).

### Authentification

| Méthode | Route | Auth requise | Description |
|---|---|---|---|
| `POST` | `/login` | Aucune | Créer un compte (`username`, `password`, `role`) |
| `POST` | `/token_API` | username + password | Obtenir un token API (connexion) |
| `GET` | `/verify_admin` | Interne Nginx | Vérifie token + quota — ne pas appeler directement |

**Créer un compte :**
```json
POST /login
{ "username": "alice", "password": "secret", "role": "user" }
→ { "message": "Utilisateur 'alice' créé avec succès." }
```

**Se connecter :**
```json
POST /token_API
{ "username": "alice", "password": "secret" }
→ {
    "access_token": "07ad35ef...",
    "token_expires_at": "2026-04-13",
    "role": "user",
    "username": "alice"
  }
```

> Le token expire au bout de **30 jours**. Après expiration, une nouvelle connexion via `/token_API` est requise.

### Prédiction

| Méthode | Route | Header requis | Description |
|---|---|---|---|
| `POST` | `/predict` | `X-API-Key: <token>` | Analyser le sentiment d'un texte |

**Requête :**
```json
POST /predict
Header: X-API-Key: 07ad35ef...
Body: { "text": "Ce produit est vraiment excellent, je recommande !" }
```

**Réponse :**
```json
{
  "texte": "Ce produit est vraiment excellent, je recommande !",
  "sentiment": "Positif",
  "prediction_score": "94.2%",
  "class_id": 2
}
```

> `class_id` : `0` = Négatif · `1` = Neutre · `2` = Positif

**Erreurs possibles :**
```json
403  → "Quota journalier atteint (5/5)."
503  → "Trop de requêtes — patientez quelques secondes."  (rate limit Nginx)
401  → "Token invalide."
```

### Monitoring & Historique

| Méthode | Route | Accès | Description |
|---|---|---|---|
| `GET` | `/health` | Tous | Santé de l'API (modèle chargé, version) |
| `GET` | `/quota/status` | Tous | Quota journalier restant |
| `GET` | `/predictions/history` | Tous | Historique personnel (50 dernières) |
| `POST` | `/feedback` | Tous | Marquer une prédiction comme correct/incorrect |
| `GET` | `/monitor/stats` | **Admin** | Statistiques de drift (KL, distributions, timeline) |

**Exemple `/quota/status` :**
```json
{
  "username": "alice",
  "role": "user",
  "quota_used": 3,
  "quota_max": 5,
  "remaining": 2
}
```

**Exemple `/predictions/history` :**
```json
{
  "username": "alice",
  "history": [
    {
      "timestamp": "2026-03-14T10:32:11.123456",
      "text_length": 45,
      "text_preview": "Ce produit est vraiment...",
      "prediction": "Positif",
      "confidence": 0.942,
      "class_id": 2,
      "feedback": "correct"
    }
  ],
  "count": 1
}
```

**Exemple `/monitor/stats` (admin) :**
```json
{
  "status": "ok",
  "total_predictions": 150,
  "recent_predictions": 45,
  "avg_confidence_recent": 0.783,
  "kl_divergence": 0.057,
  "drift_level": "normal",
  "needs_retraining": false,
  "confidence_alert": false,
  "recent_distribution": { "Négatif": 0.20, "Neutre": 0.30, "Positif": 0.50 },
  "training_distribution": { "Négatif": 0.333, "Neutre": 0.333, "Positif": 0.333 },
  "feedback_accuracy": 85.5,
  "daily_timeline": [...]
}
```

---

## 📡 Monitoring de drift

Le système surveille en continu si le modèle doit être réentraîné.

### Principe

À chaque prédiction, une entrée est ajoutée dans `data/predictions_log.jsonl` :
```json
{
  "timestamp": "2026-03-14T10:32:11.123456",
  "username": "alice",
  "text_length": 87,
  "text_preview": "Ce produit est vraiment...",
  "prediction": "Positif",
  "confidence": 0.9421,
  "class_id": 2,
  "feedback": null
}
```
Le champ `feedback` est mis à jour quand l'utilisateur clique 👍 ou 👎.

### Divergence KL (Kullback-Leibler)

Mesure l'écart entre la distribution récente des prédictions et la distribution d'entraînement (1/3 chaque classe) :

| Niveau | KL divergence | Action recommandée |
|---|---|---|
| 🟢 Normal | < 0.10 | Rien à faire |
| 🟡 Warning | 0.10 – 0.30 | Surveiller, surtout si confiance basse |
| 🔴 Critique | ≥ 0.30 | **Réentraîner le modèle** |

### Seuil de confiance

Si la confiance moyenne des prédictions récentes tombe sous **55%**, une alerte est déclenchée même si le KL est en mode "warning".

### Onglet Monitor (admin)

L'onglet `📡 Monitor` affiche :
- Statut actuel (🟢 normal / 🟡 warning / 🔴 critique)
- KL divergence en temps réel
- Distribution récente vs distribution d'entraînement
- Timeline journalière (volume + confiance)
- Accuracy basée sur les feedbacks utilisateurs

---

## 🛡️ Sécurité

### Nginx

- **Rate limiting** : max 2 requêtes/seconde par IP, burst de 5 autorisé (`limit_req_zone`)
- **auth_request** : avant chaque appel à `/predict`, Nginx interroge `/verify_admin` pour valider le token et le quota. La requête est rejetée avant d'atteindre FastAPI si invalide.

### Backend

- Mots de passe hashés en **bcrypt** (`passlib`) — résistant aux attaques par force brute
- Migration automatique : les anciens hash SHA-256 sont rehashés en bcrypt à la prochaine connexion
- Tokens API générés avec `secrets.token_hex(32)` — 64 caractères hexadécimaux, cryptographiquement sûrs
- **Expiration des tokens** : chaque token est invalide 30 jours après émission (`token_expires_at`)
- Quota journalier réinitialisé à minuit automatiquement (comparaison date ISO)
- Admins exemptés du quota
- **CORS** configuré sur les origines autorisées (`localhost:8501`, `localhost:8080`)
- **`/health`** endpoint pour vérifier l'état de l'API et du modèle
- **`threading.Lock`** sur la mise à jour du quota (protection contre les race conditions)
- Logs d'erreurs structurés via le module `logging` standard

### Bonnes pratiques

- Ne **jamais** committer `.env`, `data/users.json` avec de vrais mots de passe, ou les fichiers `.pkl`
- En production, ne pas exposer le port `8000` (FastAPI direct) — tout doit passer par Nginx
- Changer les mots de passe par défaut dès le premier déploiement
- En production, restreindre les origines CORS dans `main.py`

---

## 🎨 Frontend — Onglets

| Onglet | Accès | Description |
|---|---|---|
| ⚡ **Analyse Live** | Connecté | Analyse un texte, affiche sentiment + score de confiance + graphique SHAP des mots impactants |
| 🕑 **Historique** | Connecté | Les 50 dernières prédictions personnelles, filtrage par sentiment, feedback, donut chart |
| 📊 **Dataset** | Connecté | Statistiques et exploration du dataset d'entraînement Trustpilot |
| 🤖 **Performance** | Connecté | Métriques du modèle (accuracy 71.8%, F1-Score 0.72, matrice de confusion, top mots) |
| 📡 **Monitor** | **Admin** | Monitoring drift en temps réel, timeline journalière, alertes réentraînement |

> L'onglet **Monitor** n'est visible que pour les comptes avec `role: "admin"`.

---

## ⚙️ Configuration

### Variables d'environnement

| Variable | Défaut | Fichier | Description |
|---|---|---|---|
| `APP_BASE_DIR` | `/app` | `docker-compose.yml` | Répertoire racine dans Docker (chemins vers data + models) |
| `MODELS_DIR` | `/app/models` | `docker-compose.yml` | Répertoire des modèles (frontend, pour SHAP) |
| `API_KEY` | — | `.env` | Clé API externe (non utilisée en interne, réservée pour intégrations futures) |

### Constantes métier (`backend/app/core/config.py`)

```python
DAILY_QUOTA       = 5       # Prédictions/jour pour les users standard
TOKEN_EXPIRY_DAYS = 30      # Durée de validité d'un token API (jours)
KL_WARNING        = 0.10    # Seuil d'alerte drift (KL divergence)
KL_CRITICAL       = 0.30    # Seuil critique drift → réentraînement recommandé
MIN_CONFIDENCE    = 0.55    # Confiance minimale acceptable (55%)
MAX_LOG_LINES     = 50_000  # Rotation automatique du log JSONL au-delà de ce seuil
```

---

## 🧪 Tests

Les tests utilisent **pytest** et **FastAPI TestClient**. Le modèle ML est mocké — pas besoin des vrais fichiers `.pkl`.

### Lancer les tests

```bash
# Installer les dépendances de test
pip install pytest httpx

# Lancer tous les tests depuis la racine du projet
pytest tests/ -v
```

### Ce qui est testé

| Fichier | Couverture |
|---|---|
| `test_auth.py` | Inscription (succès, doublon, validations username/password/role), connexion (succès, mauvais mdp, user inconnu), `/verify_admin` |
| `test_predict.py` | Prédiction (succès, sans token, token invalide, texte vide/blanc), incrémentation quota, `/quota/status` user/admin, `/predictions/history`, `/feedback`, `/health` |

### Structure des fixtures (`conftest.py`)

- **`tmp_users_file`** : `users.json` vide dans un dossier temporaire (isolé entre tests)
- **`tmp_log_file`** : `predictions_log.jsonl` vide temporaire
- **`client`** : `TestClient` FastAPI avec modèle mocké (`MagicMock` retournant sentiment "Positif" à 85%)
- **`register_and_login()`** : helper pour créer un compte et récupérer un token en une ligne

---

## 🔁 Réentraîner le modèle

Si le monitoring indique un drift critique (`kl_divergence ≥ 0.30`), utiliser le script de réentraînement intégré avec tracking MLflow.

### Prérequis

```bash
pip install -r training/requirements_train.txt
```

### Lancer l'entraînement

```bash
# Entraînement avec les paramètres par défaut
python training/train.py --data path/to/df_merged_clean.csv

# Avec des hyperparamètres personnalisés
python training/train.py \
  --data path/to/df_merged_clean.csv \
  --max-features 20000 \
  --num-leaves 64 \
  --learning-rate 0.05 \
  --n-estimators 1000
```

Le script effectue automatiquement :
- Nettoyage du CSV (filtrage, gestion des NaN)
- Préprocessing texte (lowercase → suppression ponctuation/chiffres → tokenisation → stopwords → lemmatisation)
- Regroupement des labels (1,2→Négatif · 3→Neutre · 4,5→Positif)
- Rééquilibrage des classes (sous-échantillonnage)
- Vectorisation TF-IDF
- Entraînement LightGBM avec early stopping
- **Log MLflow** : hyperparamètres, métriques, artefacts (matrice de confusion, feature importance, rapport de classification)
- **Sauvegarde automatique** des `.pkl` dans `models/`

### Voir les résultats dans MLflow

```bash
# Démarrer le serveur MLflow
docker compose up mlflow

# Ou sans Docker
mlflow ui --backend-store-uri "file:///C:/Users/Lakdar/Test-api/mlflow" --port 5000
```

Ouvrir `http://localhost:5000` pour comparer les runs.

### Ce qui est loggé dans MLflow

| Type | Détail |
|---|---|
| **Paramètres** | max_features, learning_rate, num_leaves, n_estimators, ngram_range, subsample… |
| **Métriques** | accuracy, f1_weighted, f1 par classe (Négatif/Neutre/Positif), precision par classe, best_iteration |
| **Artefacts** | `confusion_matrix.png`, `feature_importance.png`, `classification_report.txt`, `.pkl` du modèle |
| **Model Registry** | Modèle enregistré sous le nom `SentimentAI-LightGBM` avec versioning automatique |

### Déployer le nouveau modèle

Après entraînement, les `.pkl` sont automatiquement sauvegardés dans `models/`. Redémarrer les services :

```bash
docker compose restart api frontend
```

> Le modèle est chargé en mémoire au démarrage (singleton dans `ml_service.py`). Un simple restart suffit — pas besoin de rebuild.

---

## 🐛 Résolution de problèmes

### Le container `api` crashe au démarrage

**Cause probable :** Les fichiers `data/users.json` ou `data/predictions_log.jsonl` n'existent pas.

```bash
# Vérifier
ls data/

# Créer les fichiers manquants
echo '{}' > data/users.json
touch data/predictions_log.jsonl
docker compose restart api
```

### Les modèles ne se chargent pas

**Cause probable :** Les fichiers `.pkl` sont absents du dossier `models/`.

```bash
ls models/
# Si vide → placer les fichiers trustpilot_lgbm_model.pkl et tfidf_vectorizer.pkl
docker compose restart api frontend
```

### Le frontend affiche "Connexion refusée"

**Cause probable :** Nginx ou le backend n'est pas encore prêt.

```bash
docker compose ps          # Vérifier le statut de chaque service
docker compose logs nginx  # Voir les erreurs Nginx
docker compose logs api    # Voir les erreurs FastAPI
```

### `403 Forbidden` sur `/predict`

Causes possibles (dans l'ordre) :
1. Token API invalide ou expiré → se reconnecter
2. Quota journalier atteint (5/5 pour les users) → attendre minuit ou passer admin
3. Rate limit Nginx déclenché (trop de requêtes) → attendre quelques secondes

### SHAP ne s'affiche pas

**Cause probable :** Le frontend n'arrive pas à charger les modèles localement (variable `MODELS_DIR`).

```bash
docker compose logs frontend | grep -i shap
# Vérifier que MODELS_DIR=/app/models est bien défini dans docker-compose.yml
```

### Reconstruire proprement depuis zéro

```bash
docker compose down --volumes --rmi all
docker compose up --build
```

> ⚠️ `--volumes` supprime les volumes nommés mais **pas** les bind mounts (`./data/`). Vos données utilisateurs et logs sont préservés.

---

## 🗃️ Technologies

| Composant | Technologie | Détail |
|---|---|---|
| **Langage** | Python 3.12 | Backend + Frontend |
| **API** | FastAPI + Uvicorn | Routing, validation Pydantic, async |
| **ML** | LightGBM | Classification 3 classes (Négatif/Neutre/Positif) |
| **Vectorisation** | scikit-learn TF-IDF | 5 000 features, fitté sur Trustpilot |
| **Explicabilité** | SHAP TreeExplainer | Top mots impactants par prédiction |
| **NLP** | NLTK | Tokenisation, lemmatisation, stopwords |
| **Proxy** | Nginx | Rate limiting, auth_request, reverse proxy |
| **Frontend** | Streamlit + Altair | Interface dark glassmorphism |
| **Conteneurisation** | Docker + Docker Compose | 3 services orchestrés |
| **Persistance** | JSON (users) + JSONL (logs) | Fichiers montés en volumes Docker |
| **Experiment tracking** | MLflow | Hyperparamètres, métriques, artefacts, model registry |
| **Versioning data** | DVC | Configuration présente (`.dvc/`) |
| **Sérialisation** | Joblib | Sauvegarde/chargement des modèles `.pkl` |
 
---

## 📄 Licence

Projet personnel / éducatif. Libre de réutilisation avec attribution.
