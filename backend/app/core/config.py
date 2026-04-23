"""
Configuration centralisée — chemins et constantes de l'application.
Les chemins sont basés sur APP_BASE_DIR (défaut : /app dans Docker).
"""

import os
from pathlib import Path

BASE_DIR   = Path(os.getenv("APP_BASE_DIR", "/app"))
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Fichiers de données
USERS_FILE      = str(DATA_DIR / "users.json")
PREDICTIONS_LOG = str(DATA_DIR / "predictions_log.jsonl")

# Modèles ML
# Chemin pour Python (chargement)
MODEL_PATH = str(MODELS_DIR / "trustpilot_lgbm_model.pkl")

# Chemin pour DVC (relatif par rapport à BASE_DIR)
# Cela donnera "models/trustpilot_lgbm_model.pkl"
DVC_MODEL_PATH = "models/trustpilot_lgbm_model.pkl"

VECTORIZER_PATH = str(MODELS_DIR / "tfidf_vectorizer.pkl")
DVC_VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# Règles métier — quotas
DAILY_QUOTA      = 5    # Prédictions/jour pour les users standard
TOKEN_EXPIRY_DAYS = 30  # Durée de validité d'un token API en jours

# Seuils de monitoring drift
KL_WARNING    = 0.10   # KL divergence : seuil d'avertissement
KL_CRITICAL   = 0.30   # KL divergence : seuil critique → réentraînement
MIN_CONFIDENCE = 0.55  # Confiance minimale acceptable

# Rotation des logs
MAX_LOG_LINES = 50_000  # Nombre max de lignes dans predictions_log.jsonl avant rotation
