"""
Chargement et utilisation du modèle LightGBM + TF-IDF.
"""

import logging
import os
import subprocess
import joblib

from ..core.config import BASE_DIR, MODEL_PATH, VECTORIZER_PATH, DVC_MODEL_PATH, DVC_VECTORIZER_PATH
from common.nlp_pipeline import processing_pipeline

logger = logging.getLogger(__name__)

LABELS = {0: "Négatif", 1: "Neutre", 2: "Positif"}

_model      = None
_vectorizer = None


def _pull_models_from_s3() -> bool:
    """
    Télécharge les modèles depuis DagsHub S3 via DVC.
    Retourne True si le pull a réussi, False sinon.
    """
    try:
        logger.info("Pull DVC depuis DagsHub S3...")
        result = subprocess.run(
            ["dvc", "pull", DVC_MODEL_PATH, DVC_VECTORIZER_PATH],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
        )
        logger.info("DVC pull OK : %s", result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        logger.warning("DVC pull échoué : %s", e.stderr.strip())
        return False


def get_model():
    """Charge le modèle une seule fois (singleton). Tente un DVC pull si les fichiers sont absents."""
    global _model, _vectorizer
    if _model is None:
        try:
            # Pull DVC si les fichiers sont absents
            if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
                _pull_models_from_s3()

            _model      = joblib.load(MODEL_PATH)
            _vectorizer = joblib.load(VECTORIZER_PATH)
            logger.info("Modèle LightGBM et TF-IDF chargés avec succès.")
        except Exception as e:
            logger.error("Erreur lors du chargement du modèle : %s", e)
    return _model, _vectorizer


def reload_model():
    """
    Force le rechargement du modèle depuis DagsHub S3 via DVC.
    Appelé par Airflow après un réentraînement réussi.
    """
    global _model, _vectorizer

    logger.info("Rechargement du modèle demandé — pull depuis DagsHub S3...")

    # 1. Pull le nouveau modèle depuis S3
    success = _pull_models_from_s3()
    if not success:
        raise RuntimeError("Impossible de télécharger le nouveau modèle depuis DagsHub S3.")

    # 2. Réinitialise le singleton pour forcer le rechargement
    _model      = None
    _vectorizer = None

    # 3. Recharge en mémoire
    model, vectorizer = get_model()

    if model is None or vectorizer is None:
        raise RuntimeError("Modèle téléchargé mais impossible de le charger en mémoire.")

    logger.info("Nouveau modèle rechargé avec succès depuis DagsHub S3.")
    return model, vectorizer


def predict(text: str) -> dict:
    """Prédit le sentiment d'un texte. Retourne sentiment, confidence, class_id."""
    model, vectorizer = get_model()
    if model is None or vectorizer is None:
        raise RuntimeError("Modèle non disponible.")
    clean_text = processing_pipeline(text)
    vec        = vectorizer.transform([clean_text])
    class_id   = int(model.predict(vec)[0])
    confidence = 100.0

    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(vec)[0]
        confidence = round(float(max(proba)) * 100, 2)

    return {
        "sentiment":  LABELS.get(class_id, "Inconnu"),
        "confidence": confidence,
        "class_id":   class_id,
    }