"""
Chargement et utilisation du modèle LightGBM + TF-IDF.
"""

import logging
import os
import subprocess
import joblib

from ..core.config import MODEL_PATH, VECTORIZER_PATH
from common.nlp_pipeline import processing_pipeline


logger = logging.getLogger(__name__)

LABELS = {0: "Négatif", 1: "Neutre", 2: "Positif"}

_model      = None
_vectorizer = None


def get_model():
    """Charge le modèle une seule fois (singleton)."""
    global _model, _vectorizer
    if _model is None:
        try:
            # Pull DVC uniquement si les fichiers sont absents (pas en Docker avec volume)
            if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
                try:
                    subprocess.run(["dvc", "pull", "models"], check=True)
                except Exception as dvc_err:
                    logger.warning("DVC pull impossible (normal en Docker) : %s", dvc_err)

            _model      = joblib.load(MODEL_PATH)
            _vectorizer = joblib.load(VECTORIZER_PATH)
            logger.info("Modèle LightGBM et TF-IDF chargés avec succès.")
        except Exception as e:
            logger.error("Erreur lors du chargement du modèle : %s", e)
    return _model, _vectorizer


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