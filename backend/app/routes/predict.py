"""
Route de prédiction de sentiment.
"""

import logging
import threading
from datetime import date
import pandas as pd
import json
import base64
import io
from typing import List

from fastapi import APIRouter, HTTPException, Depends

from ..schemas.models import Review, Base64
from ..services import ml_service, monitor_service
from ..services.users import get_users, save_users
from ..core.security import get_api_key

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Prediction"])

# Verrou pour éviter les race conditions sur la mise à jour du quota
_quota_lock = threading.Lock()


@router.post("/predict")
def predict_sentiment(review: Review, api_key: str = Depends(get_api_key)):
    """Prédit le sentiment d'un texte et consomme 1 crédit quota."""
    try:
        result = ml_service.predict(review.text)
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Modèle non chargé.")

    # Mise à jour atomique du quota
    username = "unknown"
    with _quota_lock:
        users = get_users()
        for uname, user_data in users.items():
            if user_data.get("api_key") == api_key:
                username = uname
                if user_data.get("role") != "admin":
                    today = date.today().isoformat()
                    if user_data.get("last_request_date") != today:
                        user_data["daily_count"]       = 0
                        user_data["last_request_date"] = today
                    user_data["daily_count"] = user_data.get("daily_count", 0) + 1
                    save_users(users)
                break

    # Log pour le monitoring
    try:
        monitor_service.log_prediction(
            username   = username,
            text       = review.text,
            prediction = result["sentiment"],
            confidence = result["confidence"] / 100.0,
            class_id   = result["class_id"],
        )
    except Exception as e:
        logger.warning("Échec du log de prédiction pour '%s' : %s", username, e)

    return {
        "texte":            review.text,
        "sentiment":        result["sentiment"],
        "prediction_score": f"{result['confidence']}%",
        "class_id":         result["class_id"],
    }


@router.post("/predict/batch")
def batch_predict(reviews: list[Review], api_key: str = Depends(get_api_key)):
    """Prédit le sentiment pour une liste de textes et consomme des crédits quota.
    
    Suit le même modèle que /predict/live mais traite plusieurs textes à la fois.
    """
    try:
        # Valider que des reviews sont fournies
        if not reviews:
            raise HTTPException(status_code=400, detail="La liste de reviews est vide.")
        
        # Vérifier la limite (ex: 100 textes max par batch)
        if len(reviews) > 100:
            raise HTTPException(status_code=400, detail="Limite de 100 textes par batch dépassée.")
        
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Modèle non chargé.")
    
    # Mise à jour atomique du quota pour l'ensemble du batch
    username = "unknown"
    with _quota_lock:
        users = get_users()
        for uname, user_data in users.items():
            if user_data.get("api_key") == api_key:
                username = uname
                if user_data.get("role") != "admin":
                    today = date.today().isoformat()
                    if user_data.get("last_request_date") != today:
                        user_data["daily_count"]       = 0
                        user_data["last_request_date"] = today
                    # Consommer un crédit pour le batch entier (ou len(reviews) crédits si souhaité)
                    user_data["daily_count"] = user_data.get("daily_count", 0) + len(reviews)
                    save_users(users)
                break
    
    # Traiter chaque review et collecter les résultats
    results = []
    for review in reviews:
        try:
            result = ml_service.predict(review.text)
        except RuntimeError:
            logger.warning("Erreur de prédiction pour le texte : %s", review.text)
            results.append({
                "texte":            review.text,
                "sentiment":        "Erreur",
                "prediction_score": "N/A",
                "class_id":         -1,
                "error":            "Modèle non chargé"
            })
            continue
        
        # Log pour le monitoring
        try:
            monitor_service.log_prediction(
                username   = username,
                text       = review.text,
                prediction = result["sentiment"],
                confidence = result["confidence"] / 100.0,
                class_id   = result["class_id"],
            )
        except Exception as e:
            logger.warning("Échec du log de prédiction pour '%s' : %s", username, e)
        
        # Ajouter le résultat à la liste
        results.append({
            "texte":            review.text,
            "sentiment":        result["sentiment"],
            "prediction_score": f"{result['confidence']}%",
            "class_id":         result["class_id"],
        })
    
    return {
        "count":       len(results),
        "predictions": results
    }


    
@router.get("/metrics")
def get_metrics(api_key: str = Depends(get_api_key)):
    try:
        with open("metrics/scores.json", "r") as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        logger.error(f"Erreur lors de la lecture des métriques : {e}")
        raise HTTPException(status_code=500, detail="Impossible de lire les métriques.")


@router.get("/feature-importance")
def feature_importance(api_key: str = Depends(get_api_key)):
    try:
        importances = pd.read_csv("metrics/feature_importance.csv")
        top_features = importances.head(20).to_dict(orient="records")
        return {"top_features": top_features}
    except Exception as e:
        logger.error(f"Erreur lors de la lecture de l'importance des features : {e}")
        raise HTTPException(status_code=500, detail="Impossible de lire l'importance des features.")



@router.post("/extract_text_csv", response_model=List[str])
def extract_text_csv(data: Base64, api_key: str = Depends(get_api_key)):
    """
    Décode un CSV en base64, convertit en DataFrame et retourne la colonne 'text' comme liste.
    """
    try:
        # 1️Décodage base64
        decoded = base64.b64decode(data.base64)

        # Conversion en DataFrame
        csv_buffer = io.StringIO(decoded.decode("utf-8"))
        df = pd.read_csv(csv_buffer)

        # Vérification colonne 'text'
        if "text" not in df.columns:
            raise HTTPException(status_code=400, detail="La colonne 'text' est manquante dans le CSV.")

        # Récupération de la colonne 'text'
        text_list = df["text"].astype(str).tolist()

        return text_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement du CSV : {e}")
    
