"""
Route pour entraîner le modèle de prédiction.
"""

import logging
import threading
from datetime import date

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks

from ..schemas.models import Review
from ..services import ml_service, monitor_service
from ..services.users import get_users, save_users
from training.train import train
from ..core.security import get_api_key

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Training"])

# Verrou pour éviter les race conditions sur la mise à jour du quota
_quota_lock = threading.Lock()

@router.post("/train")
def trigger_training(
    background_tasks: BackgroundTasks,
    data_path: str,
    max_features: int = 20000,
    n_estimators: int = 1000
):
    try:    # Lancement de l'entraînement en arrière-plan
        logger.info("Lancement de l'entraînement du modèle en arrière-plan...")
        background_tasks.add_task(
            train,
            csv_path=data_path,
            max_features=max_features,
            n_estimators=n_estimators
        )

        return {
            "message": "Training lancé en arrière-plan."
        }
    except Exception as e:
        logger.error(f"Erreur lors du lancement de l'entraînement : {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du lancement de l'entraînement du modèle.")


@router.post("/collect_data")
def collect_data():
    # Cette route pourrait être utilisée pour collecter de nouvelles données d'avis clients
    # et les stocker dans une base de données ou un fichier pour un futur entraînement.
    return {"message": "Route de collecte de données - à implémenter selon les besoins."}
