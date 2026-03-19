"""
Route pour entraîner le modèle de prédiction.
"""

import logging
import threading
from datetime import date
import os
import subprocess

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Path

from ..schemas.models import TrainRequest
from ..services import monitor_service
from ..services.users import get_users, save_users
from ..services.train import train
from ..core.security import get_api_key

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Training"])

# Verrou pour éviter les race conditions sur la mise à jour du quota
_quota_lock = threading.Lock()

@router.post("/train")
def trigger_training(
    train_request: TrainRequest,
    background_tasks: BackgroundTasks,
):
    try:    # Lancement de l'entraînement en arrière-plan
        logger.info("Lancement de l'entraînement du modèle en arrière-plan...")


        background_tasks.add_task(
            train,
            csv_path=train_request.data_path,
            dataset_name=train_request.dataset_name,
            max_features=train_request.max_features,
            ngram_max=train_request.ngram_max,
            learning_rate=train_request.learning_rate,
            num_leaves=train_request.num_leaves,
            n_estimators=train_request.n_estimators,
            colsample_bytree=train_request.colsample_bytree,
            subsample=train_request.subsample,
            early_stopping_rounds=train_request.early_stopping_rounds,
            test_size=train_request.test_size
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
