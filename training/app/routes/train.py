"""
Route pour entraîner le modèle de prédiction avec suivi asynchrone des jobs.
"""

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, BackgroundTasks

from ..schemas.models import TrainRequest
from ..services.train import train

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Training"])

# Dictionnaire en mémoire pour suivre l'état des entraînements
_jobs = {}


def _run_training_job(job_id: str, train_params: dict):
    """Exécute l'entraînement en tâche de fond et enregistre le statut / métriques."""
    start_time = time.time()
    _jobs[job_id]["status"] = "RUNNING"
    logger.info(f"Début de l'entraînement pour le Job ID: {job_id}")

    try:
        # train(...) retourne un dict : {"run_id": ..., "accuracy": ..., "f1_score": ...}
        result = train(**train_params)
        duration = round(time.time() - start_time, 2)

        _jobs[job_id].update({
            "status": "SUCCESS",
            "duration_seconds": duration,
            "result": result,
        })
        logger.info(f"Job {job_id} terminé avec succès en {duration}s")

    except Exception as e:
        duration = round(time.time() - start_time, 2)
        logger.error(f"Échec du Job {job_id} : {e}", exc_info=True)

        _jobs[job_id].update({
            "status": "FAILED",
            "duration_seconds": duration,
            "error": str(e),
        })


@router.post("/train")
@router.post("/train/train")
def trigger_training(
    train_request: TrainRequest,
    background_tasks: BackgroundTasks,
):
    """Lance un job d'entraînement en arrière-plan et renvoie son job_id."""
    try:
        job_id = str(uuid.uuid4())
        train_params = {
            "csv_path": train_request.data_path,
            "dataset_name": train_request.dataset_name,
            "max_features": train_request.max_features,
            "ngram_max": train_request.ngram_max,
            "learning_rate": train_request.learning_rate,
            "num_leaves": train_request.num_leaves,
            "n_estimators": train_request.n_estimators,
            "colsample_bytree": train_request.colsample_bytree,
            "subsample": train_request.subsample,
            "early_stopping_rounds": train_request.early_stopping_rounds,
            "test_size": train_request.test_size,
        }

        # Initialisation du job
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "PENDING",
            "duration_seconds": None,
            "result": None,
            "error": None,
        }

        # Lancement asynchrone
        background_tasks.add_task(_run_training_job, job_id, train_params)

        # Réponse immédiate attendue par la tâche trigger_training d'Airflow
        return {
            "job_id": job_id,
            "status": "PENDING",
            "message": "Training lancé en arrière-plan.",
        }
    except Exception as e:
        logger.error(f"Erreur lors du lancement de l'entraînement : {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du lancement de l'entraînement du modèle.")


@router.get("/status/{job_id}")
@router.get("/train/status/{job_id}")
def get_training_status(job_id: str):
    """Retourne l'état actuel du job sondé par le PythonSensor d'Airflow."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' introuvable.")
    return _jobs[job_id]


@router.get("/jobs")
@router.get("/train/jobs")
def list_jobs():
    """Liste tous les jobs d'entraînement récents."""
    return list(_jobs.values())


@router.post("/collect_data")
@router.post("/train/collect_data")
def collect_data():
    return {"message": "Route de collecte de données - à implémenter selon les besoins."}
