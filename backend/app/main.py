"""
Point d'entrée FastAPI — assemble toutes les routes.
"""

import boto3
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from .routes import auth, predict, monitoring, reload
from .services.ml_service import get_model
from .services.metrics_service import MODEL_LOADED_GAUGE, update_drift_metrics, ACTIVE_USERS_GAUGE
from .services.monitor_service import get_monitoring_stats
from .services.users import get_users

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Pré-charge le modèle au démarrage et initialise les métriques Prometheus."""
    
    # 1. Chargement du modèle ML
    model = get_model()
    MODEL_LOADED_GAUGE.set(1 if model[0] is not None else 0)
    
    # 2. Initialisation des métriques de drift et d'utilisateurs
    try:       
        stats = get_monitoring_stats()
        update_drift_metrics(stats)
        ACTIVE_USERS_GAUGE.set(len(get_users()))
        logging.getLogger(__name__).info("Métriques Prometheus initialisées avec succès.")
    except Exception as e:
        logging.getLogger(__name__).warning("Erreur initialisation métriques Prometheus: %s", e)
        
    yield


app = FastAPI(
    title="Trustpilot Sentiment API",
    description="API sécurisée · prédiction de sentiment · monitoring",
    version="4.2",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics/prometheus")

# Routes
app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(monitoring.router)
app.include_router(reload.router)     # ← rechargement à chaud du modèle (Airflow)


@app.get("/", tags=["Health"])
def root():
    return {"message": "Trustpilot Sentiment API v4.2 — /docs pour tester."}


@app.get("/health", tags=["Health"])
def health():
    """Endpoint de santé — vérifie que l'API et le modèle sont opérationnels."""
    from .services.ml_service import _model, _vectorizer
    model_loaded = _model is not None and _vectorizer is not None
    return {
        "status":       "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "version":      "4.2",
    }
