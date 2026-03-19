"""
Point d'entrée FastAPI — assemble toutes les routes.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import auth, predict, monitoring
from .services.ml_service import get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Pré-charge le modèle au démarrage et libère les ressources à l'arrêt."""
    get_model()
    yield


app = FastAPI(
    title="Trustpilot Sentiment API",
    description="API sécurisée · prédiction de sentiment · monitoring",
    version="4.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(monitoring.router)


@app.get("/", tags=["Health"])
def root():
    return {"message": "Trustpilot Sentiment API v4.1 — /docs pour tester."}


@app.get("/health", tags=["Health"])
def health():
    """Endpoint de santé — vérifie que l'API et le modèle sont opérationnels."""
    from .services.ml_service import _model, _vectorizer
    model_loaded = _model is not None and _vectorizer is not None
    return {
        "status":       "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "version":      "4.1",
    }
