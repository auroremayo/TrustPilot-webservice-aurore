"""
Point d'entrée FastAPI — assemble toutes les routes.
"""


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from .routes import auth, monitoring, train, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

import os
from pathlib import Path
import requests
import logging
from contextlib import asynccontextmanager

# remonter à la racine du projet
ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)
print("New CWD:", os.getcwd())


# ── CONFIGURATION DES WEBHOOKS MLFLOW ───────────────────────────────────────
default_mlflow_url = "http://mlflow:5000" if (os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER")) else "http://localhost:5000"
MLFLOW_INTERNAL_URL = os.environ.get("MLFLOW_TRACKING_URI", default_mlflow_url)
WEBHOOK_WORKER_URL = "http://mlflow-webhook-worker:8000/trigger"

def setup_mlflow_webhooks():
    """Configure automatiquement les webhooks dans le serveur MLflow"""
    models_to_register = ["trustpilot_lgbm_model", "vectorizer"]
    endpoint = f"{MLFLOW_INTERNAL_URL}/api/2.0/mlflow/registry-webhooks/create"
    
    logging.info("🔧 Vérification et configuration des Webhooks MLflow...")
    
    for model_name in models_to_register:
        payload = {
            "model_name": model_name,
            "events": ["MODEL_VERSION_TRANSITIONED_STAGE"],
            "url": WEBHOOK_WORKER_URL,
            "description": f"Alerte le worker pour copier {model_name} vers Dagshub S3"
        }
        
        try:
            response = requests.post(endpoint, json=payload, timeout=5)
            
            if response.status_code == 200:
                logging.info(f"✅ Webhook configuré avec succès pour le modèle : {model_name}")
            elif response.status_code == 400 and "already exists" in response.text.lower():
                logging.info(f"ℹ️ Le Webhook pour {model_name} est déjà en place.")
            else:
                logging.warning(f"⚠️ Réponse inattendue de MLflow pour {model_name} ({response.status_code}) : {response.text}")
                
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Impossible de joindre le serveur MLflow pour configurer le webhook de {model_name} : {e}")


# ── LIFECYCLE ASYNC DE L'APPLICATION ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ce code s'exécute au démarrage (Startup)
    setup_mlflow_webhooks()
    yield
    # Ce code s'exécuterait à l'arrêt (Shutdown) si nécessaire


app = FastAPI(
    title="Trustpilot Sentiment API",
    description="API sécurisée · prédiction de sentiment · monitoring",
    version="4.1",
    root_path="/train"
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
app.include_router(monitoring.router)
app.include_router(train.router)
app.include_router(health.router)

