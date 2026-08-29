"""
Point d'entrée FastAPI — assemble toutes les routes.
"""


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from .routes import train, health

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
app.include_router(train.router)
app.include_router(health.router)

