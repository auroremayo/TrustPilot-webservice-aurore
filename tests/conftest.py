"""
Fixtures partagées entre tous les tests.
Les modules backend sont importés au niveau module pour s'assurer qu'ils sont
dans sys.modules, puis leurs attributs sont patchés directement dans chaque fixture.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Pré-import pour que tous les sous-modules soient dans sys.modules
from backend.app.main import app
import backend.app.services.users as _users_mod
import backend.app.services.monitor_service as _monitor_mod
import backend.app.services.ml_service as _ml_mod


@pytest.fixture()
def tmp_users_file(tmp_path):
    """Crée un users.json vide temporaire isolé pour chaque test."""
    f = tmp_path / "users.json"
    f.write_text("{}", encoding="utf-8")
    return str(f)


@pytest.fixture()
def tmp_log_file(tmp_path):
    """Crée un predictions_log.jsonl vide temporaire isolé pour chaque test."""
    f = tmp_path / "predictions_log.jsonl"
    f.write_text("", encoding="utf-8")
    return str(f)


@pytest.fixture()
def client(tmp_users_file, tmp_log_file):
    """
    TestClient FastAPI avec :
    - USERS_FILE et PREDICTIONS_LOG redirigés vers des fichiers tmp isolés
    - Modèle ML mocké (retourne Positif à 85% de confiance)
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = [2]
    mock_model.predict_proba.return_value = [[0.05, 0.10, 0.85]]

    mock_vectorizer = MagicMock()
    mock_vectorizer.transform.return_value = MagicMock()

    with (
        patch.object(_users_mod, "USERS_FILE", tmp_users_file),
        patch.object(_monitor_mod, "PREDICTIONS_LOG", tmp_log_file),
        patch.object(_ml_mod, "_model", mock_model),
        patch.object(_ml_mod, "_vectorizer", mock_vectorizer),
    ):
        with TestClient(app) as c:
            yield c


def register_and_login(client, username="testuser", password="password123", role="user"):
    """Helper : crée un compte et retourne le token API."""
    client.post("/login", json={"username": username, "password": password, "role": role})
    resp = client.post("/token_API", json={"username": username, "password": password})
    return resp.json()["access_token"]
