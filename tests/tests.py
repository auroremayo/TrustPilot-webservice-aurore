# tests.py
import base64
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from backend.app.main import app  # Import correct de l'application FastAPI
from backend.app.schemas.models import Review

client = TestClient(app)

API_KEY = "7e8d36abc0288f2e112bec13d23a872c884c67e479eb31eb2e10781d41bee3c8"  # Mets ici la même que dans ton .env pour les tests

# ----------------------------
# 🔹 Test endpoints health
# ----------------------------
def test_health_live():
    response = client.get("/health/live", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    assert response.json() == {"status": "live"}


def test_health_ready():
    response = client.get("/health/ready", headers={"X-API-Key": API_KEY})
    # On teste juste la structure
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        assert response.json() == {"status": "ready"}
    else:
        data = response.json()
        assert "status" in data
        assert data["status"] == "not ready"
        assert "reason" in data

# ----------------------------
# 🔹 Test extraction texte CSV base64
# ----------------------------
def test_extract_text_csv():
    csv_content = "text,score\nSuper produit,2\nLivraison lente,0"
    csv_base64 = base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")
    payload = {"csv_base64": csv_base64}
    headers = {"X-API-Key": API_KEY}

    response = client.post("/extract_text_csv", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert data == ["Super produit", "Livraison lente"]