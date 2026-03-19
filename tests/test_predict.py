"""
Tests de la route /predict et des routes de monitoring.
"""

import pytest
from .conftest import register_and_login



class TestPredict:
    def test_predict_success(self, client):
        token = register_and_login(client)
        resp = client.post(
            "/predict",
            json={"text": "Ce produit est excellent !"},
            headers={"X-API-Key": token},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "sentiment" in data
        assert "prediction_score" in data
        assert data["sentiment"] in ("Négatif", "Neutre", "Positif")
        assert "class_id" in data

    def test_predict_batch_success(self, client):
        token = register_and_login(client)
        payload = [
            {"text": "Ce produit est excellent !"},
            {"text": "Livraison lente et mauvaise qualité"},
            {"text": "Tout est parfait"}
        ]
        resp = client.post(
            "/predict/batch",
            json=payload,
            headers={"X-API-Key": token},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert data["count"] == 3
        assert "predictions" in data
        assert len(data["predictions"]) == 3
        
        # Vérification de chaque prédiction
        for pred in data["predictions"]:
            assert "texte" in pred
            assert "sentiment" in pred
            assert "prediction_score" in pred
            assert "class_id" in pred
            assert pred["sentiment"] in ("Négatif", "Neutre", "Positif")
            assert pred["class_id"] in (0, 1, 2)

    def test_predict_no_token(self, client):
        resp = client.post("/predict", json={"text": "Test"})
        assert resp.status_code == 401

    def test_predict_invalid_token(self, client):
        resp = client.post(
            "/predict",
            json={"text": "Test"},
            headers={"X-API-Key": "badtoken"},
        )
        assert resp.status_code == 403

    def test_predict_empty_text(self, client):
        token = register_and_login(client)
        resp = client.post(
            "/predict",
            json={"text": ""},
            headers={"X-API-Key": token},
        )
        assert resp.status_code == 422

    def test_predict_blank_text(self, client):
        token = register_and_login(client)
        resp = client.post(
            "/predict",
            json={"text": "   "},
            headers={"X-API-Key": token},
        )
        assert resp.status_code == 422

    def test_predict_increments_quota(self, client):
        token = register_and_login(client)
        # Quota initial à 0
        quota_before = client.get("/quota/status", headers={"X-API-Key": token}).json()
        assert quota_before["quota_used"] == 0

        client.post("/predict", json={"text": "Test"}, headers={"X-API-Key": token})

        quota_after = client.get("/quota/status", headers={"X-API-Key": token}).json()
        assert quota_after["quota_used"] == 1

        # ----------------------------
    # 🔹 Test unitaire pour batch_predict
    # ----------------------------
    @patch('backend.app.routes.predict.monitor_service')
    @patch('backend.app.routes.predict.ml_service')
    @patch('backend.app.routes.predict.get_users')
    @patch('backend.app.routes.predict.save_users')
    def test_batch_predict_unit(mock_save_users, mock_get_users, mock_ml_service, mock_monitor_service):
        """Test unitaire de la fonction batch_predict avec mocks."""
        # Mock des utilisateurs
        mock_get_users.return_value = {
            "test_user": {
                "api_key": API_KEY,
                "role": "user",
                "daily_count": 0,
                "last_request_date": "2026-03-17"
            }
        }
        
        # Mock de ml_service.predict
        mock_ml_service.predict.side_effect = [
            {"sentiment": "Positif", "confidence": 85.0, "class_id": 2},
            {"sentiment": "Négatif", "confidence": 92.0, "class_id": 0}
        ]
        
        # Mock de monitor_service.log_prediction
        mock_monitor_service.log_prediction.return_value = None
        
        # Import de la fonction à tester
        from backend.app.routes.predict import batch_predict
        
        # Données de test
        reviews = [
            Review(text="Super produit, très satisfait"),
            Review(text="Livraison lente et mauvaise qualité")
        ]
        
        # Appel de la fonction
        result = batch_predict(reviews, api_key=API_KEY)
        
        # Assertions
        assert "count" in result
        assert result["count"] == 2
        assert "predictions" in result
        assert len(result["predictions"]) == 2
        
        # Vérification des prédictions
        pred1 = result["predictions"][0]
        assert pred1["texte"] == "Super produit, très satisfait"
        assert pred1["sentiment"] == "Positif"
        assert pred1["prediction_score"] == "85.0%"
        assert pred1["class_id"] == 2
        
        pred2 = result["predictions"][1]
        assert pred2["texte"] == "Livraison lente et mauvaise qualité"
        assert pred2["sentiment"] == "Négatif"
        assert pred2["prediction_score"] == "92.0%"
        assert pred2["class_id"] == 0
        
        # Vérification que les mocks ont été appelés
        assert mock_ml_service.predict.call_count == 2
        mock_ml_service.predict.assert_any_call("Super produit, très satisfait")
        mock_ml_service.predict.assert_any_call("Livraison lente et mauvaise qualité")
        
        assert mock_monitor_service.log_prediction.call_count == 2
        mock_save_users.assert_called_once()


class TestQuota:
    def test_quota_status_user(self, client):
        token = register_and_login(client)
        resp = client.get("/quota/status", headers={"X-API-Key": token})
        assert resp.status_code == 200
        data = resp.json()
        assert data["quota_max"] == 5
        assert data["role"] == "user"

    def test_quota_status_admin(self, client):
        token = register_and_login(client, username="admin", password="adminpass", role="admin")
        resp = client.get("/quota/status", headers={"X-API-Key": token})
        assert resp.status_code == 200
        data = resp.json()
        assert data["quota_max"] == "unlimited"
        assert data["role"] == "admin"

    def test_quota_missing_token(self, client):
        resp = client.get("/quota/status")
        assert resp.status_code == 401


class TestHistory:
    def test_history_empty(self, client):
        token = register_and_login(client)
        resp = client.get("/predictions/history", headers={"X-API-Key": token})
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_history_after_predict(self, client):
        token = register_and_login(client)
        client.post("/predict", json={"text": "Super produit !"}, headers={"X-API-Key": token})
        resp = client.get("/predictions/history", headers={"X-API-Key": token})
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    def test_history_invalid_token(self, client):
        resp = client.get("/predictions/history", headers={"X-API-Key": "bad"})
        assert resp.status_code == 401


class TestFeedback:
    def test_feedback_not_found(self, client):
        token = register_and_login(client)
        resp = client.post(
            "/feedback",
            json={"timestamp": "2026-01-01T00:00:00", "feedback": "correct"},
            headers={"X-API-Key": token},
        )
        assert resp.status_code == 404

    def test_feedback_invalid_value(self, client):
        token = register_and_login(client)
        resp = client.post(
            "/feedback",
            json={"timestamp": "2026-01-01T00:00:00", "feedback": "maybe"},
            headers={"X-API-Key": token},
        )
        assert resp.status_code == 422


class TestHealth:
    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
