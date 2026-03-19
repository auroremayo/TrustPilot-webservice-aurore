"""
Tests des routes d'authentification : /login, /token_API, /verify_admin.
"""

import pytest
from .conftest import register_and_login


class TestRegister:
    def test_register_success(self, client):
        resp = client.post("/login", json={"username": "alice", "password": "secret123"})
        assert resp.status_code == 200
        assert "créé" in resp.json()["message"]

    def test_register_duplicate(self, client):
        client.post("/login", json={"username": "alice", "password": "secret123"})
        resp = client.post("/login", json={"username": "alice", "password": "autremdp"})
        assert resp.status_code == 400

    def test_register_username_too_short(self, client):
        resp = client.post("/login", json={"username": "ab", "password": "secret123"})
        assert resp.status_code == 422

    def test_register_password_too_short(self, client):
        resp = client.post("/login", json={"username": "alice", "password": "ab"})
        assert resp.status_code == 422

    def test_register_invalid_role(self, client):
        resp = client.post("/login", json={"username": "alice", "password": "secret123", "role": "superadmin"})
        assert resp.status_code == 422

    def test_register_username_special_chars(self, client):
        resp = client.post("/login", json={"username": "ali ce!", "password": "secret123"})
        assert resp.status_code == 422


class TestLogin:
    def test_login_success(self, client):
        client.post("/login", json={"username": "alice", "password": "secret123"})
        resp = client.post("/token_API", json={"username": "alice", "password": "secret123"})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["username"] == "alice"
        assert data["role"] == "user"
        assert "token_expires_at" in data

    def test_login_wrong_password(self, client):
        client.post("/login", json={"username": "alice", "password": "secret123"})
        resp = client.post("/token_API", json={"username": "alice", "password": "mauvais"})
        assert resp.status_code == 401

    def test_login_unknown_user(self, client):
        resp = client.post("/token_API", json={"username": "fantome", "password": "x"})
        assert resp.status_code == 401

    def test_login_admin_role(self, client):
        client.post("/login", json={"username": "admin", "password": "adminpass", "role": "admin"})
        resp = client.post("/token_API", json={"username": "admin", "password": "adminpass"})
        assert resp.status_code == 200
        assert resp.json()["role"] == "admin"


class TestVerifyAdmin:
    def test_verify_valid_token(self, client):
        token = register_and_login(client)
        resp = client.get("/verify_admin", headers={"X-API-Key": token})
        assert resp.status_code == 200

    def test_verify_missing_token(self, client):
        resp = client.get("/verify_admin")
        assert resp.status_code == 401

    def test_verify_invalid_token(self, client):
        resp = client.get("/verify_admin", headers={"X-API-Key": "invalidtoken"})
        assert resp.status_code == 401
