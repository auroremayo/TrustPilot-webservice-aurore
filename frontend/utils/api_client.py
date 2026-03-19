"""
Client API — toutes les fonctions d'appel vers le backend.
"""

import requests
from .constants import API_URL


def _headers(token: str) -> dict:
    return {"X-API-Key": token}


def register(username: str, password: str) -> requests.Response:
    return requests.post(
        f"{API_URL}/login",
        json={"username": username, "password": password},
        timeout=6,
    )


def login(username: str, password: str) -> requests.Response:
    return requests.post(
        f"{API_URL}/token_API",
        json={"username": username, "password": password},
        timeout=6,
    )


def predict(token: str, text: str) -> requests.Response:
    return requests.post(
        f"{API_URL}/predict",
        headers=_headers(token),
        json={"text": text},
        timeout=10,
    )


def get_quota(token: str) -> dict | None:
    try:
        r = requests.get(
            f"{API_URL}/quota/status", headers=_headers(token), timeout=4
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def get_history(token: str) -> dict | None:
    try:
        r = requests.get(
            f"{API_URL}/predictions/history", headers=_headers(token), timeout=6
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def send_feedback(token: str, timestamp: str, feedback: str) -> bool:
    try:
        r = requests.post(
            f"{API_URL}/feedback",
            headers=_headers(token),
            json={"timestamp": timestamp, "feedback": feedback},
            timeout=4,
        )
        return r.status_code == 200
    except Exception:
        return False


def get_monitor_stats(
    token: str, days_recent: int = 7, days_all: int = 30
) -> dict | None:
    try:
        r = requests.get(
            f"{API_URL}/monitor/stats",
            headers=_headers(token),
            params={"days_recent": days_recent, "days_all": days_all},
            timeout=8,
        )
        return r.json() if r.status_code == 200 else {"_error": r.status_code}
    except Exception as e:
        return {"_error": str(e)}
