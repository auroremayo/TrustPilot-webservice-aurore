"""
Routes de monitoring, historique et quota.
"""

from datetime import date
from fastapi import APIRouter, HTTPException, Header, Depends

from ..schemas.models import FeedbackPayload
from ..services import monitor_service
from ..services.users import get_users
from ..core.security import get_username_from_key, require_admin
from ..core.config import DAILY_QUOTA

router = APIRouter(tags=["Monitoring"])


@router.get("/quota/status")
def quota_status(x_api_key: str = Header(None, alias="X-API-Key")):
    """Retourne le quota journalier de l'utilisateur connecté."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Clé API manquante.")

    users = get_users()
    for username, user_data in users.items():
        if user_data.get("api_key") == x_api_key:
            role = user_data.get("role", "user")
            if role == "admin":
                return {
                    "username": username, "role": "admin",
                    "quota_used": 0, "quota_max": "unlimited", "remaining": "unlimited",
                }
            today = date.today().isoformat()
            used  = (user_data.get("daily_count", 0)
                     if user_data.get("last_request_date") == today else 0)
            return {
                "username":   username,
                "role":       "user",
                "quota_used": used,
                "quota_max":  DAILY_QUOTA,
                "remaining":  max(0, DAILY_QUOTA - used),
            }
    raise HTTPException(status_code=401, detail="Clé API invalide.")


@router.get("/predictions/history")
def prediction_history(x_api_key: str = Header(None, alias="X-API-Key")):
    """Retourne l'historique des prédictions de l'utilisateur connecté."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Clé API manquante.")
    username = get_username_from_key(x_api_key)
    if not username:
        raise HTTPException(status_code=401, detail="Clé API invalide.")
    history = monitor_service.get_user_history(username)
    return {"username": username, "history": history, "count": len(history)}


@router.post("/feedback")
def submit_feedback(
    payload: FeedbackPayload,
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    """Enregistre un retour utilisateur (correct/incorrect) sur une prédiction."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Clé API manquante.")
    if get_username_from_key(x_api_key) is None:
        raise HTTPException(status_code=401, detail="Clé API invalide.")
    if payload.feedback not in ("correct", "incorrect"):
        raise HTTPException(
            status_code=400,
            detail="feedback doit être 'correct' ou 'incorrect'.",
        )
    if not monitor_service.update_feedback(payload.timestamp, payload.feedback):
        raise HTTPException(status_code=404, detail="Entrée introuvable.")
    return {"message": "Feedback enregistré."}


@router.get("/monitor/stats")
def monitoring_stats(
    days_recent: int = 7,
    days_all: int = 30,
    _admin: str = Depends(require_admin),
):
    """Statistiques de drift et de monitoring (admin uniquement)."""
    return monitor_service.get_monitoring_stats(
        days_recent=days_recent, days_all=days_all
    )

