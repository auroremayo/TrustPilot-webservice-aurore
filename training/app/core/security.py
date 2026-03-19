"""
Sécurité : hachage des mots de passe (bcrypt), validation des clés API,
expiration des tokens, contrôle d'accès admin.
"""

import hashlib
import secrets
from datetime import date, timedelta

import bcrypt as _bcrypt
from fastapi import HTTPException, Header, Security
from fastapi.security import APIKeyHeader

from ..services.users import get_users
from ..core.config import TOKEN_EXPIRY_DAYS

api_key_header = APIKeyHeader(name="X-API-Key")


def hash_password(password: str) -> str:
    """Hache un mot de passe avec bcrypt."""
    return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()


def verify_password(plain: str, stored: str) -> bool:
    """
    Vérifie un mot de passe.
    Supporte l'ancien format SHA-256 (64 hex chars) pour la migration automatique.
    """
    if len(stored) == 64 and all(c in "0123456789abcdef" for c in stored):
        # Ancien hash SHA-256 — comparaison legacy
        return hashlib.sha256(plain.encode()).hexdigest() == stored
    # Nouveau format bcrypt
    return _bcrypt.checkpw(plain.encode(), stored.encode())


def generate_token() -> str:
    """Génère un token API aléatoire de 64 caractères hexadécimaux."""
    return secrets.token_hex(32)


def token_expiry_date() -> str:
    """Retourne la date d'expiration du token (aujourd'hui + TOKEN_EXPIRY_DAYS)."""
    return (date.today() + timedelta(days=TOKEN_EXPIRY_DAYS)).isoformat()


def _is_token_expired(user_data: dict) -> bool:
    """Retourne True si le token de l'utilisateur est expiré."""
    expires = user_data.get("token_expires_at")
    return bool(expires and date.today().isoformat() > expires)


def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """Dépendance FastAPI — valide que la clé appartient à un utilisateur non expiré."""
    users = get_users()
    for _, user_data in users.items():
        if user_data.get("api_key") == api_key:
            if _is_token_expired(user_data):
                raise HTTPException(status_code=401, detail="Token expiré. Reconnectez-vous.")
            return api_key
    raise HTTPException(status_code=403, detail="Accès refusé. Clé API invalide.")


def get_username_from_key(api_key: str) -> str | None:
    """Retourne le nom d'utilisateur associé à la clé, ou None si invalide/expiré."""
    users = get_users()
    for username, user_data in users.items():
        if user_data.get("api_key") == api_key:
            if _is_token_expired(user_data):
                return None
            return username
    return None


def require_admin(x_api_key: str = Header(None, alias="X-API-Key")) -> str:
    """Dépendance FastAPI — lève 403 si l'utilisateur n'est pas admin."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Clé API manquante.")
    users = get_users()
    for username, user_data in users.items():
        if user_data.get("api_key") == x_api_key:
            if _is_token_expired(user_data):
                raise HTTPException(status_code=401, detail="Token expiré. Reconnectez-vous.")
            if user_data.get("role") != "admin":
                raise HTTPException(status_code=403, detail="Accès admin requis.")
            return username
    raise HTTPException(status_code=401, detail="Clé API invalide.")
