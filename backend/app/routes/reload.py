"""
Route /reload — Rechargement à chaud du modèle LightGBM + TF-IDF.

Appelé par Airflow après un réentraînement réussi pour éviter
de redémarrer le conteneur. Accès réservé aux admins.
"""

import logging
import os

from fastapi import APIRouter, Depends, Header
from fastapi.responses import JSONResponse

from ..core.security import require_admin
from ..services import ml_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Model Management"])


@router.post("/reload")
def reload_model(_admin: str = Depends(require_admin)) -> JSONResponse:
    """
    Force le rechargement des fichiers .pkl depuis le disque.

    - Réinitialise le singleton _model / _vectorizer dans ml_service
    - Recharge immédiatement depuis MODEL_PATH et VECTORIZER_PATH
    - Retourne les métadonnées du modèle chargé (taille fichiers, timestamp)

    Accès : admin uniquement (via require_admin)
    """
    logger.info("Rechargement du modèle demandé par l'admin '%s'.", _admin)

    # Réinitialise le singleton pour forcer le rechargement
    ml_service._model      = None
    ml_service._vectorizer = None

    model, vectorizer = ml_service.get_model()

    if model is None or vectorizer is None:
        logger.error("Échec du rechargement : get_model() a retourné None.")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Échec du rechargement — fichiers .pkl introuvables ou corrompus.",
            },
        )

    # Infos sur les fichiers rechargés (pour traçabilité dans les logs Airflow)
    from ..core.config import MODEL_PATH, VECTORIZER_PATH

    def _file_info(path: str) -> dict:
        try:
            stat = os.stat(path)
            return {
                "path":             path,
                "size_kb":          round(stat.st_size / 1024, 1),
                "last_modified":    stat.st_mtime,
            }
        except OSError:
            return {"path": path, "error": "fichier introuvable"}

    model_info      = _file_info(MODEL_PATH)
    vectorizer_info = _file_info(VECTORIZER_PATH)

    logger.info(
        "✅ Modèle rechargé avec succès. model=%s vectorizer=%s",
        model_info, vectorizer_info,
    )

    return JSONResponse(
        status_code=200,
        content={
            "success":   True,
            "message":   "Modèle rechargé avec succès.",
            "model":     model_info,
            "vectorizer": vectorizer_info,
        },
    )


@router.post("/internal/reload-model")
def reload_model_internal(
    x_internal_secret: str = Header(None, alias="X-Internal-Secret")
) -> JSONResponse:
    """
    Endpoint interne appelé par Airflow pour déclencher le rechargement du modèle.
    Valide le secret X-Internal-Secret.
    """
    expected_secret = os.environ.get("INTERNAL_SECRET", "airflow-internal-secret")
    if not x_internal_secret or x_internal_secret != expected_secret:
        logger.error("Accès refusé: Secret interne invalide.")
        return JSONResponse(
            status_code=403,
            content={"success": False, "message": "Accès refusé. Secret invalide."},
        )

    logger.info("Rechargement du modèle demandé via endpoint interne (Airflow).")

    try:
        model, vectorizer = ml_service.reload_model()
    except Exception as e:
        logger.error("Erreur lors du rechargement interne : %s", e)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Échec du rechargement interne: {str(e)}",
            },
        )

    from ..core.config import MODEL_PATH, VECTORIZER_PATH

    def _file_info(path: str) -> dict:
        try:
            stat = os.stat(path)
            return {
                "path":             path,
                "size_kb":          round(stat.st_size / 1024, 1),
                "last_modified":    stat.st_mtime,
            }
        except OSError:
            return {"path": path, "error": "fichier introuvable"}

    model_info      = _file_info(MODEL_PATH)
    vectorizer_info = _file_info(VECTORIZER_PATH)

    return JSONResponse(
        status_code=200,
        content={
            "success":   True,
            "message":   "Modèle rechargé avec succès depuis DagsHub S3 via DVC.",
            "model":     model_info,
            "vectorizer": vectorizer_info,
        },
    )
