
import logging
import threading
from datetime import date

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

# from ..core.health import liveness, readiness
from ..core.security import get_api_key

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])

# Verrou pour éviter les race conditions sur la mise à jour du quota
_quota_lock = threading.Lock()


@router.get("/health/live")
def liveness_route(api_key: str = Depends(get_api_key)):
    try:
        logger.info("Vérification de la disponibilité de l'API (liveness)...")
        # liveness_response = liveness()
        liveness_response = {"status": "alive"}
        return liveness_response    
    except Exception as e:
        logger.error("Erreur lors de la vérification de la disponibilité de l'API : %s", e)

@router.get("/health/ready")
def readiness_route(api_key: str = Depends(get_api_key)):
    try:
        logger.info("Vérification de la disponibilité des modèles de l'API (readiness)...")
        # readiness_response = readiness()
        readiness_response = {"status": "ready"}
        return readiness_response
    except Exception as e:
        logger.error("Erreur lors de la vérification de la disponibilité des modèles : %s", e)