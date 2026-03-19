from fastapi.responses import JSONResponse
from ..services.ml_service import get_model

def liveness():
    return JSONResponse(
        status_code=200,
        content={"status": "live"}
    )


def readiness():
    try:
        model, vectorizer = get_model()
        if model is None or vectorizer is None:
            return JSONResponse(
                status_code=503,
                content={"status": "not ready", "reason": "Le modèle ou le vectorizer n'est pas prêt."}
            )
        
        # Test de vectorisation
        test_text = "Ceci est un test de sentiment positif"
        vec = vectorizer.transform([test_text])
        
        # Vérifier que c'est un vecteur (array ou matrice sparse)
        if not hasattr(vec, 'shape') or vec.shape[0] != 1:
            return JSONResponse(
                status_code=503,
                content={"status": "not ready", "reason": "Le vectorizer ne retourne pas un vecteur valide."}
            )
        
        # Test de prédiction
        prediction = model.predict(vec)[0]
        
        # Vérifier que c'est un entier entre 0 et 2
        if not isinstance(prediction, (int, float)) or not (0 <= prediction <= 2):
            return JSONResponse(
                status_code=503,
                content={"status": "not ready", "reason": "Le modèle ne retourne pas une prédiction valide (entier entre 0 et 2)."}
            )
        
        return JSONResponse(
            status_code=200,
            content={"status": "ready"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "reason": f"Erreur lors du test : {str(e)}"}
        )
