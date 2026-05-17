# mlflow_extensions/webhook_handler.py
import os
import boto3
import mlflow
from fastapi import FastAPI, Request, BackgroundTasks

app = FastAPI()

def upload_to_s3(model_name, version, run_id):
    print(f"📦 Nouveau composant en Production détecté : {model_name} (v{version})")
    
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    # Configuration MLflow pour pointer sur le conteneur côte à côte
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # 1. Téléchargement local temporaire du dossier d'artefacts complet
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
    
    # Astuce pour trouver le fichier .pkl peu importe son nom (model.pkl, vectorizer.pkl, etc.)
    pkl_files = [f for f in os.listdir(local_path) if f.endswith('.pkl')]
    
    if not pkl_files:
        print(f"❌ Aucun fichier .pkl trouvé dans les artefacts pour {model_name}")
        return
        
    # On prend le premier fichier pickle trouvé dans l'artefact
    detected_model_file = pkl_files[0]
    model_file_path = os.path.join(local_path, detected_model_file)
    
    # 2. Connexion vers le S3 de DagsHub
    s3_client = boto3.client(
        's3',
        endpoint_url=os.environ.get('S3_ENDPOINT_URL'),
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_KEY')
    )
    
    # 3. Envoi vers S3
    # Le fichier sera nommé sur S3 selon son nom enregistré dans le Model Registry 
    # Exemple : prod_models/vectorizer.pkl et prod_models/trustpilot_lgbm_model.pkl
    s3_bucket = os.environ.get('S3_BUCKET_NAME')
    s3_key = f"prod_models/{model_name}.pkl"
    
    s3_client.upload_file(model_file_path, s3_bucket, s3_key)
    print(f"🚀 {model_name} poussé avec succès sur DagsHub S3 -> {s3_key} !")

@app.post("/trigger")
async def trigger(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    
    # On intercepte uniquement le passage en "Production"
    if payload.get("model_version", {}).get("current_stage") == "Production":
        model_name = payload["model_version"]["name"]
        version = payload["model_version"]["version"]
        run_id = payload["model_version"]["run_id"]
        
        # Lance le téléchargement et l'upload S3 en arrière-plan
        background_tasks.add_task(upload_to_s3, model_name, version, run_id)
        return {"status": "transfer_started", "model": model_name}
        
    return {"status": "ignored"}