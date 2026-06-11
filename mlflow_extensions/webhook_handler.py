# mlflow_extensions/webhook_handler.py
import os
import mlflow
from urllib.parse import urlparse
from fastapi import FastAPI, Request, BackgroundTasks

try:
    from minio import Minio
except Exception:
    Minio = None

app = FastAPI()


def upload_to_s3(model_name, version, run_id):
    print(f"📦 Nouveau composant en Production détecté : {model_name} (v{version})")

    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # 1. Téléchargement local temporaire du dossier d'artefacts complet
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")

    # Trouve le fichier .pkl
    pkl_files = [f for f in os.listdir(local_path) if f.endswith('.pkl')]
    if not pkl_files:
        print(f"❌ Aucun fichier .pkl trouvé dans les artefacts pour {model_name}")
        return
    detected_model_file = pkl_files[0]
    model_file_path = os.path.join(local_path, detected_model_file)

    # 2. Upload via MinIO client (compatible S3)
    if Minio is None:
        print("⚠️ Minio client non installé — impossible d'uploader vers S3")
        return

    endpoint = os.environ.get('S3_ENDPOINT_URL')
    if not endpoint:
        print("⚠️ S3_ENDPOINT_URL non configuré — annulation de l'upload")
        return

    parsed = urlparse(endpoint)
    host = parsed.netloc
    secure = parsed.scheme == 'https'

    client = Minio(
        host,
        access_key=os.environ.get('AWS_ACCESS_KEY'),
        secret_key=os.environ.get('AWS_SECRET_KEY'),
        secure=secure,
    )

    s3_bucket = os.environ.get('S3_BUCKET_NAME')
    s3_key = f"prod_models/{model_name}.pkl"

    try:
        client.fput_object(s3_bucket, s3_key, model_file_path)
        print(f"🚀 {model_name} poussé avec succès sur DagsHub S3 -> {s3_key} !")
    except Exception as e:
        print(f"❌ Échec de l'upload vers S3 pour {model_name}: {e}")


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