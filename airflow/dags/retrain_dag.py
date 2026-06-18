"""
DAG Airflow — Pipeline de réentraînement du modèle SentimentAI.

Étapes :
  1. check_drift       : vérifie la KL divergence via /monitor/stats
                         court-circuite si drift < seuil ou pas de données
  2. trigger_training  : déclenche l'entraînement via /train/train (Nginx)
  3. wait_for_training : attend la fin de l'entraînement + dvc push vers S3
  4. reload_model      : backend recharge le modèle depuis DagsHub S3
  5. verify_backend    : vérifie que le backend est sain après rechargement

Prérequis dans Airflow > Admin > Variables :
  - sentimentai_admin_key              : token API d'un compte admin
  - sentimentai_training_wait_minutes  : délai d'attente en minutes (défaut 20)

URLs — tout passe par Nginx (http://nginx:80) :
  - /monitor/stats          → location /        → auth FastAPI (require_admin)
  - /train/train            → location /train/  → rate limit 1r/m
  - /internal/reload-model  → location /internal/ → deny all externe
  - /health                 → location /        → pas d'auth
"""

from datetime import datetime, timedelta
import requests

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, ShortCircuitOperator

# ── URLs — tout passe par Nginx ───────────────────────────────────────────────
NGINX_URL       = "http://nginx:80"
INTERNAL_SECRET = "airflow-internal-secret"  # doit correspondre à reload.py et .env


def get_admin_headers() -> dict:
    """Lit le token admin depuis les Variables Airflow (jamais en dur dans le code)."""
    token = Variable.get("sentimentai_admin_key")
    return {"X-API-Key": token}


# ── Tâches ────────────────────────────────────────────────────────────────────

def check_drift(**context):
    """
    Interroge /monitor/stats pour décider si un réentraînement est nécessaire.
    - Retourne False si pas de données → court-circuite le DAG
    - Retourne False si drift normal → court-circuite le DAG
    - Retourne True si needs_retraining=True → continue le DAG
    """
    resp = requests.get(
        f"{NGINX_URL}/monitor/stats",
        headers=get_admin_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    stats = resp.json()

    # Pas encore assez de données pour calculer le drift
    if stats.get("status") == "no_data":
        print("Aucune donnée de monitoring disponible — pas de réentraînement.")
        return False

    drift_level   = stats.get("drift_level", "normal")
    needs_retrain = stats.get("needs_retraining", False)
    kl            = stats.get("kl_divergence", 0)
    avg_conf      = stats.get("avg_confidence_recent", 0)

    print(
        f"Drift level     : {drift_level}\n"
        f"KL divergence   : {kl:.4f}\n"
        f"Confiance moy.  : {avg_conf:.2%}\n"
        f"Réentraînement  : {needs_retrain}"
    )

    # Pousse les infos dans XCom pour traçabilité
    context["ti"].xcom_push(key="drift_level",   value=drift_level)
    context["ti"].xcom_push(key="kl_divergence", value=kl)
    context["ti"].xcom_push(key="avg_confidence", value=avg_conf)

    return bool(needs_retrain)


def trigger_training(**context):
    """
    Déclenche l'entraînement via Nginx → /train/train (Training API).
    La training API lance l'entraînement en arrière-plan et répond immédiatement.
    Pas de retry (retries=0) pour éviter de lancer deux entraînements en parallèle.
    """
    payload = {
        "data_path":             "data/raw",
        "dataset_name":          "df_merged_clean_sample2.csv",
        "max_features":          20000,
        "ngram_max":             2,
        "learning_rate":         0.05,
        "num_leaves":            64,
        "n_estimators":          1000,
        "colsample_bytree":      0.8,
        "subsample":             0.8,
        "early_stopping_rounds": 50,
        "test_size":             0.2,
    }

    resp = requests.post(
        f"{NGINX_URL}/train/train",
        json=payload,
        timeout=30,
    )

    # 429 = rate limit Nginx atteint
    if resp.status_code == 429:
        raise Exception("Rate limit Nginx atteint sur /train/train — réessaie dans 1 minute.")

    resp.raise_for_status()
    print(f"Training API : {resp.json().get('message', resp.json())}")


def wait_for_training(**context):
    """
    Attend la fin de l'entraînement + dvc push vers DagsHub S3.
    L'entraînement tourne en arrière-plan dans la Training API —
    on attend un délai configurable via la Variable Airflow
    'sentimentai_training_wait_minutes' (défaut : 20 minutes).

    Ajuste cette valeur selon la taille de ton dataset :
      - df_merged_clean_sample.csv  (~500 lignes)  →  3-5 min
      - df_merged_clean_sample2.csv (~2700 lignes) →  10-15 min
      - df_merged_clean.csv         (dataset complet) → 30-60 min
    """
    import time

    wait_minutes = int(
        Variable.get("sentimentai_training_wait_minutes", default_var=20)
    )
    total_seconds = wait_minutes * 60
    interval      = 30  # log toutes les 30 secondes

    print(f"Attente de {wait_minutes} minutes (entraînement + dvc push vers DagsHub S3)...")

    elapsed = 0
    while elapsed < total_seconds:
        time.sleep(interval)
        elapsed += interval
        remaining = total_seconds - elapsed
        print(f"[{elapsed}s écoulées / {total_seconds}s] — encore {remaining}s à attendre...")

    print("Délai écoulé — passage au rechargement du modèle.")


def reload_model(**context):
    """
    Demande au backend de recharger le modèle depuis DagsHub S3 via DVC.
    Passe par Nginx (location /internal/) — bloqué depuis l'extérieur,
    autorisé depuis le réseau Docker interne (Airflow → Nginx → Backend).
    Le timeout est long car le dvc pull peut prendre du temps.
    """
    resp = requests.post(
        f"{NGINX_URL}/internal/reload-model",
        headers={"X-Internal-Secret": INTERNAL_SECRET},
        timeout=120,  # dvc pull peut être long selon la taille des modèles
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"Rechargement : {data.get('message', data)}")


def verify_backend(**context):
    """
    Vérifie que le backend est sain et que le nouveau modèle est bien chargé.
    Récupère les infos de drift depuis XCom pour les loguer dans le rapport final.
    """
    resp = requests.get(
        f"{NGINX_URL}/health",
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    print(f"Backend health : {data}")

    if data.get("status") != "ok" or not data.get("model_loaded"):
        raise ValueError(
            f"Backend dégradé après rechargement du modèle : {data}"
        )

    # Rapport final avec les infos de drift récupérées depuis XCom
    ti           = context["ti"]
    drift_level  = ti.xcom_pull(task_ids="check_drift", key="drift_level")
    kl           = ti.xcom_pull(task_ids="check_drift", key="kl_divergence")
    avg_conf     = ti.xcom_pull(task_ids="check_drift", key="avg_confidence")

    print(
        f"\n{'='*50}\n"
        f"Pipeline de réentraînement terminé avec succès !\n"
        f"  Drift détecté   : {drift_level} (KL={kl:.4f})\n"
        f"  Confiance avant : {avg_conf:.2%}\n"
        f"  Modèle          : rechargé depuis DagsHub S3\n"
        f"  Backend         : {data.get('status')} v{data.get('version')}\n"
        f"{'='*50}"
    )


# ── Définition du DAG ─────────────────────────────────────────────────────────

default_args = {
    "owner":            "airflow",
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="retrain_sentimentai",
    description="Réentraîne SentimentAI si drift KL ≥ 0.30 ou confiance < 55%",
    schedule_interval="0 2 * * *",  # tous les jours à 2h du matin
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["sentimentai", "ml", "retraining"],
) as dag:

    t1 = ShortCircuitOperator(
        task_id="check_drift",
        python_callable=check_drift,
        doc_md="Vérifie KL divergence. Court-circuite si drift < seuil ou pas de données.",
    )

    t2 = PythonOperator(
        task_id="trigger_training",
        python_callable=trigger_training,
        retries=0,  # pas de retry — évite deux entraînements simultanés
        doc_md="Lance l'entraînement LightGBM via Nginx /train/train.",
    )

    t3 = PythonOperator(
        task_id="wait_for_training",
        python_callable=wait_for_training,
        execution_timeout=timedelta(minutes=90),  # timeout Airflow > délai d'attente
        doc_md="Attend la fin de l'entraînement + dvc push vers DagsHub S3.",
    )

    t4 = PythonOperator(
        task_id="reload_model",
        python_callable=reload_model,
        doc_md="Backend : dvc pull depuis DagsHub S3 + rechargement en mémoire.",
    )

    t5 = PythonOperator(
        task_id="verify_backend",
        python_callable=verify_backend,
        doc_md="Vérifie que le backend répond ok avec le nouveau modèle chargé.",
    )

    t1 >> t2 >> t3 >> t4 >> t5