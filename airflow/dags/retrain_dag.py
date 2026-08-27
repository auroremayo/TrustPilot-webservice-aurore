"""
DAG Airflow — Pipeline de réentraînement du modèle SentimentAI avec Quality Gate MLOps.

Étapes :
  1. check_drift       : vérifie la KL divergence via /monitor/stats
                         court-circuite si drift < seuil ou pas de données
  2. trigger_training  : déclenche l'entraînement via /train/train (Nginx) et récupère un job_id
  3. wait_for_training : PythonSensor (mode=reschedule) sonde /train/status/{job_id} jusqu'à SUCCESS
  4. evaluate_model    : Quality Gate MLOps (ShortCircuitOperator). Compare les métriques
                         (Accuracy & F1-Score) aux seuils minimaux requis.
                         Court-circuite et bloque le rechargement si le modèle régresse.
  5. reload_model      : backend recharge le modèle depuis DagsHub S3 via DVC
  6. verify_backend    : vérifie que le backend est sain après rechargement et affiche le bilan complet

Prérequis dans Airflow > Admin > Variables (avec valeurs par défaut automatiques) :
  - sentimentai_admin_key     : token API d'un compte admin
  - sentimentai_min_accuracy  : seuil minimal d'accuracy (défaut : 0.70)
  - sentimentai_min_f1        : seuil minimal de F1-weighted (défaut : 0.70)

URLs — tout passe par Nginx (http://nginx:80) :
  - /monitor/stats          → location /        → auth FastAPI (require_admin)
  - /train/train            → location /train/  → rate limit 1r/m
  - /train/status/{job_id}  → location /train/  → statut asynchrone
  - /internal/reload-model  → location /internal/ → deny all externe
  - /health                 → location /        → pas d'auth
"""

from datetime import datetime, timedelta
import requests

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.sensors.python import PythonSensor
from airflow.exceptions import AirflowException

# ── URLs — tout passe par Nginx ───────────────────────────────────────────────
NGINX_URL       = "http://nginx:80"
INTERNAL_SECRET = "airflow-internal-secret"  # doit correspondre à reload.py et .env


def get_admin_headers() -> dict:
    """Lit le token admin depuis les Variables Airflow. """
    token = Variable.get("sentimentai_admin_key", default_var="")
    return {"X-API-Key": token} if token else {}


# ── Tâches ────────────────────────────────────────────────────────────────────

def check_drift(**context) -> bool:
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
    context["ti"].xcom_push(key="drift_level",    value=drift_level)
    context["ti"].xcom_push(key="kl_divergence",  value=kl)
    context["ti"].xcom_push(key="avg_confidence", value=avg_conf)

    return bool(needs_retrain)


def trigger_training(**context):
    """
    Déclenche l'entraînement via Nginx → /train/train (Training API).
    La training API lance l'entraînement en arrière-plan, retourne un job_id
    et répond immédiatement.
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
        raise Exception("Rate limit Nginx atteint sur /train/train — réessayer dans 1 minute.")

    resp.raise_for_status()
    data = resp.json()
    job_id = data.get("job_id")
    print(f"Training API response : {data}")

    if not job_id:
        raise ValueError(f"Aucun job_id retourné par l'API de training : {data}")

    # Enregistre le job_id dans XCom pour le Sensor
    context["ti"].xcom_push(key="job_id", value=job_id)
    print(f"✅ Job d'entraînement lancé avec succès. Job ID: {job_id}")


def check_training_status(**context) -> bool:
    """
    Sonde le statut du job d'entraînement via /train/status/{job_id}.
    - Si SUCCESS : extrait run_id & métriques, les pousse dans XCom et termine avec succès (True).
    - Si FAILED  : lève une exception AirflowException pour marquer la tâche en échec.
    - Si RUNNING / PENDING : retourne False (le sensor libère le worker en mode reschedule).
    """
    ti = context["ti"]
    # 1. Airflow récupère le job_id stocké
    job_id = ti.xcom_pull(task_ids="trigger_training", key="job_id")
    if not job_id:
        raise AirflowException("Job ID introuvable dans XCom depuis la tâche trigger_training.")
    
    # 2. Airflow interroge la route /status/{job_id} de FastAPI
    resp = requests.get(
        f"{NGINX_URL}/train/status/{job_id}",
        timeout=10,
    )
    if resp.status_code == 404:
        raise AirflowException(f"Job '{job_id}' introuvable sur l'API de training.")

    resp.raise_for_status()
    job_info = resp.json()
    job_status = job_info.get("status")
    duration = job_info.get("duration_seconds")

    print(f"[Sensor] Statut du job {job_id} : {job_status} (durée : {duration}s)")
    
    # 3. Dès que FastAPI renvoie "SUCCESS", Airflow extrait les métriques du résultat :
    if job_status == "SUCCESS":
        result = job_info.get("result", {})
        print(f"Entraînement terminé avec succès : {result}")

        # Airflow enregistre les résultats renvoyés par train(...) pour la Quality Gate
        ti.xcom_push(key="run_id",    value=result.get("run_id"))
        ti.xcom_push(key="accuracy",  value=result.get("accuracy"))
        ti.xcom_push(key="f1_score",  value=result.get("f1_score"))
        return True

    elif job_status == "FAILED":
        error = job_info.get("error", "Erreur inconnue lors du réentraînement")
        raise AirflowException(f"Échec de l'entraînement (Job {job_id}) : {error}")

    elif job_status in ["PENDING", "RUNNING"]:
        print(f"Entraînement toujours en cours ({job_status}). Reprogrammation du sondage...")
        return False

    else:
        raise AirflowException(f"Statut inconnu reçu pour le job {job_id} : {job_status}")


MLFLOW_URL = "http://mlflow:5000"
MODEL_NAME = "SentimentAI-LightGBM"


def get_production_model_metrics(model_name: str = MODEL_NAME) -> dict:
    """
    Interroge le Model Registry de MLflow pour récupérer les métriques du modèle
    actuellement tagué avec l'alias '@production'.
    """
    try:
        resp = requests.get(
            f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/alias?name={model_name}&alias=production",
            timeout=10,
        )
        if resp.status_code != 200:
            return None

        mv = resp.json().get("model_version", {})
        version = mv.get("version")
        run_id  = mv.get("run_id")
        if not run_id:
            return None

        # Récupère les métriques du run associé au modèle de production
        r_run = requests.get(f"{MLFLOW_URL}/api/2.0/mlflow/runs/get?run_id={run_id}", timeout=10)
        if r_run.status_code != 200:
            return None

        metrics = {m["key"]: m["value"] for m in r_run.json().get("run", {}).get("data", {}).get("metrics", [])}
        return {
            "version":  version,
            "run_id":   run_id,
            "accuracy": round(float(metrics.get("accuracy", 0.0)), 4),
            "f1_score": round(float(metrics.get("f1_weighted", 0.0)), 4),
        }
    except Exception as e:
        print(f"Avertissement : Impossible d'interroger le modèle @production dans MLflow : {e}")
        return None


def get_model_version_by_run_id(run_id: str, model_name: str = MODEL_NAME) -> str:
    """Retrouve le numéro de version (v1, v2, etc.) associé à un run dans le Model Registry."""
    try:
        resp = requests.get(f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/get?name={model_name}", timeout=10)
        if resp.status_code == 200:
            versions = resp.json().get("registered_model", {}).get("latest_versions", [])
            for v in versions:
                if v.get("run_id") == run_id:
                    return v.get("version")
            if versions:
                return versions[0].get("version")
    except Exception as e:
        print(f"Avertissement : Erreur recherche version : {e}")
    return "1"


def promote_to_production(version: str, model_name: str = MODEL_NAME) -> bool:
    """
    Associe l'alias '@production' à la version spécifiée dans MLflow Model Registry.
    MLflow retire automatiquement l'alias de l'ancienne version.
    """
    try:
        payload = {"name": model_name, "alias": "production", "version": str(version)}
        resp = requests.post(f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/alias", json=payload, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"Erreur lors de la promotion vers @production : {e}")
        return False


def evaluate_model(**context) -> bool:
    """
    Quality Gate MLOps Dynamique (@production vs Challenger) :
    Compare les performances du nouveau modèle (Challenger) avec le modèle
    actuellement marqué '@production' dans MLflow Model Registry.

    - Si @production existe : Le nouveau modèle doit être au moins aussi bon
      (ou dans la limite de tolérance `sentimentai_max_allowed_drop`, défaut: 0.00).
    - Si le Challenger gagne : Airflow déplace automatiquement l'alias '@production'
      sur la nouvelle version dans MLflow et autorise le rechargement.
    - Si le Challenger perd : L'alias '@production' reste sur l'ancien modèle stable
      et le déploiement est court-circuité.
    """
    ti = context["ti"]
    challenger_run_id = ti.xcom_pull(task_ids="wait_for_training", key="run_id")
    challenger_acc    = ti.xcom_pull(task_ids="wait_for_training", key="accuracy")
    challenger_f1     = ti.xcom_pull(task_ids="wait_for_training", key="f1_score")

    if challenger_acc is None or challenger_f1 is None:
        print("❌ Métriques du nouveau modèle introuvables dans XCom — Déploiement bloqué par sécurité.")
        ti.xcom_push(key="quality_gate", value="REJECTED_NO_METRICS")
        return False

    # Identification de la version du nouveau modèle
    challenger_version = get_model_version_by_run_id(run_id=challenger_run_id)

    # Récupération du modèle actuellement en @production dans MLflow
    prod_model = get_production_model_metrics()

    # Tolérances et seuils
    max_drop = float(Variable.get("sentimentai_max_allowed_drop", default_var=0.00))
    min_absolute_acc = float(Variable.get("sentimentai_min_accuracy", default_var=0.70))
    min_absolute_f1  = float(Variable.get("sentimentai_min_f1",       default_var=0.70))

    print(f"\n{'='*70}")
    print(f"🥊 QUALITY GATE MLOPS — MATCH @PRODUCTION vs NOUVEAU MODÈLE")
    print(f"  • Challenger (Nouveau)  : Version {challenger_version} (Run: {challenger_run_id})")
    print(f"    - Accuracy : {challenger_acc:.4f}")
    print(f"    - F1-Score : {challenger_f1:.4f}")

    if prod_model:
        prod_acc  = prod_model["accuracy"]
        prod_f1   = prod_model["f1_score"]
        prod_ver  = prod_model["version"]
        prod_run  = prod_model["run_id"]
        delta_acc = challenger_acc - prod_acc
        delta_f1  = challenger_f1 - prod_f1

        print(f"  • Production Actuelle   : @production = Version {prod_ver} (Run: {prod_run})")
        print(f"    - Accuracy : {prod_acc:.4f}")
        print(f"    - F1-Score : {prod_f1:.4f}")
        print(f"  • Deltas de performance :")
        print(f"    - Δ Accuracy : {delta_acc:+.4f} (Tolérance max drop : -{max_drop})")
        print(f"    - Δ F1-Score : {delta_f1:+.4f} (Tolérance max drop : -{max_drop})")
        print(f"{'='*70}")

        # Condition de validation : dépasse ou égale la prod (et respecte le seuil absolu)
        acc_approved = challenger_acc >= (prod_acc - max_drop) and challenger_acc >= min_absolute_acc
        f1_approved  = challenger_f1 >= (prod_f1 - max_drop) and challenger_f1 >= min_absolute_f1

        if acc_approved and f1_approved:
            # ── PROMOTION OFFICIELLE DANS MLFLOW ─────────────────────────
            promoted = promote_to_production(version=challenger_version)
            if promoted:
                print(f"🏷️ [MLflow] Alias '@production' assigné avec succès à la Version {challenger_version}.")
            else:
                print(f"⚠️ [MLflow] Impossible d'assigner l'alias via API, continuation du rechargement.")

            print(
                f"✅ VICTOIRE DU NOUVEAU MODÈLE : Performances validées !\n"
                f"   L'alias @production a été mis à jour dans MLflow.\n"
                f"   Autorisation du rechargement à chaud dans l'API de prédiction."
            )
            ti.xcom_push(key="quality_gate", value="PASSED_PROMOTED_TO_PRODUCTION")
            ti.xcom_push(key="promoted_version", value=challenger_version)
            ti.xcom_push(key="delta_accuracy", value=round(delta_acc, 4))
            ti.xcom_push(key="delta_f1", value=round(delta_f1, 4))
            return True
        else:
            print(
                f"🚨 RÉGRESSION DÉTECTÉE : Le nouveau modèle est inférieur à la version actuelle en @production.\n"
                f"   Déploiement BLOQUÉ pour protéger les utilisateurs.\n"
                f"   La Version {prod_ver} reste en @production."
            )
            ti.xcom_push(key="quality_gate", value=f"REJECTED_WORSE_THAN_PRODUCTION (ΔAcc={delta_acc:+.4f})")
            return False

    else:
        # Aucun modèle @production configuré (premier run)
        print(f"  • Aucun modèle tagué @production (Initialisation).")
        print(f"  • Validation contre seuils absolus : Acc ≥ {min_absolute_acc}, F1 ≥ {min_absolute_f1}")
        print(f"{'='*70}")

        if challenger_acc >= min_absolute_acc and challenger_f1 >= min_absolute_f1:
            promote_to_production(version=challenger_version)
            print(f"✅ MODÈLE INITIAL PROMU @PRODUCTION (Version {challenger_version}).")
            ti.xcom_push(key="quality_gate", value="PASSED_INITIAL_PRODUCTION")
            ti.xcom_push(key="promoted_version", value=challenger_version)
            return True
        else:
            print(f"🚨 MODÈLE INITIAL REJETÉ : Métriques sous les seuils de base.")
            ti.xcom_push(key="quality_gate", value="FAILED_BELOW_BASE_THRESHOLDS")
            return False


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
    Récupère les infos de drift et de training depuis XCom pour le rapport final.
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

    # Rapport final complet avec traçabilité XCom
    ti           = context["ti"]
    drift_level  = ti.xcom_pull(task_ids="check_drift", key="drift_level")
    kl           = ti.xcom_pull(task_ids="check_drift", key="kl_divergence")
    avg_conf     = ti.xcom_pull(task_ids="check_drift", key="avg_confidence")
    job_id       = ti.xcom_pull(task_ids="trigger_training", key="job_id")
    run_id       = ti.xcom_pull(task_ids="wait_for_training", key="run_id")
    accuracy     = ti.xcom_pull(task_ids="wait_for_training", key="accuracy")
    f1_score     = ti.xcom_pull(task_ids="wait_for_training", key="f1_score")
    qg_status    = ti.xcom_pull(task_ids="evaluate_model", key="quality_gate")

    print(
        f"\n{'='*60}\n"
        f"🎯 Pipeline de réentraînement SentimentAI terminé avec succès !\n"
        f"  - Drift détecté      : {drift_level} (KL={kl if kl is not None else 0:.4f})\n"
        f"  - Confiance avant    : {avg_conf if avg_conf is not None else 0:.2%}\n"
        f"  - Job ID Training    : {job_id}\n"
        f"  - Run MLflow         : {run_id}\n"
        f"  - Quality Gate       : {qg_status}\n"
        f"  - Nouvelle Accuracy  : {accuracy}\n"
        f"  - Nouveau F1-Score   : {f1_score}\n"
        f"  - Backend API        : {data.get('status')} v{data.get('version')}\n"
        f"{'='*60}"
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
    description="Réentraîne SentimentAI si drift KL ≥ 0.30 ou confiance < 55% avec Quality Gate",
    schedule_interval="0 2 * * *",  # tous les jours à 2h du matin
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["sentimentai", "ml", "retraining", "quality-gate"],
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
        doc_md="Lance l'entraînement LightGBM via Nginx /train/train et récupère le job_id.",
    )

    t3 = PythonSensor(
        task_id="wait_for_training",
        python_callable=check_training_status,
        mode="reschedule",       # Libère le worker Celery entre deux sondages !
        poke_interval=20,        # Vérifie toutes les 20 secondes
        timeout=3600,            # Timeout de sécurité (1 heure)
        exponential_backoff=False,
        doc_md="Sonde /train/status/{job_id} en mode reschedule jusqu'à SUCCESS ou FAILED.",
    )

    t4 = ShortCircuitOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
        doc_md="Quality Gate : valide que Accuracy et F1-Score dépassent les seuils avant déploiement.",
    )

    t5 = PythonOperator(
        task_id="reload_model",
        python_callable=reload_model,
        doc_md="Backend : dvc pull depuis DagsHub S3 + rechargement à chaud en mémoire.",
    )

    t6 = PythonOperator(
        task_id="verify_backend",
        python_callable=verify_backend,
        doc_md="Vérifie que le backend répond ok avec le nouveau modèle chargé.",
    )

    t1 >> t2 >> t3 >> t4 >> t5 >> t6