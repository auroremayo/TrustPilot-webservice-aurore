"""
Métriques Prometheus pour SentimentAI.
Exposées via /metrics (prometheus-fastapi-instrumentator).
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Prédictions ───────────────────────────────────────────────────────────────

PREDICTIONS_COUNTER = Counter(
    "sentimentai_predictions_total",
    "Nombre total de prédictions par sentiment",
    ["sentiment"],
)

CONFIDENCE_HISTOGRAM = Histogram(
    "sentimentai_prediction_confidence",
    "Distribution des scores de confiance",
    buckets=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

# ── Drift / Airflow ───────────────────────────────────────────────────────────

KL_DIVERGENCE_GAUGE = Gauge(
    "sentimentai_kl_divergence",
    "KL divergence entre distribution récente et entraînement",
)

DRIFT_LEVEL_GAUGE = Gauge(
    "sentimentai_drift_level",
    "Niveau de drift : 0=normal, 1=warning, 2=critical",
)

NEEDS_RETRAINING_GAUGE = Gauge(
    "sentimentai_needs_retraining",
    "Réentraînement nécessaire : 0 ou 1",
)

AVG_CONFIDENCE_GAUGE = Gauge(
    "sentimentai_avg_confidence",
    "Confiance moyenne sur les 7 derniers jours",
)

RECENT_PREDICTIONS_GAUGE = Gauge(
    "sentimentai_recent_predictions",
    "Nombre de prédictions sur les 7 derniers jours",
)

# ── Feedback ─────────────────────────────────────────────────────────────────

FEEDBACK_COUNTER = Counter(
    "sentimentai_feedback_total",
    "Feedbacks par type (correct / incorrect)",
    ["feedback_type"],
)

FEEDBACK_ACCURACY_GAUGE = Gauge(
    "sentimentai_feedback_accuracy",
    "Précision estimée via feedbacks utilisateurs (0-1)",
)

# ── Système ───────────────────────────────────────────────────────────────────

MODEL_LOADED_GAUGE = Gauge(
    "sentimentai_model_loaded",
    "Modèle ML chargé en mémoire : 0 ou 1",
)

ACTIVE_USERS_GAUGE = Gauge(
    "sentimentai_active_users",
    "Nombre d'utilisateurs enregistrés",
)

_DRIFT_LEVEL_MAP = {"normal": 0, "warning": 1, "critical": 2}


def update_drift_metrics(stats: dict) -> None:
    """Met à jour les jauges Prometheus à partir des stats de monitoring."""
    if stats.get("status") == "no_data":
        return

    KL_DIVERGENCE_GAUGE.set(stats.get("kl_divergence", 0.0))
    DRIFT_LEVEL_GAUGE.set(_DRIFT_LEVEL_MAP.get(stats.get("drift_level", "normal"), 0))
    NEEDS_RETRAINING_GAUGE.set(1 if stats.get("needs_retraining") else 0)
    AVG_CONFIDENCE_GAUGE.set(stats.get("avg_confidence_recent", 0.0))
    RECENT_PREDICTIONS_GAUGE.set(stats.get("recent_predictions", 0))

    fb_acc = stats.get("feedback_accuracy")
    if fb_acc is not None:
        FEEDBACK_ACCURACY_GAUGE.set(fb_acc / 100.0)
