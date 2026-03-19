"""
Surveillance du modèle : logging des prédictions, calcul de drift (divergence KL),
recommandation de réentraînement, rotation du fichier de log.
"""

import json
import logging
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta

from ..core.config import (
    PREDICTIONS_LOG,
    KL_WARNING,
    KL_CRITICAL,
    MIN_CONFIDENCE,
    MAX_LOG_LINES,
)

logger = logging.getLogger(__name__)

# Distribution de référence (dataset équilibré 3 classes)
TRAINING_DISTRIBUTION = {
    "Négatif": 1 / 3,
    "Neutre":  1 / 3,
    "Positif": 1 / 3,
}


# ── Rotation ──────────────────────────────────────────────────────────────────

def _rotate_log_if_needed() -> None:
    """
    Si le fichier dépasse MAX_LOG_LINES lignes, supprime les plus anciennes
    en ne gardant que les 80 % les plus récentes.
    """
    if not os.path.exists(PREDICTIONS_LOG):
        return
    with open(PREDICTIONS_LOG, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]

    if len(lines) <= MAX_LOG_LINES:
        return

    keep = int(MAX_LOG_LINES * 0.8)
    lines = lines[-keep:]
    with open(PREDICTIONS_LOG, "w", encoding="utf-8") as f:
        f.writelines(lines)
    logger.info("Rotation du log : %d lignes conservées sur %d.", keep, MAX_LOG_LINES)


# ── Écriture ──────────────────────────────────────────────────────────────────

def log_prediction(
    username: str,
    text: str,
    prediction: str,
    confidence: float,
    class_id: int,
) -> None:
    """Enregistre une prédiction dans le fichier JSONL + déclenche la rotation si nécessaire."""
    entry = {
        "timestamp":    datetime.now().isoformat(),
        "username":     username,
        "text_length":  len(text),
        "text_preview": text[:80],
        "prediction":   prediction,
        "confidence":   round(confidence, 4),
        "class_id":     class_id,
        "feedback":     None,
    }
    os.makedirs(os.path.dirname(PREDICTIONS_LOG), exist_ok=True)
    with open(PREDICTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    _rotate_log_if_needed()


def update_feedback(timestamp: str, feedback: str) -> bool:
    """Met à jour le feedback (correct/incorrect) d'une entrée existante."""
    if not os.path.exists(PREDICTIONS_LOG):
        return False

    lines, updated = [], False
    with open(PREDICTIONS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("timestamp") == timestamp:
                    entry["feedback"] = feedback
                    updated = True
                lines.append(json.dumps(entry, ensure_ascii=False))
            except Exception:
                lines.append(line)

    if updated:
        with open(PREDICTIONS_LOG, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return updated


# ── Lecture ───────────────────────────────────────────────────────────────────

def read_logs(days: int = 30) -> list:
    """Retourne les entrées de log des N derniers jours."""
    if not os.path.exists(PREDICTIONS_LOG):
        return []

    cutoff = datetime.now() - timedelta(days=days)
    logs = []
    with open(PREDICTIONS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if datetime.fromisoformat(entry["timestamp"]) >= cutoff:
                    logs.append(entry)
            except Exception:
                continue
    return logs


def get_user_history(username: str, limit: int = 50) -> list:
    """Retourne les N dernières prédictions d'un utilisateur."""
    all_logs = read_logs(days=365)
    return [l for l in all_logs if l.get("username") == username][-limit:]


# ── Métriques ─────────────────────────────────────────────────────────────────

def _kl_divergence(p: dict, q: dict) -> float:
    epsilon = 1e-10
    return max(
        sum(
            max(p.get(k, 0), epsilon) * math.log(
                max(p.get(k, 0), epsilon) / max(q.get(k, 0), epsilon)
            )
            for k in set(list(p) + list(q))
        ),
        0.0,
    )


def get_monitoring_stats(days_recent: int = 7, days_all: int = 30) -> dict:
    """Calcule toutes les métriques de drift et de surveillance."""
    recent_logs = read_logs(days_recent)
    all_logs    = read_logs(days_all)

    if not all_logs:
        return {
            "status":             "no_data",
            "message":            "Aucune prédiction enregistrée.",
            "total_predictions":  0,
            "recent_predictions": 0,
        }

    # Distribution récente
    cnt   = Counter(l["prediction"] for l in recent_logs)
    total = sum(cnt.values()) or 1
    recent_dist = {k: round(v / total, 4) for k, v in cnt.items()}
    for cls in ["Négatif", "Neutre", "Positif"]:
        recent_dist.setdefault(cls, 0.0)

    kl       = _kl_divergence(recent_dist, TRAINING_DISTRIBUTION)
    avg_conf = (sum(l["confidence"] for l in recent_logs) / len(recent_logs)
                if recent_logs else 0.0)

    # Timeline journalière
    daily: dict = defaultdict(lambda: {
        "count": 0, "conf_sum": 0.0,
        "Négatif": 0, "Neutre": 0, "Positif": 0,
    })
    for log in all_logs:
        d = log["timestamp"][:10]
        daily[d]["count"]    += 1
        daily[d]["conf_sum"] += log["confidence"]
        if log["prediction"] in daily[d]:
            daily[d][log["prediction"]] += 1

    timeline = [
        {
            "date":           day,
            "count":          v["count"],
            "avg_confidence": round(v["conf_sum"] / v["count"], 3) if v["count"] else 0.0,
            "Négatif":        v["Négatif"],
            "Neutre":         v["Neutre"],
            "Positif":        v["Positif"],
        }
        for day, v in sorted(daily.items())
    ]

    drift_level   = (
        "critical" if kl >= KL_CRITICAL
        else "warning" if kl >= KL_WARNING
        else "normal"
    )
    conf_alert    = avg_conf < MIN_CONFIDENCE
    needs_retrain = drift_level == "critical" or (drift_level == "warning" and conf_alert)

    feedbacks    = [l for l in all_logs if l.get("feedback") is not None]
    feedback_acc = (
        round(sum(1 for l in feedbacks if l["feedback"] == "correct") / len(feedbacks) * 100, 1)
        if feedbacks else None
    )

    all_cnt  = Counter(l["prediction"] for l in all_logs)
    all_tot  = sum(all_cnt.values()) or 1
    all_dist = {k: round(v / all_tot, 4) for k, v in all_cnt.items()}
    for cls in ["Négatif", "Neutre", "Positif"]:
        all_dist.setdefault(cls, 0.0)

    return {
        "status":                "ok",
        "total_predictions":     len(all_logs),
        "recent_predictions":    len(recent_logs),
        "avg_confidence_recent": round(avg_conf, 4),
        "kl_divergence":         round(kl, 4),
        "drift_level":           drift_level,
        "needs_retraining":      needs_retrain,
        "confidence_alert":      conf_alert,
        "recent_distribution":   recent_dist,
        "all_distribution":      all_dist,
        "training_distribution": TRAINING_DISTRIBUTION,
        "daily_timeline":        timeline,
        "feedback_accuracy":     feedback_acc,
        "days_recent":           days_recent,
        "days_all":              days_all,
    }
