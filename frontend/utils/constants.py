"""Constantes partagées à travers tous les composants du frontend."""

import os

API_URL = os.getenv("API_URL", "http://nginx:80")

SENTIMENT_COLORS = {
    "Négatif": "#EF4444",
    "Neutre":  "#F59E0B",
    "Positif": "#10B981",
}

SENTIMENT_EMOJI = {
    "Négatif": "😞",
    "Neutre":  "😐",
    "Positif": "😊",
}

BADGE_CLASS = {
    "Négatif": "badge-negatif",
    "Neutre":  "badge-neutre",
    "Positif": "badge-positif",
}


def sentiment_badge(sentiment: str) -> str:
    css = BADGE_CLASS.get(sentiment, "badge-neutre")
    emoji = SENTIMENT_EMOJI.get(sentiment, "")
    return f'<span class="{css}">{emoji} {sentiment}</span>'
