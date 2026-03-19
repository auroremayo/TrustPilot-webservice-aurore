"""
Pipeline NLP local : tokenisation, lemmatisation, chargement du modèle pour SHAP.
"""

import os
import re
import joblib
from pathlib import Path

import nltk
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



@st.cache_resource
def download_nltk_resources():
    for res in ["punkt", "stopwords", "wordnet", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            nltk.download(res, quiet=True)


# ── Stopwords enrichis ────────────────────────────────────────────────────────
download_nltk_resources()
_stop = set(stopwords.words("english"))
_stop.update([
    ",", ".", "`", "@", "*", "(", ")", "[", "]", "...",
    "_", ">", "<", ":", "/", "//", "///", "=", "--",
    "©", "~", ";", "\\", '"', "'", "''", '""',
    "'m", "'ve", "n't", "!", "?", "'re", "rd", "'s", "%", "-",
])
_lemmatizer = WordNetLemmatizer()


def processing_pipeline(text: str) -> str:
    """Nettoie et lemmatise un texte pour la vectorisation TF-IDF."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\.+", "", text)
    text = re.sub(r"/", " ", text)
    text = re.sub(r"[0-9]+", "", text)
    try:
        tokens = word_tokenize(text, language="english")
    except Exception:
        tokens = text.split()
    return " ".join(
        _lemmatizer.lemmatize(t) for t in tokens if t not in _stop
    )
