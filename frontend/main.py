"""
Point d'entrée Streamlit — assemble tous les composants.
"""

import sys
import os

# Garantit que les imports de type "from utils.xxx" fonctionnent
# peu importe d'où Streamlit est lancé.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
import altair as alt
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

# ── Page config (doit être le premier appel Streamlit) ────────────────────────
st.set_page_config(
    page_title="SentimentAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports locaux (après set_page_config) ────────────────────────────────────
from components.styles import inject_css
from components.sidebar import render as render_sidebar
from components.tabs import (
    demo_tab, history_tab, dataset_tab, performance_tab, monitor_tab,
)
from utils.nlp_pipeline import load_model_assets


# ── Thème Altair sombre ───────────────────────────────────────────────────────
@alt.theme.register("dark_custom", enable=True)
def _dark_theme():
    return {
        "config": {
            "background": "transparent",
            "view":   {"stroke": "transparent"},
            "axis":   {"domainColor": "#30363D", "gridColor": "#21262D",
                       "labelColor": "#8B949E",  "titleColor": "#8B949E"},
            "title":  {"color": "#E6EDF3"},
            "legend": {"labelColor": "#C9D1D9", "titleColor": "#8B949E"},
        }
    }


# ── CSS ───────────────────────────────────────────────────────────────────────
inject_css()

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("token", None), ("role", None), ("username", None),
    ("text_input", ""), ("last_result", None), ("last_timestamp", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Modèles (chargés une seule fois, partagés entre onglets) ─────────────────
model, vectorizer = load_model_assets()

# ── Sidebar ───────────────────────────────────────────────────────────────────
render_sidebar()

# ═══════════════════════════════════════════════════════════════════════════════
# CONTENU PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state["token"]:
    # ── Page d'accueil (non connecté) ─────────────────────────────────────────
    st.markdown(
        '<div class="hero-title">Analyse de Sentiment IA</div>'
        '<div class="hero-subtitle">'
        "Prédiction de satisfaction client · LightGBM + TF-IDF"
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "⚡", "Analyse instantanée", "Résultats en millisecondes via API sécurisée"),
        (c2, "🔍", "Explicabilité SHAP",  "Comprenez quels mots influencent la décision"),
        (c3, "📡", "Monitor de drift",    "Détectez quand le modèle doit être réentraîné"),
    ]:
        col.markdown(f"""
        <div class="card" style="text-align:center; height:140px;">
            <div style="font-size:2rem;">{icon}</div>
            <div style="font-weight:700; color:#E6EDF3; margin:.4rem 0 .3rem;">{title}</div>
            <div style="font-size:.83rem; color:#8B949E;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-top:2rem; padding:1.5rem;
                background:rgba(99,102,241,.08); border:1px solid rgba(99,102,241,.25);
                border-radius:16px;">
        <span style="color:#8B949E;">👈 Connectez-vous ou créez un compte pour commencer</span>
    </div>
    """, unsafe_allow_html=True)

else:
    token    = st.session_state["token"]
    is_admin = (st.session_state["role"] == "admin")

    st.markdown(
        '<div class="hero-title">SentimentAI</div>'
        '<div class="hero-subtitle">Analyse de sentiment & expérience client · LightGBM</div>',
        unsafe_allow_html=True,
    )

    # ── Onglets ───────────────────────────────────────────────────────────────
    tab_labels = ["⚡ Analyse Live", "🕑 Historique", "📊 Dataset",
                  "🤖 Performance"]
    if is_admin:
        tab_labels.append("📡 Monitor")

    tabs = st.tabs(tab_labels)

    with tabs[0]:
        demo_tab.render(model, vectorizer, token, is_admin)

    with tabs[1]:
        history_tab.render(token)

    with tabs[2]:
        dataset_tab.render()

    with tabs[3]:
        performance_tab.render()

    if is_admin:
        with tabs[4]:
            monitor_tab.render(token)
