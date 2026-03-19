"""
Onglet Historique : liste des prédictions passées de l'utilisateur.
"""

import pandas as pd
import altair as alt
import streamlit as st

from utils import api_client
from utils.constants import SENTIMENT_COLORS


def render(token: str) -> None:
    st.subheader("🕑 Mes dernières prédictions")
    if st.button("🔄 Rafraîchir", key="refresh_hist"):
        st.rerun()

    data = api_client.get_history(token)
    if data is None:
        st.error("Impossible de récupérer l'historique.")
        return

    history = data.get("history", [])
    if not history:
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#8B949E;
                    background:rgba(255,255,255,.03); border-radius:16px;
                    border:1px dashed rgba(99,102,241,.3);">
            <div style="font-size:2rem; margin-bottom:.5rem;">🔭</div>
            Aucune prédiction enregistrée encore.
        </div>
        """, unsafe_allow_html=True)
        return

    df = pd.DataFrame(history)

    # ── Stats rapides ─────────────────────────────────────────────────────────
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Total",          len(df))
    h2.metric("Confiance moy.", f"{df['confidence'].mean()*100:.1f}%")
    h3.metric("😊 Positif",    int((df["prediction"] == "Positif").sum()))
    h4.metric("😞 Négatif",    int((df["prediction"] == "Négatif").sum()))

    # ── Filtre ────────────────────────────────────────────────────────────────
    filt = st.selectbox(
        "Filtrer :", ["Tous", "Positif", "Neutre", "Négatif"], key="hist_filter"
    )
    df_display = df if filt == "Tous" else df[df["prediction"] == filt]

    # ── Donut chart ───────────────────────────────────────────────────────────
    dist = df["prediction"].value_counts().reset_index()
    dist.columns = ["Sentiment", "Nombre"]
    donut = (
        alt.Chart(dist)
        .mark_arc(innerRadius=50, outerRadius=90)
        .encode(
            theta="Nombre",
            color=alt.Color(
                "Sentiment",
                scale=alt.Scale(
                    domain=["Négatif", "Neutre", "Positif"],
                    range=[SENTIMENT_COLORS["Négatif"],
                           SENTIMENT_COLORS["Neutre"],
                           SENTIMENT_COLORS["Positif"]],
                ),
            ),
            tooltip=["Sentiment", "Nombre"],
        )
        .properties(title="Répartition", height=200)
    )

    left, right = st.columns([1, 2])
    left.altair_chart(donut, use_container_width=True)

    with right:
        df_show = df_display[
            ["timestamp", "text_preview", "prediction", "confidence", "feedback"]
        ].copy()
        df_show["timestamp"]  = pd.to_datetime(df_show["timestamp"]).dt.strftime("%d/%m %H:%M")
        df_show["confidence"] = (df_show["confidence"] * 100).round(1).astype(str) + "%"
        df_show.columns = ["Date", "Aperçu", "Sentiment", "Confiance", "Feedback"]
        st.dataframe(df_show, use_container_width=True, hide_index=True)
