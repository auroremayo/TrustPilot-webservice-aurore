"""
Onglet Dataset : description et aperçu du jeu de données.
"""

import pandas as pd
import altair as alt
import streamlit as st

from utils.constants import SENTIMENT_COLORS


def render() -> None:
    st.header("📊 Jeu de données : Amazon Electronics")
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("""
        <div class="card">
            <div style="font-size:1.8rem; margin-bottom:.4rem;">📦</div>
            <b style="color:#E6EDF3;">Amazon Reviews Dataset</b>
            <ul style="color:#C9D1D9; margin-top:.6rem; padding-left:1.2rem;
                        font-size:.88rem; line-height:1.8;">
                <li>Structure identique : Texte + Note (1-5 ★)</li>
                <li>Focus Électronique : vocabulaire riche</li>
                <li>Période : 2010 – 2018</li>
                <li>Langue : Anglais</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("#### Volumétrie & Nettoyage")
        st.dataframe(pd.DataFrame({
            "Étape":   ["Avis bruts", "Après filtrage", "Après équilibrage"],
            "Volume":  ["~1 200 000", "572 950", "572 949"],
        }), hide_index=True, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.error("❌ **Prix**\nSupprimé (NAs)")
        c2.warning("⚠️ **Votes**\nImputé à 0")
        c3.info("🖼️ **Image**\nBooléen")

    st.divider()
    st.subheader("Aperçu des données")
    st.dataframe(pd.DataFrame({
        "Note":  [5, 1, 3, 5, 2],
        "Résumé": ["Amazing sound", "Waste of money", "Average", "Great service", "Disappointed"],
        "Texte": [
            "This headphone is amazing! The bass is deep.",
            "Terrible quality, stopped working after 2 days.",
            "It's okay for the price.",
            "Works perfectly, fast delivery.",
            "Poor screen resolution, not as advertised.",
        ],
        "Marque": ["Bose", "Generic", "Sony", "Samsung", "LG"],
    }), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Distribution des classes (équilibrée)")
    df_bal = pd.DataFrame({
        "Sentiment": ["Négatif", "Neutre", "Positif"],
        "Nombre":    [190983, 190983, 190983],
    })
    chart = (
        alt.Chart(df_bal)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("Sentiment", sort=None, axis=alt.Axis(labelAngle=0)),
            y="Nombre",
            color=alt.Color(
                "Sentiment",
                scale=alt.Scale(
                    domain=["Négatif", "Neutre", "Positif"],
                    range=[SENTIMENT_COLORS["Négatif"],
                           SENTIMENT_COLORS["Neutre"],
                           SENTIMENT_COLORS["Positif"]],
                ),
                legend=None,
            ),
            tooltip=["Sentiment", "Nombre"],
        )
        .properties(height=250)
    )
    st.altair_chart(chart, use_container_width=True)
