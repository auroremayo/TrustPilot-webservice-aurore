"""
Onglet Performance : métriques du modèle, matrice de confusion, feature importance.
"""

import pandas as pd
import streamlit as st


def render() -> None:
    st.header("🤖 Performance LightGBM")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy globale", "71.8%", "+vs baseline")
    m2.metric("F1-Score macro",   "0.72")
    m3.metric("Vocabulaire",      "5 000 mots")
    m4.metric("Algorithme",       "LightGBM")

    st.subheader("Matrice de confusion")
    st.dataframe(pd.DataFrame(
        [[8303, 2155,  558],
         [2303, 6734, 1979],
         [ 551, 1765, 8700]],
        columns=["Prédit Négatif", "Prédit Neutre", "Prédit Positif"],
        index=["Réel Négatif", "Réel Neutre", "Réel Positif"],
    ), use_container_width=True)
    st.success("✅ Très bonne détection des avis positifs et négatifs extrêmes.")
    st.warning("⚠️ La classe Neutre génère le plus de confusion (zone grise).")

    st.divider()
    st.subheader("Feature importance globale")
    c1, c2 = st.columns(2)
    c1.markdown("""
    <div class="card">
        <b style="color:#F87171;">📉 Indicateurs Négatifs</b>
        <div style="color:#C9D1D9; font-size:.9rem; line-height:2; margin-top:.6rem;">
            bad · poor · waste · return · money · broken · terrible · disappointed
        </div>
    </div>
    """, unsafe_allow_html=True)
    c2.markdown("""
    <div class="card">
        <b style="color:#34D399;">📈 Indicateurs Positifs</b>
        <div style="color:#C9D1D9; font-size:.9rem; line-height:2; margin-top:.6rem;">
            great · love · good · easy · perfect · excellent · amazing · recommend
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Méthodologie d'entraînement")
    steps = [
        ("1", "Nettoyage",       "Suppression ponctuation, chiffres, stopwords"),
        ("2", "Lemmatisation",   "WordNetLemmatizer (NLTK)"),
        ("3", "Vectorisation",   "TF-IDF · 5 000 features"),
        ("4", "Modèle",          "LightGBM avec tuning hyperparamètres"),
        ("5", "Évaluation",      "Cross-validation stratifiée · Accuracy 71.8%"),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div class="card" style="display:flex; align-items:center; gap:1rem; padding:.9rem 1.2rem;">
            <div style="background:linear-gradient(135deg,#6366F1,#8B5CF6); color:white;
                         border-radius:50%; width:32px; height:32px; flex-shrink:0;
                         display:flex; align-items:center; justify-content:center;
                         font-weight:700;">{num}</div>
            <div>
                <b style="color:#E6EDF3;">{title}</b>
                <span style="color:#8B949E; margin-left:.5rem;">— {desc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
