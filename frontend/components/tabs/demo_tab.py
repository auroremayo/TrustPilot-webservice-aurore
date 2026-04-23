"""
Onglet Analyse Live : prédiction, probabilités, SHAP, CSV bulk.
"""

import pandas as pd
import altair as alt
import shap
import streamlit as st

from utils import api_client
from utils.constants import SENTIMENT_COLORS, SENTIMENT_EMOJI, sentiment_badge
from common.nlp_pipeline import processing_pipeline


def render(model, vectorizer, token: str, is_admin: bool) -> None:
    def set_text(t):
        st.session_state.text_input = t

    st.subheader("Testez l'IA en temps réel")

    # ── Boutons preset ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.button("😡 Exemple négatif", use_container_width=True, on_click=set_text,
              args=["Horrible service, I waited 2 weeks and the package is broken. Never again!"])
    c2.button("😐 Exemple neutre",  use_container_width=True, on_click=set_text,
              args=["The product is okay but shipping was a bit slow. Not bad, not great."])
    c3.button("😊 Exemple positif", use_container_width=True, on_click=set_text,
              args=["Absolutely amazing! Best purchase of the year, highly recommended."])

    user_input = st.text_area(
        "Votre commentaire :", value=st.session_state.text_input,
        height=110, placeholder="Écrivez ou collez un avis client en anglais…",
    )

    # ── Analyse ───────────────────────────────────────────────────────────────
    if st.button("🚀 Lancer l'analyse", type="primary"):
        if not user_input.strip():
            st.warning("Veuillez entrer un texte à analyser.")
        else:
            with st.spinner("Analyse en cours…"):
                res = api_client.predict(token, user_input)

            if res.status_code == 200:
                st.session_state["last_result"]    = res.json()
                st.session_state["last_timestamp"] = res.json().get("timestamp")
            elif res.status_code == 403:
                st.error("🛑 Quota journalier atteint ou accès non autorisé.")
            elif res.status_code == 503:
                st.error("⏱️ Trop de requêtes — patientez quelques secondes.")
            else:
                st.error(f"Erreur API ({res.status_code})")

    # ── Résultat ──────────────────────────────────────────────────────────────
    result = st.session_state.get("last_result")
    if result:
        sentiment  = result["sentiment"]
        confidence = float(result["prediction_score"].replace("%", ""))
        pred_class = result["class_id"]
        color      = SENTIMENT_COLORS.get(sentiment, "#A78BFA")

        st.markdown("---")
        r1, r2 = st.columns([1, 2], gap="large")

        with r1:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div style="font-size:.82rem; color:#8B949E; margin-bottom:.6rem;">VERDICT</div>
                {sentiment_badge(sentiment)}
                <div style="margin-top:1rem; font-size:2.2rem; font-weight:800; color:{color};">
                    {confidence:.1f}%
                </div>
                <div style="color:#8B949E; font-size:.8rem;">confiance</div>
            </div>
            """, unsafe_allow_html=True)

            # Feedback
            ts = st.session_state.get("last_timestamp")
            if ts:
                st.markdown(
                    '<div style="text-align:center; color:#8B949E; font-size:.82rem; margin-top:.5rem;">'
                    "Ce résultat est-il correct ?</div>",
                    unsafe_allow_html=True,
                )
                fb1, fb2 = st.columns(2)
                if fb1.button("👍", use_container_width=True, key="fb_ok"):
                    api_client.send_feedback(token, ts, "correct")
                    st.success("Merci !")
                if fb2.button("👎", use_container_width=True, key="fb_ko"):
                    api_client.send_feedback(token, ts, "incorrect")
                    st.info("Feedback enregistré.")

        with r2:
            if model and vectorizer:
                clean  = processing_pipeline(user_input)
                vec_in = vectorizer.transform([clean])
                arr    = vec_in.toarray()
                proba  = model.predict_proba(arr)[0]

                df_p = pd.DataFrame({
                    "Sentiment":   ["Négatif", "Neutre", "Positif"],
                    "Probabilité": proba,
                })
                chart = (
                    alt.Chart(df_p)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X("Sentiment", sort=None, axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("Probabilité", axis=alt.Axis(format=".0%")),
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
                        tooltip=["Sentiment", alt.Tooltip("Probabilité", format=".1%")],
                    )
                    .properties(title="Distribution des probabilités", height=220)
                )
                st.altair_chart(chart, use_container_width=True)

        # ── SHAP ─────────────────────────────────────────────────────────────
        if model and vectorizer:
            st.markdown("---")
            st.subheader("🔎 Analyse SHAP — Impact des mots")
            try:
                arr        = vectorizer.transform([processing_pipeline(user_input)]).toarray()
                explainer  = shap.TreeExplainer(model)
                shap_vals  = explainer.shap_values(arr)

                if isinstance(shap_vals, list):
                    vals = shap_vals[pred_class][0] if len(shap_vals) > pred_class else shap_vals[0][0]
                elif len(shap_vals.shape) == 3:
                    vals = shap_vals[0, :, pred_class]
                else:
                    vals = shap_vals[0]

                feat_names = vectorizer.get_feature_names_out()
                df_shap = pd.DataFrame({
                    "Mot": feat_names, "SHAP": vals,
                    "TF-IDF": arr[0],
                })
                df_shap = df_shap[df_shap["TF-IDF"] > 0].copy()

                if df_shap.empty:
                    st.info("Aucun mot-clé connu détecté dans ce texte.")
                else:
                    df_top = df_shap.assign(Abs=df_shap["SHAP"].abs()).nlargest(12, "Abs")
                    chart_shap = (
                        alt.Chart(df_top)
                        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                        .encode(
                            x=alt.X("SHAP", title="Impact sur la décision"),
                            y=alt.Y("Mot", sort="-x", title=None),
                            color=alt.condition(
                                alt.datum["SHAP"] > 0,
                                alt.value(SENTIMENT_COLORS.get(sentiment, "#6366F1")),
                                alt.value("#374151"),
                            ),
                            tooltip=["Mot", alt.Tooltip("SHAP", format=".4f")],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(chart_shap, use_container_width=True)
                    st.caption(f"🟨 Coloré : renforce **{sentiment}** · ⬛ Gris : s'y oppose")
            except Exception as e:
                st.warning(f"SHAP indisponible : {e}")

    # ── CSV Bulk ─────────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📂 Analyse de masse — Fichier CSV", expanded=False):
        if not is_admin:
            st.warning("🔒 L'analyse de masse est réservée aux administrateurs.")
            return
        if model is None:
            st.error("Modèles locaux requis pour le traitement en masse.")
            return

        st.download_button(
            "📥 Modèle CSV",
            "text\nExemple: Super produit !\nExemple: Livraison trop longue...",
            "modele_avis.csv", "text/csv",
        )
        uploaded = st.file_uploader("Déposez votre CSV", type=["csv"])
        if uploaded:
            try:
                df   = pd.read_csv(uploaded)
                cols = [c for c in df.columns
                        if "text" in c.lower() or "review" in c.lower()]
                if not cols:
                    st.error("Colonne 'text' ou 'review' introuvable.")
                    return
                if st.button(f"Analyser {len(df)} avis"):
                    col_name   = cols[0]
                    df["clean"] = df[col_name].astype(str).apply(processing_pipeline)
                    vecs        = vectorizer.transform(df["clean"])
                    preds       = model.predict(vecs.toarray())
                    mapping     = {0: "Négatif", 1: "Neutre", 2: "Positif"}
                    df["Prediction"] = [mapping[p] for p in preds]

                    counts = df["Prediction"].value_counts()
                    b1, b2, b3 = st.columns(3)
                    b1.metric("😊 Positif", counts.get("Positif", 0))
                    b2.metric("😐 Neutre",  counts.get("Neutre", 0))
                    b3.metric("😞 Négatif", counts.get("Négatif", 0))

                    st.dataframe(df[[col_name, "Prediction"]], use_container_width=True)
                    st.download_button(
                        "📥 Télécharger résultats",
                        df.to_csv(index=False).encode("utf-8"),
                        "resultats.csv", "text/csv",
                    )
            except Exception as e:
                st.error(f"Erreur CSV : {e}")
