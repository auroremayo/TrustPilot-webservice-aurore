"""
Onglet Monitor : surveillance du drift, recommandation de réentraînement (admin).
"""

import pandas as pd
import altair as alt
import streamlit as st

from utils import api_client
from utils.constants import SENTIMENT_COLORS


def render(token: str) -> None:
    st.header("📡 Monitor — Surveillance du modèle")
    st.markdown(
        '<div style="color:#8B949E; margin-bottom:1rem;">'
        "Détecte le concept drift et recommande un réentraînement si la distribution"
        " des prédictions récentes s'éloigne trop des données d'entraînement.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, _ = st.columns([1, 1, 2])
    days_r = c1.number_input("Fenêtre récente (j)", 1, 30, 7, key="days_r")
    days_a = c2.number_input("Fenêtre globale (j)", 7, 90, 30, key="days_a")

    if st.button("🔄 Rafraîchir", key="refresh_monitor"):
        st.rerun()

    stats = api_client.get_monitor_stats(token, days_r, days_a)
    if stats is None or "_error" in stats:
        code = stats.get("_error") if stats else "?"
        st.error(f"Erreur lors de la récupération des stats ({code}).")
        return

    if stats.get("status") == "no_data":
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#8B949E;
                    background:rgba(255,255,255,.03); border-radius:16px;
                    border:1px dashed rgba(99,102,241,.3);">
            <div style="font-size:2.5rem; margin-bottom:.5rem;">🔭</div>
            <b>Aucune donnée de monitoring.</b><br>
            <span style="font-size:.85rem;">Lancez des prédictions pour alimenter le système.</span>
        </div>
        """, unsafe_allow_html=True)
        return

    drift   = stats.get("drift_level", "normal")
    kl      = stats.get("kl_divergence", 0.0)
    retrain = stats.get("needs_retraining", False)
    avg_c   = stats.get("avg_confidence_recent", 0.0)
    fb_acc  = stats.get("feedback_accuracy")

    # ── Bannière statut ───────────────────────────────────────────────────────
    if retrain:
        st.markdown("""
        <div class="alert-critical">
            <b>🔴 RÉENTRAÎNEMENT RECOMMANDÉ</b> — Drift significatif détecté.
            La distribution récente s'est fortement éloignée de la référence d'entraînement.
        </div>""", unsafe_allow_html=True)
    elif drift == "warning":
        st.markdown("""
        <div class="alert-warning">
            <b>🟡 DRIFT MODÉRÉ</b> — Distribution légèrement éloignée. À surveiller.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-normal">
            <b>🟢 MODÈLE STABLE</b> — Distribution proche de l'entraînement.
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Métriques ─────────────────────────────────────────────────────────────
    mk = st.columns(4 if fb_acc is None else 5)
    mk[0].metric("Prédictions totales", stats.get("total_predictions", 0))
    mk[1].metric(f"Récentes ({days_r}j)",  stats.get("recent_predictions", 0))
    mk[2].metric("Confiance moy.", f"{avg_c*100:.1f}%",
                 delta=None if avg_c >= 0.55 else "⚠️ Basse")
    icons = {"normal": "🟢", "warning": "🟡", "critical": "🔴"}
    mk[3].metric("KL Divergence", f"{kl:.4f}",
                 delta=f"{icons.get(drift,'')} {drift.upper()}")
    if fb_acc is not None:
        mk[4].metric("Précision feedback", f"{fb_acc}%")

    st.divider()

    # ── Graphiques ────────────────────────────────────────────────────────────
    timeline = stats.get("daily_timeline", [])
    if not timeline:
        st.info("Pas encore assez de données pour les graphiques.")
        return

    df_tl = pd.DataFrame(timeline)
    g1, g2 = st.columns(2)

    with g1:
        chart_vol = (
            alt.Chart(df_tl)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, color="#6366F1")
            .encode(
                x=alt.X("date:T", title="Date", axis=alt.Axis(labelAngle=-30)),
                y=alt.Y("count:Q", title="Prédictions"),
                tooltip=["date:T", "count:Q"],
            )
            .properties(title="Volume journalier", height=240)
        )
        st.altair_chart(chart_vol, use_container_width=True)

    with g2:
        threshold = alt.Chart(pd.DataFrame({"y": [0.55]})).mark_rule(
            color="#EF4444", strokeDash=[6, 3]
        ).encode(y="y:Q")
        chart_conf = (
            alt.Chart(df_tl)
            .mark_line(point=True, color="#A78BFA", strokeWidth=2)
            .encode(
                x=alt.X("date:T", title="Date", axis=alt.Axis(labelAngle=-30)),
                y=alt.Y("avg_confidence:Q", title="Confiance moy.",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format=".0%")),
                tooltip=["date:T", alt.Tooltip("avg_confidence:Q", format=".1%")],
            )
            .properties(title="Tendance de confiance", height=240)
        )
        st.altair_chart(chart_conf + threshold, use_container_width=True)

    # ── Distribution récente vs entraînement ─────────────────────────────────
    st.subheader("Distribution des classes : récente vs entraînement")
    recent_d   = stats.get("recent_distribution", {})
    training_d = stats.get("training_distribution", {})
    df_comp = pd.DataFrame(
        [{"Sentiment": cls, "Proportion": recent_d.get(cls, 0),
          "Source": f"Récente ({days_r}j)"}
         for cls in ["Négatif", "Neutre", "Positif"]]
        + [{"Sentiment": cls, "Proportion": training_d.get(cls, 1/3),
            "Source": "Entraînement"}
           for cls in ["Négatif", "Neutre", "Positif"]]
    )
    chart_comp = (
        alt.Chart(df_comp)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("Sentiment", sort=None),
            y=alt.Y("Proportion", axis=alt.Axis(format=".0%")),
            color=alt.Color("Source", scale=alt.Scale(range=["#6366F1", "#374151"])),
            column=alt.Column("Source", title=None),
            tooltip=["Sentiment", alt.Tooltip("Proportion", format=".1%"), "Source"],
        )
        .properties(height=220, width=200)
    )
    st.altair_chart(chart_comp)

    # ── Distribution empilée ──────────────────────────────────────────────────
    st.subheader("Répartition journalière des sentiments")
    df_stacked = df_tl.melt(
        id_vars=["date"],
        value_vars=["Négatif", "Neutre", "Positif"],
        var_name="Sentiment", value_name="Nombre",
    )
    chart_stack = (
        alt.Chart(df_stacked)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("Nombre:Q", stack="normalize",
                    axis=alt.Axis(format=".0%"), title="Part (%)"),
            color=alt.Color(
                "Sentiment",
                scale=alt.Scale(
                    domain=["Négatif", "Neutre", "Positif"],
                    range=[SENTIMENT_COLORS["Négatif"],
                           SENTIMENT_COLORS["Neutre"],
                           SENTIMENT_COLORS["Positif"]],
                ),
            ),
            tooltip=["date:T", "Sentiment", "Nombre"],
        )
        .properties(height=220)
    )
    st.altair_chart(chart_stack, use_container_width=True)

    # ── Légende ───────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("ℹ️ Comment interpréter ces métriques ?"):
        st.markdown("""
**KL Divergence** mesure l'écart entre la distribution récente des prédictions
et la distribution d'entraînement (1/3 par classe).

| Valeur KL | Statut | Action |
|-----------|--------|--------|
| < 0.10 | 🟢 Normal | Aucune action |
| 0.10 – 0.30 | 🟡 Attention | Surveiller |
| > 0.30 | 🔴 Critique | **Réentraîner le modèle** |

**Confiance moyenne** < 55% indique que le modèle est moins sûr de ses prédictions
— souvent signe que le vocabulaire a évolué.

**Feedback** : chaque 👍/👎 améliore le suivi de la précision réelle en production.
        """)
