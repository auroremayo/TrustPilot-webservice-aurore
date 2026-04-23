"""
Sidebar : authentification, profil, barre de quota, infos modèle.
"""

import streamlit as st
from utils import api_client


def render() -> None:
    """Affiche la sidebar. Met à jour st.session_state directement."""
    with st.sidebar:
        # ── Logo ──────────────────────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center; padding:1rem 0 .5rem;">
            <div style="font-size:2.5rem;">🧠</div>
            <div style="font-size:1.3rem; font-weight:800;
                        background:linear-gradient(135deg,#6366F1,#A78BFA);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        background-clip:text;">SentimentAI</div>
            <div style="color:#8B949E; font-size:.78rem;">Analyse d'avis textuels</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # ── Auth ──────────────────────────────────────────────────────────────
        if st.session_state["token"] is None:
            choix = st.radio("Choisissez une action", ["Connexion", "Inscription"],
                             horizontal=True, label_visibility="collapsed")
            st.divider()

            if choix == "Inscription":
                st.markdown("#### 📝 Créer un compte")
                new_user = st.text_input("Pseudo", key="reg_user")
                new_pass = st.text_input("Mot de passe", type="password", key="reg_pass")
                if st.button("S'inscrire", use_container_width=True):
                    if new_user and new_pass:
                        r = api_client.register(new_user, new_pass)
                        if r.status_code == 200:
                            st.success("✅ Compte créé ! Connectez-vous.")
                        elif r.status_code == 400:
                            st.error("❌ Ce pseudo existe déjà.")
                        elif r.status_code == 422:
                            try:
                                detail = r.json().get("detail", [])
                                msgs = [d.get("msg", "") for d in detail] if isinstance(detail, list) else [str(detail)]
                                st.error("❌ " + " | ".join(msgs))
                            except Exception:
                                st.error("❌ Pseudo ou mot de passe invalide.")
                        else:
                            st.error(f"❌ Erreur serveur ({r.status_code}).")
                    else:
                        st.warning("Remplissez tous les champs.")

            else:
                st.markdown("#### 🔒 Connexion")
                username = st.text_input("Pseudo", key="log_user")
                password = st.text_input("Mot de passe", type="password", key="log_pass")
                if st.button("Se connecter", use_container_width=True):
                    r = api_client.login(username, password)
                    if r.status_code == 200:
                        data = r.json()
                        st.session_state["token"]    = data["access_token"]
                        st.session_state["role"]     = data["role"]
                        st.session_state["username"] = data.get("username", username)
                        st.rerun()
                    else:
                        st.error("❌ Identifiants incorrects.")

        else:
            # ── Profil ────────────────────────────────────────────────────────
            role  = st.session_state["role"]
            uname = st.session_state["username"] or "Utilisateur"
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div style="font-size:2rem;">👤</div>
                <div style="font-weight:700; color:#E6EDF3;">{uname}</div>
                <div style="color:#8B949E; font-size:.85rem;">{role.upper()}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Quota ─────────────────────────────────────────────────────────
            if role != "admin":
                quota = api_client.get_quota(st.session_state["token"])
                if quota:
                    used  = quota.get("quota_used", 0)
                    maxi  = quota.get("quota_max", 5)
                    pct   = int(used / maxi * 100) if maxi else 0
                    color = "#EF4444" if pct >= 80 else "#F59E0B" if pct >= 50 else "#6366F1"
                    st.markdown(f"""
                    <div style="margin-top:.5rem;">
                        <div style="font-size:.8rem; color:#8B949E; margin-bottom:4px;">
                            Quota journalier : <b style="color:#E6EDF3;">{used}/{maxi}</b>
                        </div>
                        <div class="quota-bar-bg">
                            <div class="quota-bar-fill"
                                 style="width:{pct}%; background:{color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="color:#A78BFA; font-size:.82rem;">✨ Quota illimité</div>',
                    unsafe_allow_html=True,
                )

            st.divider()
            if st.button("Se déconnecter", use_container_width=True):
                for k in ("token", "role", "username", "last_result", "last_timestamp"):
                    st.session_state[k] = None
                st.session_state["text_input"] = ""
                st.rerun()

        # ── Infos modèle ──────────────────────────────────────────────────────
        st.divider()
        st.markdown("#### ℹ️ Modèle")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", "71.8%")
        c2.metric("F1-Score", "0.72")
        st.markdown(
            '<div style="color:#8B949E; font-size:.78rem; margin-top:.5rem;">'
            "LightGBM · TF-IDF · 5 000 features</div>",
            unsafe_allow_html=True,
        )
        st.divider()
        st.caption("DataScientest · Lakdar & Aurore")
