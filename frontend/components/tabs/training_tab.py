import streamlit as st
import requests

# URL interne à Docker
TRAINING_API_URL = "http://nginx/train/train" 

def render(token: str):
    st.header("🏋️ Centre d'Entraînement du Modèle")
    st.subheader("Configuration des hyperparamètres & Lancement")
    
    st.markdown("""
    Ajustez les paramètres de la pipeline **TF-IDF + LightGBM** ci-dessous avant de lancer l'apprentissage.
    Les résultats et artefacts seront automatiquement loggés dans **MLflow**.
    """)
    
    st.divider()

    # Formulaire Streamlit pour regrouper les inputs
    with st.form("training_form"):
        
        # ── SECTION 1 : DONNÉES ───────────────────────────────────────────────
        st.markdown("##### 📁 Données d'entrée")
        c1, c2 = st.columns(2)
        with c1:
            data_path = st.text_input(
                "Chemin vers le fichier CSV de données d'entraînement", 
                value="data/raw",
                help="Chemin absolu ou relatif aux données contenues dans le stockage S3 de DagsHub."
            )
        with c2:
            dataset_name = st.text_input(
                "Nom du Dataset", 
                value="df_merged_clean_sample.csv",
                help="Nom d'identification pour le run MLflow."
            )

        st.divider()

        # ── SECTION 2 : VECTORISATION TF-IDF ──────────────────────────────────
        st.markdown("##### 🔤 Configuration TF-IDF")
        c3, c4 = st.columns(2)
        with c3:
            max_features = st.slider(
                "Max Features (Vocabulaire)", 
                min_value=1000, max_value=100000, value=20000, step=1000,
                help="Nombre maximum de tokens uniques conservés."
            )
        with c4:
            ngram_max = st.slider(
                "N-gram max", 
                min_value=1, max_value=3, value=2,
                help="1 = Unigrammes (mots seuls), 2 = Bigrammes, 3 = Trigrammes."
            )

        st.divider()

        # ── SECTION 3 : HYPERPARAMÈTRES LIGHTGBM ──────────────────────────────
        st.markdown("##### 🌳 Configuration LightGBM")
        
        c5, c6, c7 = st.columns(3)
        with c5:
            learning_rate = st.number_input(
                "Learning Rate", 
                min_value=0.001, max_value=1.0, value=0.05, step=0.01, format="%.3f"
            )
            num_leaves = st.slider(
                "Num Leaves (Feuilles)", 
                min_value=10, max_value=1000, value=64
            )
        with c6:
            n_estimators = st.slider(
                "N Estimators (Arbres)", 
                min_value=10, max_value=10000, value=1000, step=100
            )
            early_stopping_rounds = st.slider(
                "Early Stopping Rounds", 
                min_value=10, max_value=500, value=50
            )
        with c7:
            colsample_bytree = st.slider(
                "Colsample by tree", 
                min_value=0.1, max_value=1.0, value=0.8, step=0.1,
                help="Fraction de colonnes sélectionnées aléatoirement pour chaque arbre."
            )
            subsample = st.slider(
                "Subsample (Rows)", 
                min_value=0.1, max_value=1.0, value=0.8, step=0.1,
                help="Fraction des lignes utilisées pour l'entraînement de chaque arbre."
            )

        st.divider()

        # ── SECTION 4 : VALIDATION ────────────────────────────────────────────
        st.markdown("##### 🧪 Validation")
        test_size = st.slider(
            "Taille du jeu de test (Validation)", 
            min_value=0.1, max_value=0.9, value=0.2, step=0.05
        )

        # Bouton de soumission du formulaire
        submit_button = st.form_submit_button("🚀 Lancer l'entraînement du modèle", use_container_width=True)

    # ── ACTION LORS DU CLIC ───────────────────────────────────────────────────
    if submit_button:
        
        # Payload calqué à 100% sur ton objet TrainRequest Pydantic
        payload = {
            "data_path": data_path,
            "dataset_name": dataset_name,
            "max_features": int(max_features),
            "ngram_max": int(ngram_max),
            "learning_rate": float(learning_rate),
            "num_leaves": int(num_leaves),
            "n_estimators": int(n_estimators),
            "colsample_bytree": float(colsample_bytree),
            "subsample": float(subsample),
            "early_stopping_rounds": int(early_stopping_rounds),
            "test_size": float(test_size)
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        with st.spinner("🧠 Entraînement en cours... Suivez les métriques en temps réel sur MLflow."):
            try:
                # Timeout long (10 minutes) car l'entraînement peut prendre du temps
                response = requests.post(TRAINING_API_URL, json=payload, headers=headers, timeout=600)
                
                if response.status_code == 200:
                    st.success("✅ Entraînement terminé ! Les artefacts ont été poussés sur MLflow.")
                    
                    # Optionnel : Affichage propre des résultats si ton API renvoie un JSON
                    res_data = response.json()
                    if "metrics" in res_data:
                        st.json(res_data["metrics"])
                    else:
                        st.info("Consultez l'interface MLflow (port 5000) pour analyser le run.")
                        
                elif response.status_code == 429:
                    st.error("❌ Action bloquée par le Rate Limiter de Nginx (Maximum 1 requête par minute).")
                else:
                    st.error(f"❌ Erreur de l'API ({response.status_code}) : {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"💥 Impossible de joindre l'API d'entraînement via Nginx : {e}")