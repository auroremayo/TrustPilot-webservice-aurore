import pandas as pd
import numpy as np
import re
import nltk
import joblib
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack

# --- 1. CONFIGURATION ---
print("📥 Vérification NLTK...")
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

sia = SentimentIntensityAnalyzer()

# --- 2. FONCTIONS DE NETTOYAGE (Pas de classes complexes !) ---
def clean_text(text):
    # Nettoyage simple
    return re.sub(r"[^a-zA-Z\s]", "", str(text).lower())

def get_vader_score(text):
    return sia.polarity_scores(str(text))['compound']

# --- 3. CHARGEMENT DONNÉES ---
print("📊 Chargement du CSV...")
try:
    df = pd.read_csv("df_merged_clean.csv")
    # Adaptation des noms de colonnes
    if 'reviewText' in df.columns:
        df = df.rename(columns={'reviewText': 'text', 'overall': 'score'})
    
    # Nettoyage des NaN
    df = df.dropna(subset=['text', 'score'])
    
    # On force le type entier pour le score
    df['score'] = df['score'].astype(int)
    
    # ÉCHANTILLONNAGE (Pour que ça tourne vite)
    df = df.sample(n=2000, random_state=42)
    print(f"✅ Données prêtes : {len(df)} lignes.")
    
except Exception as e:
    print(f"❌ Erreur CSV : {e}")
    exit()

# --- 4. PRÉPARATION MANUELLE (On voit tout ce qui se passe) ---
print("⚙️ Transformation des données...")

# A. Nettoyage
print("   - Nettoyage du texte...")
df['clean_text'] = df['text'].apply(clean_text)

# B. TF-IDF
print("   - Vectorisation TF-IDF...")
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
X_tfidf = tfidf.fit_transform(df['clean_text'])
print(f"     -> Forme TF-IDF : {X_tfidf.shape}")

# C. VADER
print("   - Calcul scores VADER...")
# On reshape pour avoir une colonne verticale (n_samples, 1)
X_vader = df['text'].apply(get_vader_score).values.reshape(-1, 1)
print(f"     -> Forme VADER : {X_vader.shape}")

# D. FUSION (On colle TF-IDF et VADER côte à côte)
print("   - Fusion des features...")
# On convertit tout en dense pour éviter les bugs de format
X_final = np.hstack((X_tfidf.toarray(), X_vader))
print(f"     -> Forme FINALE : {X_final.shape}")

y = df['score']

# --- 5. ENTRAÎNEMENT ---
print("🚀 Entraînement du Random Forest...")
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_final, y)
print("✅ Modèle entraîné avec succès !")

# --- 6. TEST IMMÉDIAT ---
print("🧪 Test de prédiction...")
test_text = "This product is terrible and broken"
# On doit refaire les mêmes transfos manuellement pour tester
t_clean = clean_text(test_text)
t_tfidf = tfidf.transform([t_clean]).toarray()
t_vader = np.array([[get_vader_score(test_text)]])
t_final = np.hstack((t_tfidf, t_vader))

pred = clf.predict(t_final)[0]
print(f"   -> Phrase : '{test_text}'")
print(f"   -> Note prédite : {pred}")

# --- 7. SAUVEGARDE DU PACK COMPLET ---
# Astuce Pro : On sauvegarde un dictionnaire avec TOUTES les pièces détachées
# C'est plus robuste que de sauvegarder un Pipeline complexe qui plante.
print("💾 Sauvegarde du kit complet...")
pack = {
    "tfidf": tfidf,
    "model": clf
}
joblib.dump(pack, "model_robust.pkl")
print("🎉 C'est fini ! Utilise 'model_robust.pkl' dans ton API.")