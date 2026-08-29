"""
Script d'entraînement LightGBM avec tracking MLflow.

Pipeline :
  CSV → nettoyage texte → TF-IDF → LightGBM → évaluation → MLflow log → sauvegarde .pkl

Usage :
    python training/train.py --data path/to/df_merged_clean.csv
    python training/train.py --data path/to/df_merged_clean.csv --max-features 20000 --n-estimators 1000
"""

import argparse
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
import subprocess

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.lightgbm
import mlflow.sklearn
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from ..core.config import MODELS_DIR, BASE_DIR
from common.nlp_pipeline import processing_pipeline
from common import setup_git


# ── Configuration du logger ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Ressources NLTK ────────────────────────────────────────────────────────────
def download_nltk_resources():
    for resource in ["punkt_tab", "stopwords", "wordnet"]:
        nltk.download(resource, quiet=True)

# ── Constantes ─────────────────────────────────────────────────────────────────
MODELS_DIR   = MODELS_DIR
MLFLOW_DIR   = BASE_DIR / "mlflow"
EXPERIMENT   = "SentimentAI - LightGBM"
CLASS_NAMES = ["Négatif", "Neutre", "Positif"]


# ── Fonction DVC Push vers S3 ─────────────────────────────────────────────────
def push_models_to_s3(base_dir: Path, run_id: str):
    """
    Indexe les nouveaux modèles avec DVC, les pousse vers S3/DagsHub,
    et pousse le fichier de pointeur models.dvc vers GitHub.
    """
    try:
        logger.info("[DVC] Indexation des modèles (dvc add)...")
        subprocess.run(
            ["dvc", "add", "models"],
            cwd=str(base_dir),
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("[DVC] Téléversement des modèles vers S3/DagsHub (dvc push)...")
        result = subprocess.run(
            ["dvc", "push", "models.dvc"],
            cwd=str(base_dir),
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("[DVC] Modèles poussés sur S3 avec succès : %s", result.stdout.strip())
        # Versionnage Git du pointeur models.dvc (si Git est configuré)
        try:
            logger.info("[Git] Enregistrement de models.dvc sur Git...")
            subprocess.run(["git", "add", "models.dvc", ".gitignore"], cwd=str(base_dir), check=True)
            subprocess.run(
                ["git", "commit", "-m", f"chore(models): retrained model run {run_id}"],
                cwd=str(base_dir),
                check=True,
                capture_output=True
            )
            subprocess.run(["git", "push", "origin", "HEAD"], cwd=str(base_dir), check=True)
            logger.info("[Git] models.dvc poussé sur GitHub.")
        except subprocess.CalledProcessError as git_err:
            logger.warning("[Git] Push Git non effectué (DVC push a tout de même réussi) : %s", git_err)
    except subprocess.CalledProcessError as e:
        logger.error("Échec lors du dvc push : %s", e.stderr or e.stdout)
        raise RuntimeError(f"Erreur dvc push vers S3 : {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT & NETTOYAGE DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_and_clean(data_path: str = "data/raw") -> pd.DataFrame:
    """
    Charge et fusionne dynamiquement les fichiers CSV présents
    dans le dossier de données, élimine les doublons et filtre.
    """
    path = Path(data_path)
    csv_files = list(path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans le répertoire : {data_path}")

    logger.info("Chargement des fichiers : %s", [f.name for f in csv_files])
    
    # Concaténation des fichiers du dossier
    try:
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    except Exception as e:
        logger.error(f"Erreur lors de la lecture des fichiers CSV : {e}")
        raise
    logger.info("Total lignes brutes chargées : %d", len(df))

    # Déduplication globale entre les différents batchs
    if "reviewText" in df.columns:
        df.drop_duplicates(subset=["reviewText"], keep="last", inplace=True)

    # Filtrage langue anglaise
    if "language" in df.columns:
        df = df.loc[df["language"].str.lower() == "en"].copy()

    # Remplacement des valeurs manquantes dans summary
    df["summary"] = df["summary"].fillna("")

    # Filtrage temporel
    if "year_y" in df.columns:
        df = df.loc[df["year_y"] > 2009].copy()

    df = df.reset_index(drop=True)
    logger.info("Total lignes après nettoyage et fusion : %d", len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. PRÉPROCESSING TEXTE
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_text(df: pd.DataFrame) -> pd.Series:
    """Merge reviewText + summary, nettoie, tokenise, supprime stopwords, lemmatise."""
    logger.info("Préprocessing texte...")
    df['texts'] = (df["summary"] + " " + df["reviewText"]).str.lower()

    total = len(df['texts'])
    df['texts']=df['texts'].apply(lambda x: processing_pipeline(x))

    logger.info("Préprocessing terminé.")
    return df['texts']


# ══════════════════════════════════════════════════════════════════════════════
# 3. LABEL MAPPING & RÉÉQUILIBRAGE
# ══════════════════════════════════════════════════════════════════════════════

def map_labels(y: pd.Series) -> pd.Series:
    """1, 2 → 0 (Négatif) | 3 → 1 (Neutre) | 4, 5 → 2 (Positif)"""
    return y.replace({1: 0, 2: 0, 3: 1, 4: 2, 5: 2})


def balance_dataset(X: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Sous-échantillonnage pour équilibrer les classes."""
    min_count = y.value_counts().min()
    idx = []
    for cls in y.unique():
        idx.extend(y[y == cls].sample(n=min_count, random_state=42).index.tolist())
    idx = sorted(idx)
    logger.info(
        "Dataset équilibré : %d exemples par classe (%d total).",
        min_count, len(idx),
    )
    return X.loc[idx], y.loc[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 4. ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════

def train(
    data_path: str = "data/raw",
    max_features: int = 20_000,
    ngram_max: int = 2,
    learning_rate: float = 0.05,
    num_leaves: int = 64,
    n_estimators: int = 1000,
    colsample_bytree: float = 0.8,
    subsample: float = 0.8,
    early_stopping_rounds: int = 50,
    test_size: float = 0.2,
) -> None:
    download_nltk_resources()

    setup_git.setup_git_auth()
  
    # pull dataset depuis DVC
    subprocess.run(["dvc", "pull", data_path], check=True)

    # ── Chargement ──────────────────────────────────────────────────────────
    df = load_and_clean(data_path)

    # ── Préprocessing ────────────────────────────────────────────────────────
    texts = preprocess_text(df)
    labels = map_labels(df["overall"])

    # ── Split train/test ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # ── Rééquilibrage ────────────────────────────────────────────────────────
    X_train, y_train = balance_dataset(X_train, y_train)
    X_test, y_test   = balance_dataset(X_test, y_test)

    # ── TF-IDF ──────────────────────────────────────────────────────────────
    logger.info("Vectorisation TF-IDF (max_features=%d)...", max_features)
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    # ── Paramètres LightGBM ─────────────────────────────────────────────────
    lgbm_params = {
        "objective":       "multiclass",
        "num_class":       3,
        "learning_rate":   learning_rate,
        "num_leaves":      num_leaves,
        "max_depth":       -1,
        "n_estimators":    n_estimators,
        "colsample_bytree": colsample_bytree,
        "subsample":       subsample,
        "random_state":    42,
        "metric":          "multi_logloss",
        "verbose":         -1,
    }

    # ── MLflow run ──────────────────────────────────────────────────────────
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    default_tracking_uri = "http://mlflow:5000" if (os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER")) else "http://localhost:5000"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=f"lgbm_{datetime.now():%Y%m%d_%H%M%S}") as run:
        run_id = run.info.run_id
        artifacts_dir = BASE_DIR / "training" / "artifacts" / run_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Traçabilité du dataset
        csv_files = list(Path(data_path).glob("*.csv"))
        file_names = [f.name for f in csv_files]
        
        # Création de l'objet Dataset MLflow
        mlflow_dataset = mlflow.data.from_pandas(
            df,
            source=str(data_path),
            name=f"raw_combined_{len(csv_files)}_files",
            targets="overall"
        )
        
        # Enregistrement du dataset dans le run
        mlflow.log_input(mlflow_dataset, context="training")
        
        # Log des paramètres du dataset et hyperparamètres des modèles
        mlflow.log_params({
            "data_source":          data_path,
            "source_files":         ", ".join(file_names),
            "source_files_count":   len(csv_files),
            "max_features":         max_features,
            "ngram_range":          f"(1,{ngram_max})",
            "learning_rate":        learning_rate,
            "num_leaves":           num_leaves,
            "n_estimators":         n_estimators,
            "colsample_bytree":     colsample_bytree,
            "subsample":            subsample,
            "early_stopping_rounds": early_stopping_rounds,
            "train_size":           len(X_train),
            "test_size_rows":       len(X_test),
        })

        # ── Entraînement ────────────────────────────────────────────────────
        logger.info("Entraînement LightGBM...")
        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(
            X_train_tfidf, y_train,
            eval_set=[(X_test_tfidf, y_test)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )

        # ── Évaluation ──────────────────────────────────────────────────────
        logger.info("Évaluation...")
        y_pred = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        report = classification_report(
            y_test, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, zero_division=0, output_dict=True
        )
        logger.info("Accuracy : %.4f", acc)
        logger.info("F1-Score (weighted) : %.4f", f1)

        # Log des métriques
        mlflow.log_metrics({
            "accuracy":            round(acc, 4),
            "f1_weighted":         round(f1, 4),
            "f1_negatif":          round(report.get("Négatif", {}).get("f1-score", 0.0), 4),
            "f1_neutre":           round(report.get("Neutre", {}).get("f1-score", 0.0), 4),
            "f1_positif":          round(report.get("Positif", {}).get("f1-score", 0.0), 4),
            "precision_negatif":   round(report.get("Négatif", {}).get("precision", 0.0), 4),
            "precision_neutre":    round(report.get("Neutre", {}).get("precision", 0.0), 4),
            "precision_positif":   round(report.get("Positif", {}).get("precision", 0.0), 4),
            "best_iteration":      model.best_iteration_ or n_estimators,
        })
        cm_path = artifacts_dir / "confusion_matrix.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            confusion_matrix(y_test, y_pred, labels=[0, 1, 2]),
            annot=True, fmt="d", cmap="Blues",
        )
        ax.set_title(f"Matrice de confusion — LightGBM (acc={acc:.3f})")
        ax.set_xlabel("Prédictions")
        ax.set_ylabel("Vraies classes")
        fig.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(cm_path))
        # Top 20 mots importants
        fi_path = artifacts_dir / "feature_importance.png"
        importances = pd.DataFrame({
            "feature":    tfidf.get_feature_names_out(),
            "importance": model.feature_importances_,
        }).nlargest(20, "importance")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importances, x="importance", y="feature", ax=ax)
        ax.set_title("Top 20 mots importants — LightGBM")
        fig.savefig(fi_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(fi_path))

        # Classification report (texte)
        report_path = artifacts_dir / "classification_report.txt"
        report_path.write_text(
            classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, zero_division=0),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(report_path))

        # Log du modèle dans MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="lgbm_model",
            registered_model_name="SentimentAI-LightGBM",
            serialization_format="pickle",
        )

        logger.info("Run MLflow : %s", run_id)

        # ── Sauvegarde des .pkl dans models/ ────────────────────────────────
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "trustpilot_lgbm_model.pkl"
        tfidf_path = MODELS_DIR / "tfidf_vectorizer.pkl"

        joblib.dump(model, model_path)
        joblib.dump(tfidf, tfidf_path)
        logger.info("Modèles sauvegardés dans %s", MODELS_DIR)

        # ── Push automatique vers S3 via DVC ──────────────────────────────
        push_models_to_s3(base_dir=BASE_DIR, run_id=run_id)

        # Log des pkl comme artefacts MLflow également
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(tfidf_path))

        logger.info(
            "\n Entraînement terminé !\n"
            "   Accuracy  : %.4f\n"
            "   F1-Score  : %.4f\n"
            "   Run ID    : %s\n"
            "   MLflow UI : http://localhost:5000",
            acc, f1, run_id,
        )

        return {
            "run_id": run_id,
            "accuracy": round(float(acc), 4),
            "f1_score": round(float(f1), 4),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 5. CLI & MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement LightGBM SentimentAI")
    parser.add_argument("--data",         type=str, default="data/raw", help="Chemin du dossier de données ou du CSV")
    parser.add_argument("--max-features",        type=int,   default=20000)
    parser.add_argument("--ngram-max",           type=int,   default=2)
    parser.add_argument("--learning-rate",       type=float, default=0.05)
    parser.add_argument("--num-leaves",          type=int,   default=64)
    parser.add_argument("--n-estimators",        type=int,   default=1000)
    parser.add_argument("--colsample-bytree",    type=float, default=0.8)
    parser.add_argument("--subsample",           type=float, default=0.8)
    parser.add_argument("--early-stopping",      type=int,   default=50)
    parser.add_argument("--test-size",           type=float, default=0.2)
    args = parser.parse_args()

    train(
        data_path             = args.data,
        max_features          = args.max_features,
        ngram_max             = args.ngram_max,
        learning_rate         = args.learning_rate,
        num_leaves            = args.num_leaves,
        n_estimators          = args.n_estimators,
        colsample_bytree      = args.colsample_bytree,
        subsample             = args.subsample,
        early_stopping_rounds = args.early_stopping,
        test_size             = args.test_size,
    )
