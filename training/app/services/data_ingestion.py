import os
import subprocess
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict
from ..core.config import BASE_DIR
from common import setup_git

logger = logging.getLogger(__name__)

RAW_DATA_DIR = BASE_DIR / "data" / "raw"

def ingest_and_version_data(
    new_reviews: List[Dict],
    dataset_name: str = "df_merged_clean_sample2.csv",
    auto_push_dvc: bool = True
) -> dict:
    """
    1. Charge le dataset existant
    2. Ajoute et déduplique les nouveaux avis
    3. Exécute dvc add + dvc push + git commit
    """
    csv_file = RAW_DATA_DIR / dataset_name
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if csv_file.exists():
        df_existing = pd.read_csv(csv_file)
        logger.info(f"Dataset existant chargé : {len(df_existing)} lignes.")
    else:
        df_existing = pd.DataFrame()

    df_new = pd.DataFrame(new_reviews)
    
    # Concaténation et déduplication sur le texte
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    initial_count = len(df_combined)
    df_combined.drop_duplicates(subset=["reviewText"], keep="last", inplace=True)
    df_combined.reset_index(drop=True, inplace=True)
    
    added_count = len(df_combined) - len(df_existing)
    logger.info(f"{added_count} nouveaux avis uniques ajoutés ({len(df_combined)} au total).")

    # Sauvegarde sur disque
    df_combined.to_csv(csv_file, index=False)

    # Versionnage DVC
    dvc_info = {}
    if auto_push_dvc:
        setup_git.setup_git_auth()
        try:
            rel_path = f"data/raw/{dataset_name}"
            logger.info(f"[DVC] Indexation du dataset {rel_path}...")
            subprocess.run(["dvc", "add", rel_path], cwd=str(BASE_DIR), check=True, capture_output=True)
            
            logger.info("[DVC] Push vers S3/DagsHub...")
            subprocess.run(["dvc", "push", f"{rel_path}.dvc"], cwd=str(BASE_DIR), check=True, capture_output=True)
            
            # Git commit du pointeur .dvc
            dvc_file = f"{rel_path}.dvc"
            subprocess.run(["git", "add", dvc_file, ".gitignore"], cwd=str(BASE_DIR), check=True)
            subprocess.run(
                ["git", "commit", "-m", f"ingestion {dataset_name} (+{added_count} reviews)"],
                cwd=str(BASE_DIR),
                check=True,
                capture_output=True
            )
            subprocess.run(["git", "push", "origin", "HEAD"], cwd=str(BASE_DIR), check=True)
            dvc_info["status"] = "dvc_pushed_and_committed"
        except Exception as e:
            logger.warning(f"Avertissement lors du versionnage DVC/Git : {e}")
            dvc_info["status"] = f"warning: {e}"

    return {
        "status": "success",
        "total_rows": len(df_combined),
        "added_rows": added_count,
        "dvc_info": dvc_info,
    }