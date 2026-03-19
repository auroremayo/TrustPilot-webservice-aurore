from typing import Literal
from pydantic import BaseModel, Field, field_validator


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=100)
    role: Literal["user", "admin"] = "user"

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        if not all(c.isalnum() or c in "_-" for c in v):
            raise ValueError("Le nom d'utilisateur ne peut contenir que lettres, chiffres, _ et -.")
        return v.lower()


class UserLogin(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=100)



class FeedbackPayload(BaseModel):
    timestamp: str = Field(min_length=1)
    feedback: Literal["correct", "incorrect"]

class Base64(BaseModel):
    base64: str  # Données encodées en base64


class TrainRequest(BaseModel):
    data_path: str = Field(description="Chemin vers le fichier CSV de données d'entraînement")
    dataset_name: str = Field(description="Nom du dataset pour l'entraînement")
    max_features: int = Field(default=20000, ge=1000, le=100000, description="Nombre maximum de features TF-IDF")
    ngram_max: int = Field(default=2, ge=1, le=3, description="N-gram maximum pour TF-IDF")
    learning_rate: float = Field(default=0.05, gt=0, le=1, description="Taux d'apprentissage LightGBM")
    num_leaves: int = Field(default=64, ge=10, le=1000, description="Nombre de feuilles LightGBM")
    n_estimators: int = Field(default=1000, ge=10, le=10000, description="Nombre d'estimateurs")
    colsample_bytree: float = Field(default=0.8, gt=0, le=1, description="Fraction de colonnes par arbre")
    subsample: float = Field(default=0.8, gt=0, le=1, description="Fraction d'échantillons par arbre")
    early_stopping_rounds: int = Field(default=50, ge=10, le=500, description="Rounds d'arrêt précoce")
    test_size: float = Field(default=0.2, gt=0, lt=1, description="Taille du jeu de test")