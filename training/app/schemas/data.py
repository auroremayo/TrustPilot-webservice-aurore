from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime

## Schema pour les avis bruts provenant de Trustpilot, pour l'importation dans la base de données. 
## Les champs sont définis avec des contraintes et des descriptions pour assurer la validité des données.

class RawReviewItem(BaseModel):
    reviewText: str = Field(..., min_length=3, description="Texte de l'avis")
    summary: Optional[str] = Field(default="", description="Titre ou résumé de l'avis")
    overall: int = Field(..., ge=1, le=5, description="Note de 1 à 5 étoiles")
    asin: Optional[str] = Field(default="TRUSTPILOT", description="Identifiant produit ou marque")
    language: Optional[str] = Field(default="en", description="Code de langue de l'avis")
    year_y: Optional[float] = Field(default_factory=lambda: float(datetime.now().year))

    @field_validator("reviewText")
    @classmethod
    def check_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Le texte de l'avis ne peut pas être composé uniquement d'espaces.")
        return v.strip()


    @field_validator("language", mode="before")
    @classmethod
    def validate_and_normalize_language(cls, v: str) -> str:
        if isinstance(v, str):
            v_clean = v.strip().lower()
            if v_clean in ["en", "english"]:
                return "en"
        raise ValueError(f"Langue non supportée : '{v}'. Seuls les avis en anglais ('en') sont acceptés.")