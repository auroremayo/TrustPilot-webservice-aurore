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


class Review(BaseModel):
    text: str = Field(min_length=1, max_length=5000)

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Le texte ne peut pas être vide.")
        return v


class FeedbackPayload(BaseModel):
    timestamp: str = Field(min_length=1)
    feedback: Literal["correct", "incorrect"]

class Base64(BaseModel):
    base64: str  # Données encodées en base64