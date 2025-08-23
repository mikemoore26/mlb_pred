# config.py
import os

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key")  # Replace in production
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        "sqlite:///mlb_site.db"  # fallback local db
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Optional: default model path
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "models/full_model_ml2.pkl")
