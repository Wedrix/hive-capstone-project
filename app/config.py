"""
Configuration settings for the application
"""

from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    APP_NAME: str = "Hive Weather Inference API"

    # CORS Settings
    # SECURITY: In production, replace ["*"] with specific allowed origins
    # Example: ["https://yourdomain.com", "https://www.yourdomain.com"]
    CORS_ORIGINS: List[str] = ["*"]  # WARNING: Allows all origins - restrict in production

    # Model Settings
    MODEL_PATH: str = "models/"
    MODEL_NAME: str = "model.joblib"

    # Team Members / Model Mappings
    TEAM_MEMBERS: List[str] = [
        "KAMILE SEIDU",
        "JAMES WEDAM ANEWENAH",
        "ANTHONY SEDJOAH",
        "Nana Duah",
        "FRANKLIN HOGBA",
        "MASHUD BAWA ABDULAI",
        "Eric Okyere",
        "Alexander Adade",
        "PELEG TEYE DARKEY",
        "Silas Yakalim",
    ]

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
