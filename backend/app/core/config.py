"""
Configuration management for the Legal Research Assistant backend.
Reads environment variables and provides typed configuration objects.
"""

import os
from typing import Optional
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
import yaml


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Model Configuration
    llm_config_path: str = Field("./llm_config.yaml", env="LLM_CONFIG_PATH")
    llm_config: Optional[dict] = None

    # API Keys
    google_api_key: Optional[str] = Field(..., env="GOOGLE_API_KEY")
    openai_api_key: Optional[str] = Field(..., env="OPENAI_API_KEY")

    # LangSmith Configuration
    langsmith_tracing: bool = Field(False, env="LANGSMITH_TRACING")
    langsmith_api_key: Optional[str] = Field(None, env="LANGSMITH_API_KEY")
    langsmith_endpoint: Optional[str] = Field(None, env="LANGSMITH_ENDPOINT")
    langsmith_project: Optional[str] = Field(None, env="LANGSMITH_PROJECT")

    # Neo4j Configuration
    neo4j_uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    neo4j_username: str = Field("neo4j", env="NEO4J_USERNAME")
    neo4j_password: str = Field(..., env="NEO4J_PASSWORD")
    neo4j_database: str = Field("neo4j", env="NEO4J_DATABASE")

    # Vector Index Configuration
    vector_index_name: str = Field("chunk_embeddings", env="VECTOR_INDEX_NAME")

    # Document Processing Configuration
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")

    # Database Configuration
    database_url: Optional[str] = Field(None, env="DATABASE_URL")
    postgres_user: str = Field("postgres", env="POSTGRES_USER")
    postgres_password: str = Field("", env="POSTGRES_PASSWORD")
    postgres_host: str = Field("localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(5432, env="POSTGRES_PORT")
    postgres_db: str = Field("legal_assistant", env="POSTGRES_DB")

    # Application Configuration
    api_version: str = Field("v1", env="API_VERSION")
    debug: bool = Field(False, env="DEBUG")

    environment: str = Field("development", env="ENVIRONMENT")

    def model_post_init(self, __context):
        self.llm_config = self._load_config(self.llm_config_path)

    def get_database_url(self) -> str:
        """Get the database URL based on environment and configuration."""
        # If DATABASE_URL is explicitly set, use it
        if self.database_url:
            return self.database_url

        # Use PostgreSQL for production environments, SQLite for development/testing
        if self.environment.lower() in ["production", "prod", "staging", "stage"]:
            return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        else:
            return "sqlite:///./legal_assistant.db"

    def _load_config(self, path: Path) -> dict:
        """Load configuration from a YAML file."""
        with open(path, "r") as file:
            return yaml.safe_load(file)

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
