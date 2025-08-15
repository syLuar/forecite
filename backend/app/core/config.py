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

    # Application Configuration
    api_version: str = Field("v1", env="API_VERSION")
    debug: bool = Field(False, env="DEBUG")

    def model_post_init(self, __context):
        self.llm_config = self._load_config(self.llm_config_path)

    def _load_config(self, path: Path) -> dict:
        """Load configuration from a YAML file."""
        with open(path, "r") as file:
            return yaml.safe_load(file)

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
