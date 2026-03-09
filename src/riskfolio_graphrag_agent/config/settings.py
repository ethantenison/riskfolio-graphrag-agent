"""Application settings loaded from environment variables / .env file.

All settings can be overridden via environment variables.  A ``.env`` file in
the working directory is loaded automatically (via ``python-dotenv``).

Example::

    from riskfolio_graphrag_agent.config.settings import Settings
    cfg = Settings()
    print(cfg.neo4j_uri)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import Literal


class Settings(BaseSettings):
    """Project-wide settings resolved from environment variables.

    Attributes:
        neo4j_uri: Bolt URI for the Neo4j instance.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.
        openai_api_key: API key for the OpenAI-compatible LLM endpoint.
        openai_model: Model name to use for generation.
        openai_base_url: Base URL for an OpenAI-compatible API.
        openai_timeout_seconds: Timeout for LLM HTTP requests in seconds.
        openai_enable_generation: Enables model-backed answer generation.
        embedding_model: Model name to use for text embeddings.
        embedding_dim: Dimensionality of embedding vectors.
        vector_store_backend: Which vector store to use ("chroma" | "qdrant").
        chroma_persist_dir: Local directory for ChromaDB persistence.
        log_level: Python logging level string (e.g. "INFO").
        riskfolio_source_dir: Local path to Riskfolio-Lib source / docs.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_timeout_seconds: float = 30.0
    openai_enable_generation: bool = True

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Vector store
    vector_store_backend: Literal["chroma", "qdrant"] = "chroma"
    chroma_persist_dir: str = ".chroma"

    # Logging
    log_level: str = "INFO"

    # Ingestion
    riskfolio_source_dir: str = "./data/riskfolio-lib"

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> str:
        if value is None:
            return "INFO"
        text = str(value).strip()
        return text.upper() if text else "INFO"

    @field_validator("vector_store_backend", mode="before")
    @classmethod
    def _normalize_vector_store_backend(cls, value: object) -> str:
        if value is None:
            return "chroma"
        text = str(value).strip().lower()
        return text if text else "chroma"
