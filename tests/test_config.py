"""Tests for riskfolio_graphrag_agent.config.settings."""

from __future__ import annotations

from riskfolio_graphrag_agent.config.settings import Settings


def test_settings_defaults():
    """Settings should load with sensible defaults when no .env file exists."""
    settings = Settings()
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.log_level == "INFO"
    assert settings.embedding_dim == 1536
    assert settings.vector_store_backend == "chroma"
    assert settings.openai_base_url == "https://api.openai.com/v1"
    assert settings.openai_timeout_seconds == 45.0
    assert settings.openai_retry_attempts == 2
    assert settings.openai_retry_backoff_seconds == 1.5
    assert settings.openai_enable_generation is True


def test_settings_override(monkeypatch):
    """Environment variable overrides should take precedence over defaults."""
    monkeypatch.setenv("NEO4J_URI", "bolt://testhost:9999")
    monkeypatch.setenv("LOG_LEVEL", " debug ")
    monkeypatch.setenv("VECTOR_STORE_BACKEND", " CHROMA ")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("OPENAI_RETRY_ATTEMPTS", "4")
    monkeypatch.setenv("OPENAI_RETRY_BACKOFF_SECONDS", "0.2")
    monkeypatch.setenv("OPENAI_ENABLE_GENERATION", "false")
    monkeypatch.setenv("OPENAI_ENABLE_GRAPH_EXTRACTION", "false")
    settings = Settings()
    assert settings.neo4j_uri == "bolt://testhost:9999"
    assert settings.log_level == "DEBUG"
    assert settings.vector_store_backend == "chroma"
    assert settings.openai_timeout_seconds == 60.0
    assert settings.openai_retry_attempts == 4
    assert settings.openai_retry_backoff_seconds == 0.2
    assert settings.openai_enable_generation is False
    assert settings.openai_enable_graph_extraction is False
