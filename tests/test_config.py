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
    assert settings.openai_enable_generation is True


def test_settings_override(monkeypatch):
    """Environment variable overrides should take precedence over defaults."""
    monkeypatch.setenv("NEO4J_URI", "bolt://testhost:9999")
    monkeypatch.setenv("LOG_LEVEL", " debug ")
    monkeypatch.setenv("VECTOR_STORE_BACKEND", " CHROMA ")
    monkeypatch.setenv("OPENAI_ENABLE_GENERATION", "false")
    settings = Settings()
    assert settings.neo4j_uri == "bolt://testhost:9999"
    assert settings.log_level == "DEBUG"
    assert settings.vector_store_backend == "chroma"
    assert settings.openai_enable_generation is False
