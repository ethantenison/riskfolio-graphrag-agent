"""Tests for retrieval embedding provider resolution."""

from __future__ import annotations

from riskfolio_graphrag_agent.retrieval.embeddings import HashEmbeddingProvider, resolve_embedding_provider


def test_hash_embedding_provider_output_count_matches_input():
    provider = HashEmbeddingProvider(dimension=32)
    vectors = provider.embed_texts(["risk parity", "cvar"])
    assert len(vectors) == 2
    assert all(len(vector) == 32 for vector in vectors)


def test_openai_resolution_falls_back_to_hash_without_api_key():
    resolution = resolve_embedding_provider(
        provider_name="openai",
        embedding_dim=64,
        openai_api_key="",
        openai_embedding_model="text-embedding-3-small",
        openai_base_url="https://api.openai.com/v1",
        openai_timeout_seconds=5.0,
    )

    assert resolution.selected_provider == "hash"
    assert resolution.fallback_provider == "hash"
    assert resolution.fallback_reason is not None
