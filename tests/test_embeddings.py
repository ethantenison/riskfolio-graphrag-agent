"""Tests for retrieval embedding provider resolution."""

from __future__ import annotations

import json
from urllib.error import URLError

from riskfolio_graphrag_agent.retrieval.embeddings import (
    HashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    _chunk_embedding_inputs,
    resolve_embedding_provider,
)


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


def test_chunk_embedding_inputs_respects_estimated_token_limit():
    texts = ["x" * 400000, "y" * 400000, "z" * 100]

    batches = _chunk_embedding_inputs(texts, max_batch_texts=128, max_estimated_tokens=200000)

    assert len(batches) == 2
    assert batches[0] == [texts[0], texts[1]]
    assert batches[1] == [texts[2]]


def test_openai_embedding_provider_batches_requests(monkeypatch):
    requested_batch_sizes: list[int] = []

    class _FakeResponse:
        def __init__(self, payload: str) -> None:
            self._payload = payload.encode("utf-8")

        def read(self) -> bytes:
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            _ = exc_type, exc, tb
            return False

    def _fake_urlopen(req, timeout=None, context=None):
        _ = timeout, context
        body = json.loads(req.data.decode("utf-8"))
        inputs = body["input"]
        requested_batch_sizes.append(len(inputs))
        payload = {
            "data": [{"embedding": [0.0, 1.0, 2.0]} for _ in inputs],
        }
        return _FakeResponse(json.dumps(payload))

    monkeypatch.setattr("riskfolio_graphrag_agent.retrieval.embeddings.request.urlopen", _fake_urlopen)

    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        model="text-embedding-3-small",
        base_url="https://api.openai.com/v1",
        timeout_seconds=5.0,
        dimension=3,
    )
    provider._MAX_BATCH_TEXTS = 2
    provider._MAX_ESTIMATED_TOKENS = 1000000

    vectors = provider.embed_texts(["alpha", "beta", "gamma", "delta", "epsilon"])

    assert requested_batch_sizes == [2, 2, 1]
    assert len(vectors) == 5


def test_openai_embedding_provider_retries_transient_network_errors(monkeypatch):
    calls = {"count": 0}

    class _FakeResponse:
        def __init__(self, payload: str) -> None:
            self._payload = payload.encode("utf-8")

        def read(self) -> bytes:
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            _ = exc_type, exc, tb
            return False

    def _fake_urlopen(req, timeout=None, context=None):
        _ = req, timeout, context
        calls["count"] += 1
        if calls["count"] == 1:
            raise URLError("temporary upstream timeout")
        payload = {"data": [{"embedding": [0.0, 1.0, 2.0]}]}
        return _FakeResponse(json.dumps(payload))

    monkeypatch.setattr("riskfolio_graphrag_agent.retrieval.embeddings.request.urlopen", _fake_urlopen)

    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        model="text-embedding-3-small",
        base_url="https://api.openai.com/v1",
        timeout_seconds=5.0,
        dimension=3,
        retry_attempts=1,
        retry_backoff_seconds=0.0,
    )

    vectors = provider.embed_texts(["alpha"])

    assert calls["count"] == 2
    assert len(vectors) == 1
