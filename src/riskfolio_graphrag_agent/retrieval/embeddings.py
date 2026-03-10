"""Embedding provider abstractions for retrieval backends."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import ssl
from dataclasses import dataclass
from typing import Protocol
from urllib import request
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Embedding provider interface for dense retrieval."""

    @property
    def dimension(self) -> int:
        """Return the embedding dimension used by this provider."""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed text list and return one vector per input text."""
        ...


@dataclass
class ProviderResolution:
    """Resolved embedding provider and fallback metadata."""

    provider: EmbeddingProvider
    selected_provider: str
    fallback_provider: str | None = None
    fallback_reason: str | None = None


class HashEmbeddingProvider:
    """Deterministic hash embedding fallback used for offline/dev runs."""

    def __init__(self, dimension: int = 256) -> None:
        self._dimension = max(8, int(dimension))

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embedding(text, dim=self._dimension) for text in texts]


class OpenAIEmbeddingProvider:
    """OpenAI-compatible embeddings provider."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: float,
        dimension: int,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OPENAI_API_KEY is required for OpenAIEmbeddingProvider")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._dimension = max(8, int(dimension))
        self._ssl_context = ssl_context

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = {
            "model": self._model,
            "input": texts,
        }
        endpoint = f"{self._base_url}/embeddings"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with request.urlopen(req, timeout=self._timeout_seconds, context=self._ssl_context) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Embedding HTTP error {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Embedding endpoint unreachable: {exc}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Embedding provider returned non-JSON response") from exc

        data = parsed.get("data", [])
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError("Embedding provider returned invalid data length")

        vectors: list[list[float]] = []
        for row in data:
            embedding = row.get("embedding") if isinstance(row, dict) else None
            if not isinstance(embedding, list):
                raise RuntimeError("Embedding provider returned malformed vector")
            vector = [float(value) for value in embedding]
            vectors.append(vector)

        return vectors


def resolve_embedding_provider(
    *,
    provider_name: str,
    embedding_dim: int,
    openai_api_key: str,
    openai_embedding_model: str,
    openai_base_url: str,
    openai_timeout_seconds: float,
    ssl_context: ssl.SSLContext | None = None,
) -> ProviderResolution:
    """Resolve requested embedding provider with deterministic fallback."""
    normalized = provider_name.strip().lower() if provider_name else "hash"

    if normalized in {"openai", "openai-compatible"}:
        try:
            provider = OpenAIEmbeddingProvider(
                api_key=openai_api_key,
                model=openai_embedding_model,
                base_url=openai_base_url,
                timeout_seconds=openai_timeout_seconds,
                dimension=embedding_dim,
                ssl_context=ssl_context,
            )
            return ProviderResolution(provider=provider, selected_provider="openai")
        except Exception as exc:
            logger.warning("Falling back to hash embeddings because OpenAI provider is unavailable: %s", exc)
            return ProviderResolution(
                provider=HashEmbeddingProvider(dimension=embedding_dim),
                selected_provider="hash",
                fallback_provider="hash",
                fallback_reason=str(exc),
            )

    return ProviderResolution(
        provider=HashEmbeddingProvider(dimension=embedding_dim),
        selected_provider="hash",
    )


def hash_embedding(text: str, dim: int = 256) -> list[float]:
    """Public deterministic hash embedding helper for compatibility."""
    return _hash_embedding(text, dim=dim)


def _hash_embedding(text: str, dim: int = 256) -> list[float]:
    vector = [0.0] * max(8, dim)
    tokens = _query_tokens(text)
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % len(vector)
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _query_tokens(query: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query.lower())
