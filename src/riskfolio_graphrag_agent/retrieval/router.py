"""Adaptive query router for per-question retrieval tool selection."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from riskfolio_graphrag_agent.retrieval.embeddings import EmbeddingProvider, HashEmbeddingProvider
from riskfolio_graphrag_agent.retrieval.retriever import RetrievalMode


@dataclass
class RouteDecision:
    """Routing result for a single query."""

    mode: RetrievalMode
    confidence: float
    reason: str


class QueryToolRouter:
    """Routes each query to the most suitable retrieval mode.

    The router combines two signals:
    - Rule-based intent detection for high-precision routing hints.
    - Lightweight embedding similarity against intent prototypes.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        min_confidence: float = 0.2,
    ) -> None:
        self._embedding_provider = embedding_provider or HashEmbeddingProvider(dimension=128)
        self._min_confidence = max(0.0, min(1.0, min_confidence))
        self._prototype_text: dict[RetrievalMode, tuple[str, ...]] = {
            "dense": (
                "explain concept definition and overview",
                "summarize method and rationale",
                "what does this term mean",
            ),
            "sparse": (
                "exact function name parameter signature",
                "file path section line range",
                "specific keyword search in code",
            ),
            "graph": (
                "relationship between entities dependencies",
                "which components mention concept",
                "connected nodes and neighbourhood",
            ),
            "hybrid_rerank": (
                "compare methods and evidence from docs and code",
                "multi-hop question with entities and details",
                "retrieve broad context and rerank",
            ),
        }
        self._prototype_vectors = self._build_prototype_vectors()

    def decide(self, query: str) -> RouteDecision:
        text = query.strip()
        if not text:
            return RouteDecision(mode="hybrid_rerank", confidence=0.0, reason="empty_query_fallback")

        rule_mode, rule_score, rule_reason = self._rule_signal(text)
        embedding_scores = self._embedding_signal(text)

        combined: dict[RetrievalMode, float] = {}
        for mode, embedding_score in embedding_scores.items():
            combined[mode] = (0.65 * embedding_score) + (0.35 * (rule_score if mode == rule_mode else 0.0))

        ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
        selected_mode, score = ranked[0]
        if score < self._min_confidence:
            return RouteDecision(mode="hybrid_rerank", confidence=score, reason="low_confidence_fallback")

        reason = f"embedding={score:.3f}"
        if rule_mode == selected_mode and rule_reason:
            reason = f"{rule_reason}; {reason}"
        return RouteDecision(mode=selected_mode, confidence=score, reason=reason)

    def _build_prototype_vectors(self) -> dict[RetrievalMode, list[float]]:
        vectors: dict[RetrievalMode, list[float]] = {}
        for mode, exemplars in self._prototype_text.items():
            embeddings = self._embedding_provider.embed_texts(list(exemplars))
            vectors[mode] = _mean_vector(embeddings)
        return vectors

    def _embedding_signal(self, query: str) -> dict[RetrievalMode, float]:
        query_vector = self._embedding_provider.embed_texts([query])[0]
        scores: dict[RetrievalMode, float] = {}
        for mode, prototype_vector in self._prototype_vectors.items():
            scores[mode] = _cosine_similarity(query_vector, prototype_vector)
        return scores

    def _rule_signal(self, query: str) -> tuple[RetrievalMode, float, str]:
        lowered = query.lower()
        graph_patterns = (
            r"\b(relationship|related|connected|dependenc|neighbou?r|graph|mention)\w*\b",
            r"\bbetween\b.*\band\b",
        )
        sparse_patterns = (
            r"\b(line|lines|path|file|section|exact|regex|keyword|parameter|signature)\b",
            r"\b(test_|def\s+|class\s+)\b",
        )
        dense_patterns = (r"\b(define|definition|what is|meaning|overview|explain)\b",)
        hybrid_patterns = (r"\b(compare|trade[- ]?off|versus|vs\.?|multi[- ]?hop|end[- ]?to[- ]?end)\b",)

        if any(re.search(pattern, lowered) for pattern in graph_patterns):
            return "graph", 1.0, "rule_graph_intent"
        if any(re.search(pattern, lowered) for pattern in sparse_patterns):
            return "sparse", 0.95, "rule_sparse_intent"
        if any(re.search(pattern, lowered) for pattern in hybrid_patterns):
            return "hybrid_rerank", 0.9, "rule_hybrid_intent"
        if any(re.search(pattern, lowered) for pattern in dense_patterns):
            return "dense", 0.85, "rule_dense_intent"
        return "hybrid_rerank", 0.6, "rule_default"


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    if width == 0:
        return []

    totals = [0.0] * width
    for vector in vectors:
        for index in range(min(width, len(vector))):
            totals[index] += float(vector[index])

    return [value / len(vectors) for value in totals]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0

    width = min(len(left), len(right))
    dot = sum(float(left[index]) * float(right[index]) for index in range(width))
    left_norm = math.sqrt(sum(float(left[index]) ** 2 for index in range(width)))
    right_norm = math.sqrt(sum(float(right[index]) ** 2 for index in range(width)))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    cosine = dot / (left_norm * right_norm)
    return max(0.0, min(1.0, cosine))
