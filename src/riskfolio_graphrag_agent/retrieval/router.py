"""Adaptive query router for per-question retrieval tool selection."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

from riskfolio_graphrag_agent.retrieval.embeddings import EmbeddingProvider, HashEmbeddingProvider
from riskfolio_graphrag_agent.retrieval.retriever import RetrievalMode

logger = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    """Routing result for a single query."""

    mode: RetrievalMode
    confidence: float
    reason: str


class QueryToolRouter:
    """Routes each query to the most suitable retrieval mode.

    The router combines multiple signals:
    - Rule hints for high-precision patterns.
    - Embedding similarity against intent prototypes.
    - Lexical intent overlap with curated mode vocabularies.
    - Query structure features (code-like, relationship-like, compare-like).
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        min_confidence: float = 0.2,
        ambiguity_margin: float = 0.08,
    ) -> None:
        self._embedding_provider = embedding_provider or HashEmbeddingProvider(dimension=256)
        self._min_confidence = max(0.0, min(1.0, min_confidence))
        self._ambiguity_margin = max(0.0, min(1.0, ambiguity_margin))
        self._prototype_text: dict[RetrievalMode, tuple[str, ...]] = {
            "dense": (
                "explain concept definition and overview",
                "summarize method and rationale",
                "what does this term mean",
                "describe intuition behind an optimization method",
            ),
            "sparse": (
                "exact function name parameter signature",
                "file path section line range",
                "specific keyword search in code",
                "riskfolio api parameter default value signature",
                "python function class method module",
                "locate implementation details by symbol name",
            ),
            "graph": (
                "relationship between entities dependencies",
                "which components mention concept",
                "connected nodes and neighbourhood",
                "how one concept affects another in the graph",
            ),
            "hybrid_rerank": (
                "compare methods and evidence from docs and code",
                "multi-hop question with entities and details",
                "retrieve broad context and rerank",
                "synthesize evidence across multiple sources",
            ),
        }
        self._mode_lexicon: dict[RetrievalMode, tuple[str, ...]] = {
            "dense": (
                "define",
                "definition",
                "overview",
                "explain",
                "intuition",
                "concept",
                "meaning",
            ),
            "sparse": (
                "line",
                "path",
                "file",
                "signature",
                "parameter",
                "class",
                "method",
                "function",
                "module",
                "regex",
                "keyword",
            ),
            "graph": (
                "relationship",
                "related",
                "connected",
                "dependency",
                "graph",
                "neighbor",
                "entity",
            ),
            "hybrid_rerank": (
                "compare",
                "versus",
                "tradeoff",
                "multi-hop",
                "end-to-end",
                "across",
                "combine",
            ),
        }
        self._prototype_vectors: dict[RetrievalMode, list[float]] | None = None

    def decide(self, query: str) -> RouteDecision:
        text = query.strip()
        if not text:
            return RouteDecision(mode="hybrid_rerank", confidence=0.0, reason="empty_query_fallback")

        if len(text.split()) <= 1:
            return RouteDecision(mode="hybrid_rerank", confidence=0.0, reason="insufficient_query_context")

        rule_mode, rule_score, rule_reason = self._rule_signal(text)
        embedding_scores = self._embedding_signal(text)
        lexical_scores = self._lexical_intent_signal(text)
        structure_scores = self._structure_signal(text)

        combined: dict[RetrievalMode, float] = {}
        for mode in self._prototype_text:
            embedding_score = embedding_scores.get(mode, 0.0)
            lexical_score = lexical_scores.get(mode, 0.0)
            structure_score = structure_scores.get(mode, 0.0)
            rule_bonus = rule_score if mode == rule_mode else 0.0
            combined[mode] = (0.45 * embedding_score) + (0.25 * lexical_score) + (0.2 * structure_score) + (0.1 * rule_bonus)

        ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
        selected_mode, raw_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = max(0.0, raw_score - second_score)
        confidence = max(0.0, min(1.0, raw_score * (0.75 + min(0.25, margin))))

        if margin < self._ambiguity_margin:
            return RouteDecision(
                mode="hybrid_rerank",
                confidence=confidence,
                reason=f"ambiguous_intent margin={margin:.3f}",
            )

        if confidence < self._min_confidence:
            return RouteDecision(mode="hybrid_rerank", confidence=confidence, reason="low_confidence_fallback")

        reason = (
            f"embed={embedding_scores.get(selected_mode, 0.0):.3f}; "
            f"lex={lexical_scores.get(selected_mode, 0.0):.3f}; "
            f"struct={structure_scores.get(selected_mode, 0.0):.3f}; margin={margin:.3f}"
        )
        if rule_mode == selected_mode and rule_reason:
            reason = f"{rule_reason}; {reason}"
        return RouteDecision(mode=selected_mode, confidence=confidence, reason=reason)

    def _build_prototype_vectors(self) -> dict[RetrievalMode, list[float]]:
        vectors: dict[RetrievalMode, list[float]] = {}
        for mode, exemplars in self._prototype_text.items():
            embeddings = self._embedding_provider.embed_texts(list(exemplars))
            vectors[mode] = _mean_vector(embeddings)
        return vectors

    def _ensure_prototype_vectors(self) -> dict[RetrievalMode, list[float]]:
        if self._prototype_vectors is not None:
            return self._prototype_vectors
        try:
            self._prototype_vectors = self._build_prototype_vectors()
        except Exception as exc:
            logger.warning("Router embedding initialization failed; falling back to hash embeddings: %s", exc)
            self._embedding_provider = HashEmbeddingProvider(dimension=256)
            self._prototype_vectors = self._build_prototype_vectors()
        return self._prototype_vectors

    def _embedding_signal(self, query: str) -> dict[RetrievalMode, float]:
        prototype_vectors = self._ensure_prototype_vectors()
        try:
            query_vector = self._embedding_provider.embed_texts([query])[0]
        except Exception as exc:
            logger.warning("Router query embedding failed; returning neutral embedding signal: %s", exc)
            return {mode: 0.0 for mode in self._prototype_text}
        scores: dict[RetrievalMode, float] = {}
        for mode, prototype_vector in prototype_vectors.items():
            scores[mode] = _cosine_similarity(query_vector, prototype_vector)
        return scores

    def _lexical_intent_signal(self, query: str) -> dict[RetrievalMode, float]:
        tokens = set(re.findall(r"[a-z][a-z0-9_-]{2,}", query.lower()))
        if not tokens:
            return {mode: 0.0 for mode in self._prototype_text}

        scores: dict[RetrievalMode, float] = {}
        for mode, keywords in self._mode_lexicon.items():
            overlap = sum(1 for token in tokens if token in keywords)
            scores[mode] = min(1.0, overlap / max(3.0, len(keywords) * 0.35))
        return scores

    def _structure_signal(self, query: str) -> dict[RetrievalMode, float]:
        lowered = query.lower()
        scores = {mode: 0.0 for mode in self._prototype_text}

        if re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)", query) or re.search(r"\.(py|ipynb|rst)\b", lowered):
            scores["sparse"] += 0.9
        if any(fragment in lowered for fragment in ("/", "::", "line ", "lines ", "signature", "parameter")):
            scores["sparse"] += 0.6
        if any(fragment in lowered for fragment in ("relationship", "connected", "dependency", "graph", "entity")):
            scores["graph"] += 0.85
        if any(fragment in lowered for fragment in ("compare", "versus", "vs", "trade-off", "tradeoff", "across")):
            scores["hybrid_rerank"] += 0.9
        if any(fragment in lowered for fragment in ("what is", "define", "overview", "meaning")):
            scores["dense"] += 0.8

        return {mode: max(0.0, min(1.0, value)) for mode, value in scores.items()}

    def _rule_signal(self, query: str) -> tuple[RetrievalMode, float, str]:
        lowered = query.lower()
        graph_patterns = (
            r"\b(relationship|related|connected|dependenc|neighbou?r|graph|mention)\w*\b",
            r"\bbetween\b.*\band\b",
            r"\b(which|what uses|how does .+ relate|dependencies of|connected to)\b",
        )
        sparse_patterns = (
            r"\b(line|lines|path|file|section|exact|regex|keyword|parameter|signature)\b",
            r"\b(test_|def\s+|class\s+)\b",
            r"rp\.",
            r"rf\.",
            r"\b(rp\.Portfolio|Portfolio\(\)|optimize|to_pandas|plot_|show_)\b",
            r"\b(docstring|__init__|args:|returns:|attributes:)\b",
            r"\b[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)",
        )
        dense_patterns = (r"\b(define|definition|what is|meaning|overview|explain)\b",)
        hybrid_patterns = (
            r"\b(compare|trade[- ]?off|versus|vs\.?|multi[- ]?hop|end[- ]?to[- ]?end)\b",
            r"\b(difference between|pros and cons|better than|worse than)\b",
        )

        if any(re.search(pattern, lowered) for pattern in graph_patterns):
            return "graph", 1.0, "rule_graph_intent"
        if any(re.search(pattern, lowered) for pattern in sparse_patterns):
            return "sparse", 0.95, "rule_sparse_intent"
        if any(re.search(pattern, lowered) for pattern in hybrid_patterns):
            return "hybrid_rerank", 0.9, "rule_hybrid_intent"
        if any(re.search(pattern, lowered) for pattern in dense_patterns):
            return "dense", 0.85, "rule_dense_intent"
        # Default: bias toward sparse for the code-heavy Riskfolio corpus.
        return "sparse", 0.55, "rule_default"


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
