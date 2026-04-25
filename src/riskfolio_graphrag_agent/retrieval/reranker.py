"""Reranker abstractions and implementations for the retrieval layer.

This module provides the reranker interface and concrete implementations used
to improve the final ranking step in ``hybrid_rerank`` mode. It sits inside
the retrieval layer and is consumed by ``HybridRetriever``.

Package fit:
    - ``riskfolio_graphrag_agent.retrieval.retriever`` calls ``Reranker.rerank``
      after graph/context enrichment when a reranker is configured.
    - ``riskfolio_graphrag_agent.config.settings`` exposes ``reranker_backend``
      and ``reranker_model`` to control which implementation is used.

Architecture position:
    dense+sparse candidate generation → graph/context enrichment
        → **optional learned reranking** → top-k truncation

The intended use is::

    from riskfolio_graphrag_agent.retrieval.reranker import CrossEncoderReranker
    from riskfolio_graphrag_agent.retrieval.retriever import HybridRetriever

    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    retriever = HybridRetriever(..., reranker=reranker)
    results = retriever.retrieve("What is Hierarchical Risk Parity?")

Non-obvious design decisions:
    - ``CrossEncoderReranker`` lazy-loads ``sentence_transformers`` so the
      package remains importable even when the library is absent.  A clear
      ``ImportError`` is raised at construction time if the dependency is
      missing, making the failure explicit and easy to diagnose.
    - ``PassthroughReranker`` is a no-op that makes the reranker slot always
      safe to call; callers do not need to special-case ``None``.
    - The ``Reranker`` protocol uses structural typing so third-party or test
      implementations do not need to inherit from anything.

What this module does not do:
    - It does not train or fine-tune reranker models.
    - It does not call external API-based reranker services.
    - It does not redesign graph retrieval candidate generation.
    - It does not manage embedding or vector store backends.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from riskfolio_graphrag_agent.retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)


@runtime_checkable
class Reranker(Protocol):
    """Protocol for reranking a list of retrieval results given a query.

    Any object that implements ``rerank`` with the expected signature satisfies
    this protocol.  Callers may pass instances of ``PassthroughReranker`` or
    ``CrossEncoderReranker`` (or any compatible implementation) wherever a
    ``Reranker`` is expected.
    """

    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        """Re-order and truncate *candidates* to *top_k* items.

        Args:
            query: The original user query string used to score relevance.
            candidates: Pre-scored candidates from upstream retrieval.  The
                implementation may re-score, re-order, or filter them.
            top_k: Maximum number of results to return.

        Returns:
            A list of at most *top_k* ``RetrievalResult`` objects, ranked by
            descending relevance.
        """
        ...


class PassthroughReranker:
    """No-op reranker that preserves the upstream ordering.

    Use this as a safe default or for benchmarking the heuristic pipeline
    without a learned model.  It still respects ``top_k`` truncation.

    Example::

        reranker = PassthroughReranker()
        results = reranker.rerank("HRP query", candidates, top_k=5)
        # results == candidates[:5], unchanged order
    """

    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        """Return the first *top_k* candidates without reordering.

        Args:
            query: User query (unused; present for interface conformance).
            candidates: Pre-ranked candidates from upstream retrieval.
            top_k: Maximum number of results to return.

        Returns:
            The first *top_k* items from *candidates*.
        """
        _ = query
        return candidates[:top_k]


class CrossEncoderReranker:
    """Local cross-encoder reranker backed by ``sentence_transformers``.

    Uses a cross-encoder model to score each (query, passage) pair and
    re-order candidates accordingly.  The default model is
    ``cross-encoder/ms-marco-MiniLM-L-6-v2``, which is small and fast while
    providing strong MRR on MS-MARCO.

    The ``sentence_transformers`` library is loaded lazily at construction
    time.  If it is unavailable an ``ImportError`` is raised immediately with
    a clear installation hint.

    Args:
        model_name: HuggingFace model identifier or local path.
            Defaults to ``"cross-encoder/ms-marco-MiniLM-L-6-v2"``.

    Raises:
        ImportError: If ``sentence_transformers`` is not installed.

    Example::

        reranker = CrossEncoderReranker()
        results = reranker.rerank("What is CVaR?", candidates, top_k=5)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderReranker requires 'sentence_transformers'. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self._model_name = model_name
        logger.info("Loading cross-encoder model: %s", model_name)
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        """Score each candidate with the cross-encoder and return the top *top_k*.

        Args:
            query: User query string.
            candidates: Pre-scored candidates from upstream retrieval.
            top_k: Maximum number of results to return.

        Returns:
            At most *top_k* ``RetrievalResult`` objects re-ordered by
            cross-encoder score (descending).  Each result's ``score``
            attribute is updated to reflect the cross-encoder score.
        """
        if not candidates:
            return []

        pairs = [(query, result.content) for result in candidates]
        scores: list[float] = self._model.predict(pairs).tolist()

        ranked = sorted(zip(scores, candidates), key=lambda item: item[0], reverse=True)
        results: list[RetrievalResult] = []
        for score, result in ranked[:top_k]:
            result.score = round(float(score), 6)
            results.append(result)
        return results


def build_reranker(backend: str, model_name: str) -> Reranker:
    """Construct a reranker from configuration strings.

    This factory is the preferred entry point when building a reranker from
    settings.  It maps ``backend`` string values from ``Settings`` to concrete
    ``Reranker`` implementations.

    Args:
        backend: Reranker backend identifier.  Supported values:
            ``"none"`` – returns a ``PassthroughReranker``;
            ``"cross_encoder"`` – returns a ``CrossEncoderReranker``.
        model_name: Model name or path forwarded to the reranker implementation
            when ``backend`` is ``"cross_encoder"``.

    Returns:
        A ``Reranker`` instance corresponding to the requested backend.

    Raises:
        ValueError: If *backend* is not a recognised value.
        ImportError: If ``backend="cross_encoder"`` and
            ``sentence_transformers`` is not installed.

    Example::

        from riskfolio_graphrag_agent.retrieval.reranker import build_reranker
        reranker = build_reranker("none", "")
        # reranker is a PassthroughReranker
    """
    if backend == "none":
        return PassthroughReranker()
    if backend == "cross_encoder":
        return CrossEncoderReranker(model_name=model_name)
    raise ValueError(f"Unknown reranker backend: {backend!r}. Supported values: 'none', 'cross_encoder'.")
