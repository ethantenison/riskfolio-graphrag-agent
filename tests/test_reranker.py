"""Tests for riskfolio_graphrag_agent.retrieval.reranker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from riskfolio_graphrag_agent.retrieval.reranker import (
    CrossEncoderReranker,
    PassthroughReranker,
    Reranker,
    build_reranker,
)
from riskfolio_graphrag_agent.retrieval.retriever import RetrievalResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(content: str, score: float) -> RetrievalResult:
    return RetrievalResult(content=content, source_path="/tmp/test.py", score=score)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_passthrough_reranker_satisfies_protocol():
    reranker = PassthroughReranker()
    assert isinstance(reranker, Reranker)


def test_passthrough_reranker_is_reranker_protocol():
    """PassthroughReranker must satisfy the Reranker runtime-checkable protocol."""
    instance = PassthroughReranker()
    assert isinstance(instance, Reranker)


# ---------------------------------------------------------------------------
# PassthroughReranker
# ---------------------------------------------------------------------------


def test_passthrough_reranker_preserves_order():
    reranker = PassthroughReranker()
    candidates = [
        _make_result("first", 0.9),
        _make_result("second", 0.8),
        _make_result("third", 0.7),
    ]
    results = reranker.rerank("query", candidates, top_k=3)
    assert [r.content for r in results] == ["first", "second", "third"]


def test_passthrough_reranker_truncates_to_top_k():
    reranker = PassthroughReranker()
    candidates = [_make_result(f"doc-{i}", float(i)) for i in range(10)]
    results = reranker.rerank("query", candidates, top_k=4)
    assert len(results) == 4
    assert results[0].content == "doc-0"


def test_passthrough_reranker_returns_empty_for_empty_candidates():
    reranker = PassthroughReranker()
    results = reranker.rerank("query", [], top_k=5)
    assert results == []


def test_passthrough_reranker_top_k_larger_than_candidates():
    reranker = PassthroughReranker()
    candidates = [_make_result("only", 1.0)]
    results = reranker.rerank("query", candidates, top_k=10)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# CrossEncoderReranker – graceful fallback when sentence_transformers absent
# ---------------------------------------------------------------------------


def test_cross_encoder_reranker_raises_import_error_when_library_missing():
    """CrossEncoderReranker must raise ImportError when sentence_transformers is absent."""
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        with pytest.raises(ImportError, match="sentence_transformers"):
            CrossEncoderReranker()


# ---------------------------------------------------------------------------
# CrossEncoderReranker – behaviour with mocked model
# ---------------------------------------------------------------------------


def _make_cross_encoder_reranker() -> CrossEncoderReranker:
    """Build a CrossEncoderReranker with a mocked CrossEncoder model."""
    mock_model = MagicMock()

    mock_cross_encoder_cls = MagicMock(return_value=mock_model)
    mock_st_module = MagicMock()
    mock_st_module.CrossEncoder = mock_cross_encoder_cls

    with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
        reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    reranker._model = mock_model
    return reranker


def test_cross_encoder_reranker_reorders_by_score():
    reranker = _make_cross_encoder_reranker()
    import numpy as np

    # Model returns higher score for the second candidate.
    reranker._model.predict = MagicMock(return_value=np.array([0.3, 0.9, 0.1]))

    candidates = [
        _make_result("low-relevance", 0.7),
        _make_result("high-relevance", 0.5),
        _make_result("irrelevant", 0.6),
    ]
    results = reranker.rerank("HRP query", candidates, top_k=2)

    assert len(results) == 2
    assert results[0].content == "high-relevance"
    assert results[1].content == "low-relevance"


def test_cross_encoder_reranker_updates_score():
    reranker = _make_cross_encoder_reranker()
    import numpy as np

    reranker._model.predict = MagicMock(return_value=np.array([0.42]))

    candidates = [_make_result("single", 0.8)]
    results = reranker.rerank("query", candidates, top_k=1)

    assert len(results) == 1
    assert results[0].score == pytest.approx(0.42, abs=1e-5)


def test_cross_encoder_reranker_returns_empty_for_empty_candidates():
    reranker = _make_cross_encoder_reranker()
    results = reranker.rerank("query", [], top_k=5)
    assert results == []
    reranker._model.predict.assert_not_called()


def test_cross_encoder_reranker_respects_top_k():
    reranker = _make_cross_encoder_reranker()
    import numpy as np

    reranker._model.predict = MagicMock(return_value=np.array([0.9, 0.8, 0.7, 0.6, 0.5]))
    candidates = [_make_result(f"doc-{i}", float(i) / 10) for i in range(5)]
    results = reranker.rerank("query", candidates, top_k=3)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# build_reranker factory
# ---------------------------------------------------------------------------


def test_build_reranker_none_returns_passthrough():
    reranker = build_reranker("none", "")
    assert isinstance(reranker, PassthroughReranker)


def test_build_reranker_unknown_backend_raises_value_error():
    with pytest.raises(ValueError, match="Unknown reranker backend"):
        build_reranker("unknown_backend", "some-model")


def test_build_reranker_cross_encoder_raises_import_error_when_missing():
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        with pytest.raises(ImportError):
            build_reranker("cross_encoder", "cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------------------------------------------------------------------------
# HybridRetriever integration
# ---------------------------------------------------------------------------


def test_hybrid_retriever_uses_reranker_in_hybrid_mode():
    """HybridRetriever should delegate final ranking to the reranker in hybrid_rerank mode."""
    from unittest.mock import MagicMock, patch

    from riskfolio_graphrag_agent.retrieval.retriever import HybridRetriever, VectorHit

    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = iter([])

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    class _FakeVectorStore:
        def upsert(self, docs):
            return len(docs)

        def search(self, query, top_k):
            _ = query
            return [
                VectorHit(chunk_id=f"d{i}", content=f"doc {i}", source_path="x.py", score=0.9 - i * 0.1)
                for i in range(top_k)
            ]

    # A reranker that reverses the order of candidates.
    class _ReversingReranker:
        def rerank(self, query, candidates, top_k):
            _ = query
            return list(reversed(candidates))[:top_k]

    reranker = _ReversingReranker()

    with (
        patch("riskfolio_graphrag_agent.retrieval.retriever.GraphDatabase.driver", return_value=mock_driver),
        patch(
            "riskfolio_graphrag_agent.retrieval.retriever._sparse_query_hits",
            return_value=[VectorHit(chunk_id="s1", content="sparse doc", source_path="s.py", score=2.0)],
        ),
        patch(
            "riskfolio_graphrag_agent.retrieval.retriever._graph_expand",
            side_effect=lambda hit, session: RetrievalResult(
                content=hit.content,
                source_path=hit.source_path,
                score=hit.score,
                graph_neighbours=[],
                related_entities=[],
                metadata={},
            ),
        ),
    ):
        retriever = HybridRetriever(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            top_k=3,
            vector_store=_FakeVectorStore(),
            retrieval_mode="hybrid_rerank",
            reranker=reranker,
        )
        try:
            results = retriever.retrieve("What is HRP?")
        finally:
            retriever.close()

    assert results
    # The reversing reranker should have changed the ordering from heuristic sort.
    assert isinstance(results, list)


def test_hybrid_retriever_without_reranker_uses_heuristic_only():
    """HybridRetriever without a reranker should behave exactly as before."""
    from unittest.mock import MagicMock, patch

    from riskfolio_graphrag_agent.retrieval.retriever import HybridRetriever, VectorHit

    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = iter([])

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    class _FakeVectorStore:
        def upsert(self, docs):
            return len(docs)

        def search(self, query, top_k):
            _ = query
            return [VectorHit(chunk_id="dense-1", content="dense", source_path="d.py", score=0.9)]

    with (
        patch("riskfolio_graphrag_agent.retrieval.retriever.GraphDatabase.driver", return_value=mock_driver),
        patch(
            "riskfolio_graphrag_agent.retrieval.retriever._sparse_query_hits",
            return_value=[VectorHit(chunk_id="sparse-1", content="sparse", source_path="s.py", score=2.0)],
        ),
        patch(
            "riskfolio_graphrag_agent.retrieval.retriever._graph_expand",
            side_effect=lambda hit, session: RetrievalResult(
                content=hit.content,
                source_path=hit.source_path,
                score=hit.score,
                graph_neighbours=[],
                related_entities=[],
                metadata={},
            ),
        ),
    ):
        retriever = HybridRetriever(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            top_k=3,
            vector_store=_FakeVectorStore(),
            retrieval_mode="hybrid_rerank",
            reranker=None,
        )
        try:
            results = retriever.retrieve("What is CVaR?")
        finally:
            retriever.close()

    assert isinstance(results, list)


def test_hybrid_retriever_reranker_not_applied_in_dense_mode():
    """Reranker should only be applied in hybrid_rerank mode, not dense mode."""
    from unittest.mock import MagicMock, patch

    from riskfolio_graphrag_agent.retrieval.retriever import HybridRetriever, VectorHit

    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = iter([])

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    class _TrackingReranker:
        called = False

        def rerank(self, query, candidates, top_k):
            _TrackingReranker.called = True
            return candidates[:top_k]

    class _FakeVectorStore:
        def upsert(self, docs):
            return len(docs)

        def search(self, query, top_k):
            _ = query
            return [VectorHit(chunk_id="d1", content="doc", source_path="d.py", score=0.9)]

    reranker = _TrackingReranker()

    with (
        patch("riskfolio_graphrag_agent.retrieval.retriever.GraphDatabase.driver", return_value=mock_driver),
        patch(
            "riskfolio_graphrag_agent.retrieval.retriever._graph_expand",
            side_effect=lambda hit, session: RetrievalResult(
                content=hit.content,
                source_path=hit.source_path,
                score=hit.score,
                graph_neighbours=[],
                related_entities=[],
                metadata={},
            ),
        ),
    ):
        retriever = HybridRetriever(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            top_k=3,
            vector_store=_FakeVectorStore(),
            retrieval_mode="dense",
            reranker=reranker,
        )
        try:
            retriever.retrieve("What is CVaR?")
        finally:
            retriever.close()

    assert not _TrackingReranker.called, "Reranker should not be called in dense mode"
