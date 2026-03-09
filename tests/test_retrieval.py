"""Tests for riskfolio_graphrag_agent.retrieval.retriever."""

from __future__ import annotations

from riskfolio_graphrag_agent.retrieval.retriever import (
    HybridRetriever,
    RetrievalResult,
    _graph_expand,
    _vector_search,
)


def test_retrieval_result_dataclass():
    result = RetrievalResult(content="some text", source_path="/a/b.py", score=0.9)
    assert result.score == 0.9
    assert result.graph_neighbours == []


def test_vector_search_stub():
    """_vector_search stub should return an empty list without raising."""
    results = _vector_search("test query", top_k=3)
    assert isinstance(results, list)


def test_graph_expand_stub():
    """_graph_expand stub should return the result unchanged."""
    r = RetrievalResult(content="x", source_path="/a.py")
    expanded = _graph_expand(r)
    assert expanded is r


def test_hybrid_retriever_retrieve_stub():
    """HybridRetriever.retrieve should return a list (empty in stub mode)."""
    retriever = HybridRetriever(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        top_k=3,
    )
    try:
        results = retriever.retrieve("What is portfolio optimisation?")
        assert isinstance(results, list)
    finally:
        retriever.close()
