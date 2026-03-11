"""Tests for adaptive query routing."""

from __future__ import annotations

from riskfolio_graphrag_agent.retrieval.router import QueryToolRouter


def test_router_prefers_graph_for_relationship_queries():
    router = QueryToolRouter(min_confidence=0.1)
    decision = router.decide("What relationship exists between HRP and CVaR in the graph?")
    assert decision.mode == "graph"
    assert decision.confidence >= 0.1


def test_router_prefers_sparse_for_exact_lookup_queries():
    router = QueryToolRouter(min_confidence=0.1)
    decision = router.decide("Find exact file path and line for the hrp_allocation function signature")
    assert decision.mode == "sparse"


def test_router_falls_back_to_hybrid_on_low_confidence():
    router = QueryToolRouter(min_confidence=0.99)
    decision = router.decide("x")
    assert decision.mode == "hybrid_rerank"
