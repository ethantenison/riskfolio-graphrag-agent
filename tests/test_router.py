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


def test_router_routes_rp_portfolio_to_sparse():
    """A query referencing rp.Portfolio (API pattern) should route to sparse."""
    router = QueryToolRouter(min_confidence=0.1)
    decision = router.decide("What parameters does rp.Portfolio accept?")
    assert decision.mode == "sparse", (
        f"Expected 'sparse' for rp.Portfolio query, got '{decision.mode}' (reason: {decision.reason})"
    )


def test_router_routes_compare_query_to_hybrid_rerank():
    """A query comparing two methods should route to hybrid_rerank."""
    router = QueryToolRouter(min_confidence=0.1)
    decision = router.decide("Compare HRP and MVO in terms of risk-adjusted returns")
    assert decision.mode == "hybrid_rerank", (
        f"Expected 'hybrid_rerank' for compare query, got '{decision.mode}' (reason: {decision.reason})"
    )
