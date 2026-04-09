"""Tests for riskfolio_graphrag_agent.app.server."""

from __future__ import annotations

from fastapi.testclient import TestClient

from riskfolio_graphrag_agent.agent.workflow import AgentState
from riskfolio_graphrag_agent.app.server import (
    GraphStatsResponse,
    NLToCypherResponse,
    _build_background_hint,
    _is_definition_question,
    create_app,
)
from riskfolio_graphrag_agent.graph.builder import GraphBuilder


class _FakeAgentWorkflow:
    def __init__(self, *args, **kwargs):
        _ = args, kwargs

    def run(self, query: str) -> AgentState:
        _ = query
        return AgentState(
            question=query,
            answer="For 'What is HRP?', retrieved evidence indicates key concepts: HRP.",
            citations=[
                {
                    "chunk_id": "Portfolio.py::chunk:0",
                    "source_path": "/tmp/Portfolio.py",
                    "relative_path": "Portfolio.py",
                    "chunk_index": 0,
                    "section": "hrp_allocation",
                    "line_start": 10,
                    "line_end": 24,
                    "score": 0.91,
                    "matched_entities": ["HRP", "Portfolio"],
                    "graph_neighbours": ["Portfolio.py::chunk:1"],
                }
            ],
            verified=True,
        )


def test_health_endpoint():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_graph_stats_endpoint(monkeypatch):
    monkeypatch.setattr(
        GraphBuilder,
        "get_stats",
        lambda self: {
            "nodes": 10,
            "relationships": 5,
            "node_counts_by_label": {"Document": 4},
            "relationship_counts_by_type": {"MENTIONS": 5},
        },
    )

    client = TestClient(create_app())
    response = client.get("/graph/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["nodes"] == 10
    # promoted_mode defaults False when not returned by get_stats mock
    assert body["promoted_mode"] is False


def test_graph_stats_endpoint_promoted_mode(monkeypatch):
    monkeypatch.setattr(
        GraphBuilder,
        "get_stats",
        lambda self: {
            "nodes": 42,
            "relationships": 20,
            "node_counts_by_label": {"CanonicalEntity": 30},
            "relationship_counts_by_type": {"ASSERTS_SUBJECT": 20},
            "promoted_mode": True,
        },
    )

    client = TestClient(create_app())
    response = client.get("/graph/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["nodes"] == 42
    assert body["promoted_mode"] is True
    assert "CanonicalEntity" in body["node_counts_by_label"]


def test_query_endpoint_returns_citations(monkeypatch):
    monkeypatch.setattr(
        "riskfolio_graphrag_agent.app.server.AgentWorkflow",
        _FakeAgentWorkflow,
    )

    client = TestClient(create_app())
    response = client.post("/query", json={"question": "What is HRP?", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert "retrieved evidence" in body["answer"]
    assert len(body["citations"]) == 1
    assert body["citations"][0]["chunk_id"] == "Portfolio.py::chunk:0"
    assert body["citations"][0]["line_start"] == 10
    assert body["citations"][0]["line_end"] == 24


def test_query_endpoint_wires_llm_generate_when_openai_configured(monkeypatch):
    captured: dict[str, object] = {}

    class _CaptureWorkflow:
        def __init__(self, *args, **kwargs):
            _ = args
            captured.update(kwargs)

        def run(self, query: str) -> AgentState:
            _ = query
            return AgentState(
                question=query,
                answer="Model-backed answer",
                citations=[
                    {
                        "chunk_id": "Portfolio.py::chunk:0",
                        "source_path": "/tmp/Portfolio.py",
                        "relative_path": "Portfolio.py",
                        "chunk_index": 0,
                        "section": "hrp_allocation",
                        "line_start": 10,
                        "line_end": 24,
                        "score": 0.91,
                        "matched_entities": ["HRP", "Portfolio"],
                        "graph_neighbours": ["Portfolio.py::chunk:1"],
                    }
                ],
                verified=True,
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_ENABLE_GENERATION", "true")
    monkeypatch.setattr(
        "riskfolio_graphrag_agent.app.server.AgentWorkflow",
        _CaptureWorkflow,
    )

    client = TestClient(create_app())
    response = client.post("/query", json={"question": "What is HRP?", "top_k": 3})
    assert response.status_code == 200
    assert callable(captured.get("llm_generate"))


def test_nl2cypher_endpoint_blocks_unsafe_query():
    client = TestClient(create_app())
    response = client.post("/graph/nl2cypher", json={"question": "delete all nodes", "tenant_id": "demo"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "blocked"
    assert body["requires_human_review"] is True
    # graph_mode is unknown for blocked queries that never reach Neo4j
    assert body["graph_mode"] == "unknown"


def test_graph_stats_response_model_defaults():
    """GraphStatsResponse should expose promoted_mode with False default."""
    stats = GraphStatsResponse(nodes=5, relationships=3)
    assert stats.promoted_mode is False

    promoted_stats = GraphStatsResponse(nodes=10, relationships=8, promoted_mode=True)
    assert promoted_stats.promoted_mode is True


def test_nl2cypher_response_model_defaults():
    """NLToCypherResponse graph_mode should default to 'unknown'."""
    resp = NLToCypherResponse(status="safe", reason="ok", requires_human_review=False)
    assert resp.graph_mode == "unknown"

    resp_promoted = NLToCypherResponse(status="safe", reason="ok", requires_human_review=False, graph_mode="promoted")
    assert resp_promoted.graph_mode == "promoted"


def test_definition_question_detection():
    assert _is_definition_question("What is CVaR?") is True
    assert _is_definition_question("Define HRP") is True
    assert _is_definition_question("How does HRP work?") is False


def test_background_hint_detects_aliases():
    hint = _build_background_hint("What is CVaR?")
    assert "CVaR" in hint
