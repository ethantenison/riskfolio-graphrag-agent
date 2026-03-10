"""Tests for riskfolio_graphrag_agent.app.server."""

from __future__ import annotations

from fastapi.testclient import TestClient

from riskfolio_graphrag_agent.agent.workflow import AgentState
from riskfolio_graphrag_agent.app.server import create_app
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
    assert response.json()["nodes"] == 10


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
