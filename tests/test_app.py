"""Tests for riskfolio_graphrag_agent.app.server."""

from __future__ import annotations

from fastapi.testclient import TestClient

from riskfolio_graphrag_agent.app.server import create_app
from riskfolio_graphrag_agent.graph.builder import GraphBuilder
from riskfolio_graphrag_agent.retrieval.retriever import RetrievalResult


class _FakeHybridRetriever:
    def __init__(self, *args, **kwargs):
        _ = args, kwargs

    def retrieve(self, query: str) -> list[RetrievalResult]:
        _ = query
        return [
            RetrievalResult(
                content="hrp summary",
                source_path="/tmp/Portfolio.py",
                score=0.91,
                graph_neighbours=["Portfolio.py::chunk:1"],
                related_entities=["HRP", "Portfolio"],
                metadata={
                    "chunk_id": "Portfolio.py::chunk:0",
                    "relative_path": "Portfolio.py",
                    "chunk_index": 0,
                    "section": "hrp_allocation",
                    "line_start": 10,
                    "line_end": 24,
                },
            )
        ]

    def close(self) -> None:
        return None


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
        "riskfolio_graphrag_agent.app.server.HybridRetriever",
        _FakeHybridRetriever,
    )

    client = TestClient(create_app())
    response = client.post("/query", json={"question": "What is HRP?", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert "ranked hybrid contexts" in body["answer"]
    assert len(body["citations"]) == 1
    assert body["citations"][0]["chunk_id"] == "Portfolio.py::chunk:0"
    assert body["citations"][0]["line_start"] == 10
    assert body["citations"][0]["line_end"] == 24
