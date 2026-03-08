"""Tests for riskfolio_graphrag_agent.app.server."""

from __future__ import annotations

from fastapi.testclient import TestClient

from riskfolio_graphrag_agent.app.server import create_app
from riskfolio_graphrag_agent.graph.builder import GraphBuilder


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def run(self, _cypher, **_kwargs):
        return _FakeResult(self._rows)


class _FakeDriver:
    def __init__(self, rows):
        self._rows = rows

    def session(self):
        return _FakeSession(self._rows)


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
    rows = [
        {
            "document": "Portfolio_py_0",
            "source_path": "/tmp/Portfolio.py",
            "chunk_index": 0,
            "matched_entities": ["HRP", "Portfolio"],
            "score": 2,
        }
    ]

    monkeypatch.setattr(GraphBuilder, "_ensure_driver", lambda self: _FakeDriver(rows))

    client = TestClient(create_app())
    response = client.post("/query", json={"question": "What is HRP?", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert "matching graph chunks" in body["answer"]
    assert len(body["citations"]) == 1
    assert body["citations"][0]["document"] == "Portfolio_py_0"
