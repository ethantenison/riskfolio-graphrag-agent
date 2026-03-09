"""Tests for Gradio app helper functions."""

from __future__ import annotations

from riskfolio_graphrag_agent.agent.workflow import AgentState
from riskfolio_graphrag_agent.app.gradio_ui import _render_graph_svg, run_query_with_graph


class _FakeRetriever:
    def __init__(self, *args, **kwargs):
        _ = args, kwargs

    def close(self):
        return None


class _FakeWorkflow:
    def __init__(self, *args, **kwargs):
        _ = args, kwargs

    def run(self, query: str) -> AgentState:
        return AgentState(
            question=query,
            answer="HRP uses clustering for risk-balanced allocation.",
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
                    "matched_entities": ["HRP"],
                    "graph_neighbours": ["Portfolio.py::chunk:1"],
                }
            ],
            verified=True,
        )


class _FakeGraphBuilder:
    def __init__(self, *args, **kwargs):
        _ = args, kwargs

    def get_query_subgraph(self, query: str, max_seed_nodes: int = 12, max_nodes: int = 40, max_edges: int = 80):
        _ = query, max_seed_nodes, max_nodes, max_edges
        return {
            "nodes": [
                {"id": "n1", "name": "Hierarchical Risk Parity", "labels": ["PortfolioMethod"], "source_path": "docs/hrp.md"},
                {"id": "n2", "name": "CVaR", "labels": ["RiskMeasure"], "source_path": "docs/risk.md"},
            ],
            "edges": [{"source": "n1", "target": "n2", "type": "SUPPORTS_RISK_MEASURE"}],
        }

    def close(self):
        return None


def test_run_query_with_graph_returns_answer_citations_and_graph(monkeypatch):
    monkeypatch.setattr("riskfolio_graphrag_agent.app.gradio_ui.HybridRetriever", _FakeRetriever)
    monkeypatch.setattr("riskfolio_graphrag_agent.app.gradio_ui.AgentWorkflow", _FakeWorkflow)
    monkeypatch.setattr("riskfolio_graphrag_agent.app.gradio_ui.GraphBuilder", _FakeGraphBuilder)

    answer, citations, graph = run_query_with_graph("What is HRP?", top_k=3)

    assert "HRP" in answer
    assert len(citations) == 1
    assert len(graph["nodes"]) == 2
    assert len(graph["edges"]) == 1


def test_render_graph_svg_contains_svg_markup():
    graph = {
        "nodes": [
            {"id": "n1", "name": "Hierarchical Risk Parity", "labels": ["PortfolioMethod"], "source_path": "docs/hrp.md"},
            {"id": "n2", "name": "CVaR", "labels": ["RiskMeasure"], "source_path": "docs/risk.md"},
        ],
        "edges": [{"source": "n1", "target": "n2", "type": "SUPPORTS_RISK_MEASURE"}],
    }

    rendered = _render_graph_svg(graph)
    assert "<svg" in rendered
    assert "Hierarchical Risk Parity" in rendered
    assert "SUPPORTS_RISK_MEASURE" in rendered


def test_render_graph_svg_empty_graph_message():
    rendered = _render_graph_svg({"nodes": [], "edges": []})
    assert "No graph data available" in rendered
