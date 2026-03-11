"""Tests for Gradio app helper functions."""

from __future__ import annotations

from riskfolio_graphrag_agent.agent.workflow import AgentState
from riskfolio_graphrag_agent.app.gradio_ui import (
    _render_governance_html,
    _render_graph_evidence_html,
    _render_graph_svg,
    _render_grounding_html,
    _render_routing_html,
    run_query_with_graph,
)


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
                {
                    "id": "n1",
                    "name": "Hierarchical Risk Parity",
                    "labels": ["PortfolioMethod"],
                    "source_path": "docs/hrp.md",
                },
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

    answer, citations, graph, insights = run_query_with_graph("What is HRP?", top_k=3)

    assert "HRP" in answer
    assert len(citations) == 1
    assert len(graph["nodes"]) == 2
    assert len(graph["edges"]) == 1
    assert isinstance(insights, dict)
    assert {"routing", "grounding", "graph_evidence", "governance"} <= insights.keys()
    # grounding: verified=True, citation_count=1, avg_score=0.91, entity HRP present
    assert insights["grounding"]["verified"] is True
    assert insights["grounding"]["citation_count"] == 1
    assert "HRP" in insights["grounding"]["unique_entities"]
    # graph_evidence: subgraph populated by _FakeGraphBuilder
    assert insights["graph_evidence"]["subgraph_nodes"] == 2
    assert insights["graph_evidence"]["subgraph_edges"] == 1
    # governance keys present
    assert "model" in insights["governance"]
    assert "adaptive_routing_enabled" in insights["governance"]


def test_render_routing_html_with_data():
    insights = {"routing": [{"sub_question": "What is HRP?", "mode": "graph", "confidence": 0.92, "reason": "rule_graph_intent"}]}
    html_out = _render_routing_html(insights)
    assert "graph" in html_out
    assert "What is HRP?" in html_out
    assert "0.92" in html_out


def test_render_grounding_html_verified():
    insights = {
        "grounding": {
            "verified": True,
            "citation_count": 2,
            "avg_score": 0.85,
            "unique_entities": ["HRP", "CVaR"],
        }
    }
    html_out = _render_grounding_html(insights)
    assert "Verified" in html_out
    assert "HRP" in html_out


def test_render_graph_evidence_html_with_data():
    insights = {
        "graph_evidence": {
            "unique_entities": ["HRP"],
            "unique_neighbours": ["CVaR"],
            "subgraph_nodes": 3,
            "subgraph_edges": 2,
        }
    }
    html_out = _render_graph_evidence_html(insights)
    assert "HRP" in html_out
    assert "CVaR" in html_out
    assert "3" in html_out


def test_render_governance_html_with_data():
    insights = {
        "governance": {
            "model": "gpt-4o-mini",
            "base_retrieval_mode": "hybrid_rerank",
            "adaptive_routing_enabled": True,
            "vector_backend": "chroma",
            "sub_questions": ["What is HRP?"],
            "estimated_cost_usd": 0.000003,
        }
    }
    html_out = _render_governance_html(insights)
    assert "gpt-4o-mini" in html_out
    assert "hybrid_rerank" in html_out
    assert "ON" in html_out
    assert "What is HRP?" in html_out


def test_render_graph_svg_contains_svg_markup():
    graph = {
        "nodes": [
            {
                "id": "n1",
                "name": "Hierarchical Risk Parity",
                "labels": ["PortfolioMethod"],
                "source_path": "docs/hrp.md",
            },
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
