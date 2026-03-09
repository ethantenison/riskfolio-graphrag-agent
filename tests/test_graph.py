"""Tests for riskfolio_graphrag_agent.graph.builder."""

from __future__ import annotations

from riskfolio_graphrag_agent.graph.builder import (
    GraphBuilder,
    GraphEdge,
    GraphNode,
    _extract_entities,
)
from riskfolio_graphrag_agent.ingestion.loader import Document


def _make_doc(content: str = "def foo(): pass") -> Document:
    return Document(content=content, source_path="/fake/path.py")


def test_graph_node_dataclass():
    node = GraphNode(label="Function", name="foo")
    assert node.label == "Function"
    assert node.name == "foo"
    assert node.properties == {}


def test_graph_edge_dataclass():
    edge = GraphEdge(source_name="foo", target_name="bar", relation_type="CALLS")
    assert edge.relation_type == "CALLS"


def test_extract_entities_returns_lists():
    """_extract_entities stub should return empty lists without raising."""
    doc = _make_doc()
    nodes, edges = _extract_entities(doc)
    assert isinstance(nodes, list)
    assert isinstance(edges, list)


def test_extract_entities_includes_llm_nodes_and_edges():
    doc = Document(
        content="HRP optimization uses CVaR",
        source_path="/fake/path.py",
        metadata={"source_type": "python", "relative_path": "riskfolio/src/module.py"},
    )

    def _fake_llm_extract(*, content: str, source_type: str, model_name: str):
        assert content
        assert source_type == "python"
        assert model_name == "gpt-4o-mini"
        return {
            "nodes": [
                {"label": "PortfolioMethod", "name": "Hierarchical Risk Parity", "properties": {}},
                {"label": "RiskMeasure", "name": "CVaR", "properties": {}},
            ],
            "edges": [
                {
                    "source_name": "Hierarchical Risk Parity",
                    "source_label": "PortfolioMethod",
                    "target_name": "CVaR",
                    "target_label": "RiskMeasure",
                    "relation_type": "SUPPORTS_RISK_MEASURE",
                }
            ],
        }

    nodes, edges = _extract_entities(doc, llm_extract=_fake_llm_extract, llm_model_name="gpt-4o-mini")
    node_pairs = {(node.label, node.name) for node in nodes}
    edge_keys = {
        (edge.source_label, edge.source_name, edge.relation_type, edge.target_label, edge.target_name)
        for edge in edges
    }

    assert ("PortfolioMethod", "Hierarchical Risk Parity") in node_pairs
    assert ("RiskMeasure", "CVaR") in node_pairs
    assert (
        "PortfolioMethod",
        "Hierarchical Risk Parity",
        "SUPPORTS_RISK_MEASURE",
        "RiskMeasure",
        "CVaR",
    ) in edge_keys


def test_extract_entities_filters_invalid_llm_output():
    doc = _make_doc(content="some content")

    def _fake_llm_extract(*, content: str, source_type: str, model_name: str):
        _ = content, source_type, model_name
        return {
            "nodes": [
                {"label": "UnknownLabel", "name": "Bad Node", "properties": {}},
            ],
            "edges": [
                {
                    "source_name": "x",
                    "source_label": "Unknown",
                    "target_name": "y",
                    "target_label": "Concept",
                    "relation_type": "UNKNOWN_REL",
                }
            ],
        }

    nodes, edges = _extract_entities(doc, llm_extract=_fake_llm_extract)
    names = {node.name for node in nodes}
    assert "Bad Node" not in names
    assert all(edge.relation_type != "UNKNOWN_REL" for edge in edges)


def test_graph_builder_build_stub(tmp_source_dir):
    """GraphBuilder.build should run without error in stub mode."""
    from riskfolio_graphrag_agent.ingestion.loader import load_directory

    docs = load_directory(tmp_source_dir)
    builder = GraphBuilder(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
    )
    builder._ensure_driver = lambda: (_ for _ in ()).throw(OSError("offline test mode"))
    # Should not raise when Neo4j is unavailable.
    builder.build(docs)
    builder.close()
