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
