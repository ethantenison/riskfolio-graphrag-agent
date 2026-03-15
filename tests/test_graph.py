"""Tests for riskfolio_graphrag_agent.graph.builder."""

from __future__ import annotations

from typing import Any

from riskfolio_graphrag_agent.graph.builder import (
    GraphBuilder,
    GraphEdge,
    GraphNode,
    _batched_rows,
    _extract_entities,
    _extract_entities_with_llm,
    _upsert_edges,
    _upsert_nodes,
    emit_taxonomy_edges,
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
    edge_keys = {(edge.source_label, edge.source_name, edge.relation_type, edge.target_label, edge.target_name) for edge in edges}

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


def test_extract_entities_with_llm_timeout_returns_empty(caplog):
    doc = Document(
        content="def foo(): pass",
        source_path="/fake/path.py",
        metadata={"source_type": "python"},
    )

    def _timeout_llm_extract(*, content: str, source_type: str, model_name: str):
        _ = content, source_type, model_name
        raise TimeoutError("The read operation timed out")

    with caplog.at_level("WARNING"):
        nodes, edges = _extract_entities_with_llm(
            doc=doc,
            source_name="riskfolio.src.module",
            source_label="PythonModule",
            chunk_id="AuxFunctions.py::chunk:3",
            llm_extract=_timeout_llm_extract,
            llm_model_name="gpt-4o-mini",
        )

    assert nodes == []
    assert edges == []
    assert "LLM extraction failed for chunk AuxFunctions.py::chunk:3" in caplog.text
    assert "The read operation timed out" in caplog.text


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


class _FakeResult(list):
    def single(self):
        return self[0] if self else None

    def consume(self):
        return None


class _FakeSession:
    def __init__(self):
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        _ = exc_type, exc, tb
        return False

    def run(self, query: str, **params):
        self.calls.append((query, params))
        if "RETURN [n IN nodes" in query:
            terms = params.get("terms", [])
            if not terms:
                return _FakeResult([])
            return _FakeResult(
                [
                    {
                        "nodes": [
                            {
                                "id": "n1",
                                "name": "Hierarchical Risk Parity",
                                "labels": ["PortfolioMethod"],
                                "source_path": "docs/hrp.md",
                            },
                            {
                                "id": "n2",
                                "name": "CVaR",
                                "labels": ["RiskMeasure"],
                                "source_path": "docs/risk.md",
                            },
                        ]
                    }
                ]
            )
        if "MATCH (a)-[r]->(b)" in query:
            return _FakeResult(
                [
                    {
                        "source": "n1",
                        "target": "n2",
                        "type": "SUPPORTS_RISK_MEASURE",
                    }
                ]
            )
        return _FakeResult([])


def test_batched_rows_splits_large_payloads():
    rows = [{"index": index} for index in range(1201)]

    batches = _batched_rows(rows, batch_size=500)

    assert [len(batch) for batch in batches] == [500, 500, 201]


def test_upsert_nodes_batches_by_label():
    session = _FakeSession()
    nodes = [GraphNode(label="Concept", name=f"concept-{index}") for index in range(501)]

    _upsert_nodes(session, nodes)

    assert len(session.calls) == 2
    assert all("MERGE (n:Concept" in query for query, _ in session.calls)
    assert len(session.calls[0][1]["rows"]) == 500
    assert len(session.calls[1][1]["rows"]) == 1


def test_upsert_edges_batches_and_skips_missing_endpoints():
    session = _FakeSession()
    edges = [
        GraphEdge(
            source_name=f"source-{index}",
            target_name=f"target-{index}",
            relation_type="RELATED_TO",
            source_label="Concept",
            target_label="Concept",
        )
        for index in range(510)
    ]
    edges.append(
        GraphEdge(
            source_name="missing-source",
            target_name="missing-target",
            relation_type="RELATED_TO",
            source_label="Concept",
            target_label="Concept",
        )
    )
    known_node_names = frozenset({edge.source_name for edge in edges[:-1]} | {edge.target_name for edge in edges[:-1]})

    skipped = _upsert_edges(session, edges, known_node_names=known_node_names)

    assert skipped == 1
    assert len(session.calls) == 2
    assert all("UNWIND $rows AS row MATCH (s:Concept {name: row.source_name})" in query for query, _ in session.calls)
    assert len(session.calls[0][1]["rows"]) == 500
    assert len(session.calls[1][1]["rows"]) == 10


def test_emit_taxonomy_edges_is_subtype_of_cvar():
    """emit_taxonomy_edges should yield IS_SUBTYPE_OF edge from CVaR → RiskMeasure."""
    nodes, edges = emit_taxonomy_edges()

    node_names = {(n.label, n.name) for n in nodes}
    assert ("RiskMeasure", "CVaR") in node_names, "CVaR node with label RiskMeasure should be emitted"
    assert ("RiskMeasure", "RiskMeasure") in node_names, "RiskMeasure category node should be emitted"

    subtype_edges = {(e.source_name, e.target_name) for e in edges if e.relation_type == "IS_SUBTYPE_OF"}
    assert ("CVaR", "RiskMeasure") in subtype_edges, (
        f"IS_SUBTYPE_OF edge CVaR → RiskMeasure not found. Present IS_SUBTYPE_OF edges: {sorted(subtype_edges)[:10]}"
    )


def test_emit_taxonomy_edges_is_subtype_of_hrp():
    """emit_taxonomy_edges should yield IS_SUBTYPE_OF edge from HRP → PortfolioMethod."""
    _, edges = emit_taxonomy_edges()
    subtype_edges = {(e.source_name, e.target_name) for e in edges if e.relation_type == "IS_SUBTYPE_OF"}
    assert ("Hierarchical Risk Parity", "PortfolioMethod") in subtype_edges, (
        "IS_SUBTYPE_OF edge 'Hierarchical Risk Parity' → PortfolioMethod not found."
    )


def test_emit_taxonomy_edges_alternative_to():
    """emit_taxonomy_edges should yield bidirectional ALTERNATIVE_TO edges for CVaR ↔ VaR."""
    _, edges = emit_taxonomy_edges()
    alt_edges = {(e.source_name, e.target_name) for e in edges if e.relation_type == "ALTERNATIVE_TO"}
    assert ("CVaR", "VaR") in alt_edges
    assert ("VaR", "CVaR") in alt_edges


class _FakeDriver:
    def session(self):
        return _FakeSession()

    def close(self):
        return None


def test_graph_builder_get_query_subgraph_returns_nodes_and_edges(monkeypatch):
    builder = GraphBuilder(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
    )
    monkeypatch.setattr(builder, "_ensure_driver", lambda: _FakeDriver())

    graph = builder.get_query_subgraph("What is HRP and CVaR?")
    assert len(graph["nodes"]) == 2
    assert len(graph["edges"]) == 1
    assert graph["edges"][0]["type"] == "SUPPORTS_RISK_MEASURE"


def test_graph_builder_get_query_subgraph_empty_for_blank_query(monkeypatch):
    builder = GraphBuilder(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
    )
    monkeypatch.setattr(builder, "_ensure_driver", lambda: _FakeDriver())

    graph = builder.get_query_subgraph("  ")
    assert graph == {"nodes": [], "edges": []}
