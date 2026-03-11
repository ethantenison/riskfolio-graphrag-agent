"""Tests for RDF/SPARQL export and SHACL-like validation."""

from __future__ import annotations

from rdflib.namespace import OWL, RDFS

from riskfolio_graphrag_agent.graph.semantic_interop import (
    build_rdf_graph,
    export_rdf_owl_from_records,
    run_basic_sparql_queries,
    shacl_like_validate,
)


def _sample_nodes():
    return [
        {"name": "ChunkA", "labels": ["Chunk"]},
        {"name": "HRP", "labels": ["PortfolioMethod"]},
    ]


def _sample_edges():
    return [
        {"source": "ChunkA", "target": "HRP", "relation": "MENTIONS"},
    ]


def test_export_rdf_owl_from_records(tmp_path):
    output = tmp_path / "graph.ttl"
    info = export_rdf_owl_from_records(nodes=_sample_nodes(), edges=_sample_edges(), output_path=output)

    assert output.exists()
    assert info["triples"] > 0


def test_run_basic_sparql_queries(tmp_path):
    output = tmp_path / "graph.ttl"
    export_rdf_owl_from_records(nodes=_sample_nodes(), edges=_sample_edges(), output_path=output)

    results = run_basic_sparql_queries(output)
    assert "sample_nodes" in results
    assert "sample_mentions" in results


def test_shacl_like_validate_fail_case(tmp_path):
    report = shacl_like_validate(
        nodes=[{"name": "ChunkA", "labels": ["Chunk"]}],
        edges=[{"source": "ChunkA", "target": "MissingNode", "relation": "MENTIONS"}],
        output_path=tmp_path / "report.json",
    )

    assert report["status"] == "fail"
    assert report["fail_count"] > 0


def test_build_rdf_graph_contains_subclass_triples():
    """The exported graph must contain at least one rdfs:subClassOf triple from DOMAIN_ALIASES."""
    graph = build_rdf_graph(nodes=[], edges=[])
    subclass_triples = list(graph.triples((None, RDFS.subClassOf, None)))
    assert len(subclass_triples) > 0, "Expected rdfs:subClassOf triples from DOMAIN_ALIASES taxonomy but found none."


def test_build_rdf_graph_contains_datatype_property():
    """The exported graph must declare owl:DatatypeProperty entries (e.g. hasConfidenceLevel)."""
    from riskfolio_graphrag_agent.graph.semantic_interop import RF

    graph = build_rdf_graph(nodes=[], edges=[])
    dp_triples = list(graph.triples((None, None, OWL.DatatypeProperty)))
    assert len(dp_triples) > 0, "Expected owl:DatatypeProperty declarations, found none."
    # Specifically check for hasConfidenceLevel
    assert (RF["hasConfidenceLevel"], None, OWL.DatatypeProperty) in graph or any(
        str(s) == str(RF["hasConfidenceLevel"]) for s, _, _ in dp_triples
    ), "hasConfidenceLevel DatatypeProperty not found in graph."


def test_run_basic_sparql_queries_returns_subclass_results(tmp_path):
    """run_basic_sparql_queries must return IS_SUBTYPE_OF-like pairs in sample_subclass."""
    output = tmp_path / "graph.ttl"
    export_rdf_owl_from_records(nodes=_sample_nodes(), edges=_sample_edges(), output_path=output)
    results = run_basic_sparql_queries(output)
    assert "sample_subclass" in results
    # The taxonomy is embedded unconditionally; subclass results should be non-empty.
    assert len(results["sample_subclass"]) > 0, "Expected rdfs:subClassOf query results from built-in taxonomy, got empty list."
