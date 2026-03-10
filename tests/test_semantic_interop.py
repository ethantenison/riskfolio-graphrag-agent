"""Tests for RDF/SPARQL export and SHACL-like validation."""

from __future__ import annotations

from riskfolio_graphrag_agent.graph.semantic_interop import (
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
