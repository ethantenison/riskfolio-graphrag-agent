"""Tests for the redesigned KG induction pipeline."""

from __future__ import annotations

import json

from riskfolio_graphrag_agent.ingestion.loader import Document
from riskfolio_graphrag_agent.kg_pipeline import KnowledgeGraphPipeline


def test_kg_pipeline_writes_expected_artifacts(tmp_path):
    documents = [
        Document(
            content="Hierarchical Risk Parity uses CVaR.\n\ndef optimize(weights):\n    return weights\n",
            source_path=str(tmp_path / "risk.md"),
            chunk_index=0,
            chunk_id="risk.md::chunk:0",
            content_hash="hash-1",
            section="risk-overview",
            line_start=1,
            line_end=3,
            metadata={"source_type": "docs", "relative_path": "risk.md", "module_name": "risk"},
        )
    ]

    pipeline = KnowledgeGraphPipeline()
    result = pipeline.run(documents=documents, artifact_dir=tmp_path / "artifacts")

    assert result.graph_quality.num_chunks == 1
    assert result.graph_quality.num_mentions >= 2
    assert result.graph_quality.num_canonical_entities >= 1
    assert "schema_review" in result.artifact_paths
    assert (tmp_path / "artifacts" / "semantic" / "ontology.ttl").exists()
    assert (tmp_path / "artifacts" / "semantic" / "instances.ttl").exists()


def test_kg_pipeline_materialized_graph_contains_assertion_structure(tmp_path):
    documents = [
        Document(
            content="Hierarchical Risk Parity uses CVaR.",
            source_path=str(tmp_path / "risk.md"),
            chunk_index=0,
            chunk_id="risk.md::chunk:0",
            content_hash="hash-2",
            line_start=1,
            line_end=1,
            metadata={"source_type": "docs", "relative_path": "risk.md", "module_name": "risk"},
        )
    ]

    pipeline = KnowledgeGraphPipeline()
    pipeline.run(documents=documents, artifact_dir=tmp_path / "artifacts")

    materialized_graph = json.loads((tmp_path / "artifacts" / "materialized_graph.json").read_text())
    relationship_types = {edge["relationship_type"] for edge in materialized_graph["edges"]}
    node_labels = {node["label"] for node in materialized_graph["nodes"]}

    assert "CanonicalEntity" in node_labels
    assert "Assertion" in node_labels
    assert "ASSERTS_SUBJECT" in relationship_types
    assert "SUPPORTED_BY" in relationship_types
