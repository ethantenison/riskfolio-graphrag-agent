"""Tests for the redesigned KG induction pipeline."""

from __future__ import annotations

import json

from riskfolio_graphrag_agent.extraction.pipeline import LLMOpenExtractor
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


def test_llm_open_extractor_emits_assertions_and_events(tmp_path):
    document = Document(
        content="Hierarchical Risk Parity uses CVaR.",
        source_path=str(tmp_path / "risk.md"),
        chunk_index=0,
        chunk_id="risk.md::chunk:0",
        content_hash="hash-3",
        line_start=1,
        line_end=1,
        metadata={"source_type": "docs", "relative_path": "risk.md", "module_name": "risk"},
    )

    extractor = LLMOpenExtractor(
        llm_extract=lambda **kwargs: {
            "candidate_assertions": [
                {
                    "subject_text": "Hierarchical Risk Parity",
                    "subject_type_guess": "portfolio_method",
                    "predicate_text": "uses",
                    "object_text": "CVaR",
                    "object_type_guess": "risk_measure_like",
                    "statement": "Hierarchical Risk Parity uses CVaR",
                    "evidence_text": "Hierarchical Risk Parity uses CVaR.",
                    "confidence": 0.88,
                    "metadata": {"source": "llm"},
                }
            ],
            "candidate_events": [
                {
                    "trigger_text": "uses",
                    "event_type_guess": "usage",
                    "arguments": [
                        {"role": "subject", "text": "Hierarchical Risk Parity", "type_guess": "portfolio_method"},
                        {"role": "object", "text": "CVaR", "type_guess": "risk_measure_like"},
                    ],
                    "evidence_text": "Hierarchical Risk Parity uses CVaR.",
                    "confidence": 0.81,
                    "metadata": {"source": "llm"},
                }
            ],
        },
        model_name="test-llm-open-extractor",
    )

    extraction = extractor.extract_chunk(document)

    assert len(extraction.candidate_assertions) == 1
    assert extraction.candidate_assertions[0].relation_guess == "uses"
    assert len(extraction.candidate_events) == 1
    assert extraction.candidate_events[0].event_type_guess == "usage"


def test_kg_pipeline_accepts_llm_open_extractor(tmp_path):
    documents = [
        Document(
            content="Hierarchical Risk Parity uses CVaR.",
            source_path=str(tmp_path / "risk.md"),
            chunk_index=0,
            chunk_id="risk.md::chunk:0",
            content_hash="hash-4",
            line_start=1,
            line_end=1,
            metadata={"source_type": "docs", "relative_path": "risk.md", "module_name": "risk"},
        )
    ]

    extractor = LLMOpenExtractor(
        llm_extract=lambda **kwargs: {
            "candidate_assertions": [
                {
                    "subject_text": "Hierarchical Risk Parity",
                    "predicate_text": "uses",
                    "object_text": "CVaR",
                    "statement": "Hierarchical Risk Parity uses CVaR",
                    "evidence_text": "Hierarchical Risk Parity uses CVaR.",
                    "confidence": 0.9,
                    "metadata": {},
                }
            ],
            "candidate_events": [],
        },
        model_name="test-llm-open-extractor",
    )

    pipeline = KnowledgeGraphPipeline(extractor=extractor)
    result = pipeline.run(documents=documents, artifact_dir=tmp_path / "artifacts")

    assert result.graph_quality.num_candidate_assertions >= 1
    assert result.graph_quality.num_ontology_properties >= 1
