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


def test_kg_pipeline_materializes_lowercase_assertions(tmp_path):
    documents = [
        Document(
            content="hierarchical risk parity uses cvar.",
            source_path=str(tmp_path / "risk.md"),
            chunk_index=0,
            chunk_id="risk.md::chunk:0",
            content_hash="hash-2b",
            line_start=1,
            line_end=1,
            metadata={"source_type": "docs", "relative_path": "risk.md", "module_name": "risk"},
        )
    ]

    pipeline = KnowledgeGraphPipeline()
    result = pipeline.run(documents=documents, artifact_dir=tmp_path / "artifacts")

    relationship_types = {
        edge["relationship_type"]
        for edge in json.loads((tmp_path / "artifacts" / "materialized_graph.json").read_text())["edges"]
    }

    assert result.graph_quality.num_candidate_assertions >= 1
    assert result.graph_quality.num_ontology_properties >= 1
    assert "SUPPORTED_BY" in relationship_types


def test_kg_pipeline_materializes_code_signature_assertions(tmp_path):
    documents = [
        Document(
            content="def optimize_portfolio(returns, covariance):\n    return returns\n",
            source_path=str(tmp_path / "risk.py"),
            chunk_index=0,
            chunk_id="risk.py::chunk:0",
            content_hash="hash-2c",
            line_start=1,
            line_end=2,
            metadata={"source_type": "code", "relative_path": "risk.py", "module_name": "riskfolio"},
        )
    ]

    pipeline = KnowledgeGraphPipeline()
    result = pipeline.run(documents=documents, artifact_dir=tmp_path / "artifacts")

    materialized_graph = json.loads((tmp_path / "artifacts" / "materialized_graph.json").read_text())
    assertion_nodes = [node for node in materialized_graph["nodes"] if node["label"] == "Assertion"]
    extractions = json.loads((tmp_path / "artifacts" / "extractions.json").read_text())
    extracted_relation_guesses = {
        assertion["relation_guess"] for extraction in extractions for assertion in extraction["candidate_assertions"]
    }

    assert result.graph_quality.num_candidate_assertions >= 1
    assert "defines" in extracted_relation_guesses or "accepts_parameter" in extracted_relation_guesses
    assert assertion_nodes == []


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


def test_semantic_relation_patterns_extraction(tmp_path):
    """Test that new semantic domain relations are correctly extracted."""
    documents = [
        Document(
            content=(
                "Risk Parity belongs to family of portfolio optimization methods. "
                "Equal Weight is alternative to Minimum Variance. "
                "Mean-Variance optimization requires return scenarios. "
                "Standard Deviation measures market volatility. "
                "Hierarchical Risk Parity method for portfolio construction. "
                "Black-Litterman is based on Bayesian inference. "
                "ChildOptimizer extends BaseOptimizer. "
                "RiskEngine implements interface OptimizerProtocol."
            ),
            source_path=str(tmp_path / "semantic.md"),
            chunk_index=0,
            chunk_id="semantic.md::chunk:0",
            content_hash="hash-semantic",
            line_start=1,
            line_end=5,
            metadata={"source_type": "docs", "relative_path": "semantic.md", "module_name": "portfolio"},
        )
    ]

    from riskfolio_graphrag_agent.extraction.pipeline import HeuristicOpenExtractor

    extractor = HeuristicOpenExtractor()
    extraction = extractor.extract_chunk(documents[0])

    relation_guesses = {a.relation_guess for a in extraction.candidate_assertions}

    assert "belongs_to_family" in relation_guesses
    assert "alternative_to" in relation_guesses
    assert "requires" in relation_guesses
    assert "measures" in relation_guesses
    assert "method_for" in relation_guesses
    assert "based_on" in relation_guesses
    assert "extends" in relation_guesses
    assert "implements_interface" in relation_guesses
