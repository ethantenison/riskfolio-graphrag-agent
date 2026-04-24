"""Graph-quality metrics for the redesigned KG induction pipeline.

This module evaluates the graph itself rather than only answer quality. The
metrics are intentionally lightweight but structurally aligned with the new
pipeline: extraction volume, canonicalization compression, schema support, and
promotion yield.

Inputs are extraction, canonicalization, schema induction, and materialization
outputs. Output is a compact `GraphQualityReport` suitable for artifacts and
future regression gates.
"""

from __future__ import annotations

from collections.abc import Sequence

from riskfolio_graphrag_agent.kg_models import (
    CanonicalizationResult,
    GraphQualityReport,
    GraphWritePlan,
    OpenChunkExtraction,
    SchemaInductionResult,
)


def evaluate_graph_quality(
    *,
    extractions: Sequence[OpenChunkExtraction],
    canonicalization: CanonicalizationResult,
    schema_induction: SchemaInductionResult,
    write_plan: GraphWritePlan,
) -> GraphQualityReport:
    """Compute graph-quality metrics over one pipeline run.

    Args:
        extractions: Chunk-level open extraction outputs.
        canonicalization: Canonicalization outputs.
        schema_induction: Schema induction outputs.
        write_plan: Final materialized graph write plan.

    Returns:
        A graph-quality report aligned with the redesigned pipeline.
    """
    num_chunks = len(extractions)
    mentions = [mention for extraction in extractions for mention in extraction.mentions]
    candidate_entities = [candidate for extraction in extractions for candidate in extraction.candidate_entities]
    assertions = [assertion for extraction in extractions for assertion in extraction.candidate_assertions]
    events = [event for extraction in extractions for event in extraction.candidate_events]
    promoted_assertions = [node for node in write_plan.nodes if node.label == "Assertion"]

    return GraphQualityReport(
        num_chunks=num_chunks,
        num_mentions=len(mentions),
        num_candidate_entities=len(candidate_entities),
        num_candidate_assertions=len(assertions),
        num_candidate_events=len(events),
        num_canonical_entities=len(canonicalization.canonical_entities),
        num_canonical_predicates=len(canonicalization.canonical_predicates),
        num_ontology_classes=len(schema_induction.ontology_classes),
        num_ontology_properties=len(schema_induction.ontology_properties),
        promoted_assertion_ratio=(len(promoted_assertions) / len(assertions)) if assertions else 0.0,
        entity_compression_ratio=(
            len(canonicalization.canonical_entities) / len(candidate_entities) if candidate_entities else 0.0
        ),
        mean_assertion_confidence=(
            sum(assertion.confidence for assertion in assertions) / len(assertions) if assertions else 0.0
        ),
        mean_canonical_entity_confidence=(
            sum(entity.confidence for entity in canonicalization.canonical_entities) / len(canonicalization.canonical_entities)
            if canonicalization.canonical_entities
            else 0.0
        ),
        schema_support_ratio=(
            (len(schema_induction.ontology_classes) + len(schema_induction.ontology_properties))
            / len(schema_induction.candidates)
            if schema_induction.candidates
            else 0.0
        ),
    )
