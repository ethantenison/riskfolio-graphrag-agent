"""Schema induction for the redesigned KG induction pipeline.

This module aggregates open-world type guesses and relation phrases into
reviewable schema proposals. It deliberately separates schema candidates from
stabilized ontology commitments so reviewers can inspect cluster support before
promotion.

Inputs are open extraction artifacts and canonicalization outputs. Outputs are
schema candidates, ontology classes, ontology properties, concept schemes, and
review-oriented Markdown summaries.

This module does not write Neo4j or produce semantic exports directly.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from riskfolio_graphrag_agent.kg_models import (
    CanonicalizationResult,
    ConceptScheme,
    OntologyClass,
    OntologyProperty,
    OpenChunkExtraction,
    ReviewStatus,
    SchemaInductionCandidate,
    SchemaInductionResult,
    slugify,
    stable_id,
)


class SchemaInductionPipeline:
    """Aggregate semantic guesses into reviewable schema proposals."""

    def run(
        self,
        extractions: Sequence[OpenChunkExtraction],
        canonicalization: CanonicalizationResult,
    ) -> SchemaInductionResult:
        """Build schema proposals and stabilized ontology elements.

        Args:
            extractions: Chunk-level open extraction results.
            canonicalization: Canonicalization outputs for entities and predicates.

        Returns:
            Reviewable schema induction results.
        """
        entity_type_support: dict[str, list[str]] = defaultdict(list)
        relation_support: dict[str, list[str]] = defaultdict(list)
        event_type_support: dict[str, list[str]] = defaultdict(list)

        for entity in canonicalization.canonical_entities:
            entity_type_support[entity.type_guess].append(entity.canonical_entity_id)
        for predicate in canonicalization.canonical_predicates:
            relation_support[predicate.preferred_label].extend(predicate.assertion_ids)
        for event_type in canonicalization.canonical_event_types:
            event_type_support[event_type.preferred_label].extend(event_type.event_ids)

        candidates: list[SchemaInductionCandidate] = []
        ontology_classes: list[OntologyClass] = []
        ontology_properties: list[OntologyProperty] = []

        for label, support_ids in entity_type_support.items():
            candidate_id = stable_id("schema-candidate", "entity-type", label)
            confidence = round(min(1.0, 0.45 + (0.08 * len(support_ids))), 3)
            candidates.append(
                SchemaInductionCandidate(
                    candidate_id=candidate_id,
                    candidate_kind="entity_type",
                    proposed_label=self._humanize_label(label),
                    lexical_variants=sorted({label}),
                    supporting_ids=support_ids,
                    support_count=len(support_ids),
                    confidence=confidence,
                    status=ReviewStatus.PROPOSED,
                    notes="Derived from canonical entity type guesses.",
                )
            )
            ontology_classes.append(
                OntologyClass(
                    ontology_class_id=stable_id("ontology-class", label),
                    label=self._humanize_label(label),
                    definition=f"Induced class for entities labeled as {label} during open extraction.",
                    source_candidate_ids=[candidate_id],
                    confidence=confidence,
                    status=ReviewStatus.PROPOSED,
                )
            )

        for label, support_ids in relation_support.items():
            candidate_id = stable_id("schema-candidate", "relation-type", label)
            confidence = round(min(1.0, 0.5 + (0.06 * len(support_ids))), 3)
            candidates.append(
                SchemaInductionCandidate(
                    candidate_id=candidate_id,
                    candidate_kind="relation_type",
                    proposed_label=self._humanize_label(label),
                    lexical_variants=sorted({label}),
                    supporting_ids=support_ids,
                    support_count=len(support_ids),
                    confidence=confidence,
                    status=ReviewStatus.PROPOSED,
                    notes="Derived from canonical predicate clusters.",
                )
            )
            ontology_properties.append(
                OntologyProperty(
                    ontology_property_id=stable_id("ontology-property", label),
                    label=self._humanize_label(label),
                    definition=f"Induced property for assertions expressed with relation phrase '{label}'.",
                    source_candidate_ids=[candidate_id],
                    confidence=confidence,
                    status=ReviewStatus.PROPOSED,
                )
            )

        if event_type_support:
            for label, support_ids in event_type_support.items():
                candidates.append(
                    SchemaInductionCandidate(
                        candidate_id=stable_id("schema-candidate", "event-type", label),
                        candidate_kind="event_type",
                        proposed_label=self._humanize_label(label),
                        lexical_variants=sorted({label}),
                        supporting_ids=support_ids,
                        support_count=len(support_ids),
                        confidence=round(min(1.0, 0.45 + (0.06 * len(support_ids))), 3),
                        status=ReviewStatus.PROPOSED,
                        notes="Derived from candidate event frames.",
                    )
                )

        concept_schemes = [
            ConceptScheme(
                concept_scheme_id=stable_id("concept-scheme", "induced-entity-types"),
                label="Induced Entity Types",
                concept_ids=[item.ontology_class_id for item in ontology_classes],
                status=ReviewStatus.PROPOSED,
            ),
            ConceptScheme(
                concept_scheme_id=stable_id("concept-scheme", "induced-relation-types"),
                label="Induced Relation Types",
                concept_ids=[item.ontology_property_id for item in ontology_properties],
                status=ReviewStatus.PROPOSED,
            ),
        ]

        review_markdown = self._build_review_markdown(extractions, canonicalization, candidates)
        return SchemaInductionResult(
            candidates=candidates,
            ontology_classes=ontology_classes,
            ontology_properties=ontology_properties,
            concept_schemes=concept_schemes,
            review_markdown=review_markdown,
        )

    def _build_review_markdown(
        self,
        extractions: Sequence[OpenChunkExtraction],
        canonicalization: CanonicalizationResult,
        candidates: list[SchemaInductionCandidate],
    ) -> str:
        extraction_count = len(extractions)
        class_count = len(canonicalization.canonical_entities)
        predicate_count = len(canonicalization.canonical_predicates)
        counts = Counter(candidate.candidate_kind for candidate in candidates)
        lines = [
            "# Schema Induction Review",
            "",
            f"- Source chunks reviewed: {extraction_count}",
            f"- Canonical entities: {class_count}",
            f"- Canonical predicates: {predicate_count}",
            f"- Proposed entity types: {counts.get('entity_type', 0)}",
            f"- Proposed relation types: {counts.get('relation_type', 0)}",
            f"- Proposed event types: {counts.get('event_type', 0)}",
            "",
            "## Candidate Summary",
            "",
        ]
        for candidate in sorted(candidates, key=lambda item: (item.candidate_kind, -item.support_count, item.proposed_label)):
            lines.append(
                f"- [{candidate.candidate_kind}] {candidate.proposed_label}: "
                f"support={candidate.support_count} confidence={candidate.confidence:.2f}"
            )
        return "\n".join(lines)

    def _humanize_label(self, label: str) -> str:
        words = slugify(label).split("-")
        return " ".join(word.capitalize() for word in words if word)
