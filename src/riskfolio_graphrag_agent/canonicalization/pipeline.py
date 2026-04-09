"""Canonicalization for the redesigned KG induction pipeline.

This module clusters candidate entities, relation phrases, and event-type
guesses into canonical semantic records without collapsing away provenance.
Decisions remain reviewable and explicitly record the mechanism used,
confidence, and alternatives considered.

Inputs are open extraction artifacts. Outputs are canonical entities,
predicates, event types, and canonicalization decision records.

This module does not stabilize ontology classes or write Neo4j.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from difflib import SequenceMatcher

from riskfolio_graphrag_agent.kg_models import (
    CanonicalEntity,
    CanonicalEventType,
    CanonicalizationDecision,
    CanonicalizationResult,
    CanonicalPredicate,
    DecisionSource,
    OpenChunkExtraction,
    ReviewStatus,
    stable_id,
)


class CanonicalizationPipeline:
    """Cluster open extraction outputs into canonical semantic records."""

    def run(self, extractions: Sequence[OpenChunkExtraction]) -> CanonicalizationResult:
        """Canonicalize entities, predicates, and event types.

        Args:
            extractions: Chunk-level open extraction results.

        Returns:
            Canonicalization outputs across entities, predicates, and events.
        """
        entity_groups = self._cluster_entities(extractions)
        predicate_groups = self._cluster_predicates(extractions)
        event_groups = self._cluster_event_types(extractions)

        decisions: list[CanonicalizationDecision] = []
        canonical_entities: list[CanonicalEntity] = []
        canonical_predicates: list[CanonicalPredicate] = []
        canonical_event_types: list[CanonicalEventType] = []

        for normalized_label, candidates in entity_groups.items():
            preferred_label = Counter(candidate.display_name for candidate in candidates).most_common(1)[0][0]
            type_guess = Counter(candidate.type_guess for candidate in candidates).most_common(1)[0][0]
            canonical_id = stable_id("canonical-entity", normalized_label, type_guess)
            mention_ids = [mention_id for candidate in candidates for mention_id in candidate.mention_ids]
            provenance_chunk_ids = list(dict.fromkeys(candidate.chunk_id for candidate in candidates))
            canonical_entities.append(
                CanonicalEntity(
                    canonical_entity_id=canonical_id,
                    preferred_label=preferred_label,
                    normalized_label=normalized_label,
                    type_guess=type_guess,
                    mention_ids=mention_ids,
                    candidate_entity_ids=[candidate.candidate_entity_id for candidate in candidates],
                    provenance_chunk_ids=provenance_chunk_ids,
                    confidence=round(sum(candidate.confidence for candidate in candidates) / len(candidates), 3),
                    status=ReviewStatus.PROPOSED,
                    metadata={"cluster_size": len(candidates)},
                )
            )
            for candidate in candidates:
                decisions.append(
                    CanonicalizationDecision(
                        decision_id=stable_id("decision", candidate.candidate_entity_id, canonical_id),
                        source_kind="entity",
                        source_id=candidate.candidate_entity_id,
                        canonical_id=canonical_id,
                        decision_source=DecisionSource.HYBRID,
                        confidence=self._lexical_similarity(candidate.normalized_name, normalized_label),
                        rationale=(f"Grouped by lexical normalization and similarity around '{preferred_label}'."),
                        alternatives=[group for group in entity_groups if group != normalized_label][:3],
                        status=ReviewStatus.PROPOSED,
                    )
                )

        for normalized_label, payload in predicate_groups.items():
            relation_guesses, assertion_ids, confidence = payload
            canonical_predicate_id = stable_id("canonical-predicate", normalized_label)
            canonical_predicates.append(
                CanonicalPredicate(
                    canonical_predicate_id=canonical_predicate_id,
                    preferred_label=Counter(relation_guesses).most_common(1)[0][0],
                    normalized_label=normalized_label,
                    relation_guesses=sorted(set(relation_guesses)),
                    assertion_ids=assertion_ids,
                    confidence=confidence,
                    status=ReviewStatus.PROPOSED,
                )
            )
            for assertion_id in assertion_ids:
                decisions.append(
                    CanonicalizationDecision(
                        decision_id=stable_id("decision", assertion_id, canonical_predicate_id),
                        source_kind="predicate",
                        source_id=assertion_id,
                        canonical_id=canonical_predicate_id,
                        decision_source=DecisionSource.HEURISTIC,
                        confidence=confidence,
                        rationale="Grouped relation phrases by lexical normalization.",
                        alternatives=[],
                        status=ReviewStatus.PROPOSED,
                    )
                )

        for normalized_label, payload in event_groups.items():
            labels, event_ids, confidence = payload
            canonical_event_type_id = stable_id("canonical-event-type", normalized_label)
            canonical_event_types.append(
                CanonicalEventType(
                    canonical_event_type_id=canonical_event_type_id,
                    preferred_label=Counter(labels).most_common(1)[0][0],
                    normalized_label=normalized_label,
                    event_ids=event_ids,
                    confidence=confidence,
                    status=ReviewStatus.PROPOSED,
                )
            )
            for event_id in event_ids:
                decisions.append(
                    CanonicalizationDecision(
                        decision_id=stable_id("decision", event_id, canonical_event_type_id),
                        source_kind="event_type",
                        source_id=event_id,
                        canonical_id=canonical_event_type_id,
                        decision_source=DecisionSource.HEURISTIC,
                        confidence=confidence,
                        rationale="Grouped event-type guesses by lexical normalization.",
                        alternatives=[],
                        status=ReviewStatus.PROPOSED,
                    )
                )

        return CanonicalizationResult(
            canonical_entities=canonical_entities,
            canonical_predicates=canonical_predicates,
            canonical_event_types=canonical_event_types,
            decisions=decisions,
        )

    def _cluster_entities(self, extractions: Sequence[OpenChunkExtraction]) -> dict[str, list]:
        candidates = [candidate for extraction in extractions for candidate in extraction.candidate_entities]
        groups: dict[str, list] = defaultdict(list)
        for candidate in candidates:
            group_key = self._best_group_key(candidate.normalized_name, groups)
            groups[group_key].append(candidate)
        return dict(groups)

    def _cluster_predicates(self, extractions: Sequence[OpenChunkExtraction]) -> dict[str, tuple[list[str], list[str], float]]:
        groups: dict[str, tuple[list[str], list[str], float]] = {}
        raw_groups: dict[str, list] = defaultdict(list)
        for extraction in extractions:
            for assertion in extraction.candidate_assertions:
                raw_groups[assertion.relation_guess.casefold()].append(assertion)
        for normalized_label, assertions in raw_groups.items():
            groups[normalized_label] = (
                [assertion.relation_guess for assertion in assertions],
                [assertion.assertion_id for assertion in assertions],
                round(sum(assertion.confidence for assertion in assertions) / len(assertions), 3),
            )
        return groups

    def _cluster_event_types(self, extractions: Sequence[OpenChunkExtraction]) -> dict[str, tuple[list[str], list[str], float]]:
        groups: dict[str, tuple[list[str], list[str], float]] = {}
        raw_groups: dict[str, list] = defaultdict(list)
        for extraction in extractions:
            for event in extraction.candidate_events:
                raw_groups[event.event_type_guess.casefold()].append(event)
        for normalized_label, events in raw_groups.items():
            groups[normalized_label] = (
                [event.event_type_guess for event in events],
                [event.candidate_event_id for event in events],
                round(sum(event.confidence for event in events) / len(events), 3),
            )
        return groups

    def _best_group_key(self, normalized_name: str, groups: dict[str, list]) -> str:
        if normalized_name in groups:
            return normalized_name
        for group_key in groups:
            if self._lexical_similarity(group_key, normalized_name) >= 0.9:
                return group_key
        return normalized_name

    def _lexical_similarity(self, left: str, right: str) -> float:
        return round(SequenceMatcher(a=left, b=right).ratio(), 3)
