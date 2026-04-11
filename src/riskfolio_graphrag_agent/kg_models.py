"""Typed records for the redesigned knowledge graph induction pipeline.

This module defines the contracts shared across the open extraction,
canonicalization, schema induction, materialization, semantic export, and
graph-quality evaluation stages. The models are intentionally layered so the
system can preserve mention-level provenance and defer ontology commitments
until later review and promotion steps.

Inputs are chunked `Document` records from the ingestion layer plus stage-level
decisions. Outputs are stable, JSON-serializable records and pipeline result
objects that can be persisted as artifacts or written to Neo4j.

This module does not perform extraction, canonicalization, schema induction, or
graph writes by itself.
"""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum

from pydantic import BaseModel, Field

from riskfolio_graphrag_agent.ingestion.loader import Document


def slugify(value: str) -> str:
    """Return a deterministic ASCII slug for IDs and artifact keys.

    Args:
        value: Input text to normalize.

    Returns:
        Lowercase slug text made from letters, numbers, and single hyphens.
    """
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return normalized.strip("-") or "unknown"


def stable_id(kind: str, *parts: str) -> str:
    """Create a stable ID scoped by record kind.

    Args:
        kind: Record category such as ``mention`` or ``assertion``.
        parts: Values that uniquely identify the record.

    Returns:
        A deterministic identifier with a short hash suffix.
    """
    joined = "||".join(part.strip() for part in parts if part.strip())
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    prefix = slugify(parts[0] if parts else kind)
    return f"{kind}:{prefix}:{digest}"


class ReviewStatus(StrEnum):
    """Represent the lifecycle of extraction and schema decisions."""

    PROPOSED = "proposed"
    REVIEWED = "reviewed"
    PROMOTED = "promoted"
    REJECTED = "rejected"


class DecisionSource(StrEnum):
    """Track which mechanism produced a normalization decision."""

    HEURISTIC = "heuristic"
    EMBEDDING = "embedding"
    LLM = "llm"
    HUMAN = "human"
    HYBRID = "hybrid"


class SourceDocumentRecord(BaseModel):
    """Describe a source document in the KG source layer.

    Attributes:
        document_id: Stable document identifier.
        source_path: Absolute or repo-relative source path.
        source_type: Source category such as ``python`` or ``docs``.
        title: Human-readable title for review and export artifacts.
        metadata: Additional source metadata carried from ingestion.
    """

    document_id: str
    source_path: str
    source_type: str
    title: str
    metadata: dict[str, str | int | float] = Field(default_factory=dict)

    @classmethod
    def from_document(cls, document: Document) -> SourceDocumentRecord:
        """Build a source-layer record from an ingestion document.

        Args:
            document: Ingestion-layer document chunk.

        Returns:
            A source-document record detached from chunk granularity.
        """
        relative_path = str(document.metadata.get("relative_path", document.source_path))
        document_id = stable_id("document", relative_path)
        source_type = str(document.metadata.get("source_type", "unknown"))
        title = str(document.metadata.get("module_name", document.section or relative_path))
        return cls(
            document_id=document_id,
            source_path=document.source_path,
            source_type=source_type,
            title=title,
            metadata={key: value for key, value in document.metadata.items() if isinstance(value, str | int | float)},
        )


class ChunkRecord(BaseModel):
    """Describe a chunk in the source layer.

    Attributes:
        chunk_id: Stable chunk identifier from ingestion.
        document_id: Parent document identifier.
        source_path: Origin file path.
        chunk_index: Chunk offset within the source document.
        section: Human-readable source section or symbol name.
        line_start: 1-based starting line number.
        line_end: 1-based ending line number.
        content_hash: Deterministic content hash.
        content: Raw chunk content.
        metadata: Additional chunk metadata carried from ingestion.
    """

    chunk_id: str
    document_id: str
    source_path: str
    chunk_index: int
    section: str
    line_start: int
    line_end: int
    content_hash: str
    content: str
    metadata: dict[str, str | int | float] = Field(default_factory=dict)

    @classmethod
    def from_document(cls, document: Document) -> ChunkRecord:
        """Build a chunk-layer record from an ingestion document.

        Args:
            document: Ingestion-layer document chunk.

        Returns:
            A chunk record used by downstream extraction stages.
        """
        source_document = SourceDocumentRecord.from_document(document)
        chunk_id = document.chunk_id or stable_id(
            "chunk",
            str(document.metadata.get("relative_path", document.source_path)),
            str(document.chunk_index),
        )
        return cls(
            chunk_id=chunk_id,
            document_id=source_document.document_id,
            source_path=document.source_path,
            chunk_index=document.chunk_index,
            section=document.section,
            line_start=document.line_start,
            line_end=document.line_end,
            content_hash=document.content_hash or stable_id("hash", document.content)[:24],
            content=document.content,
            metadata={key: value for key, value in document.metadata.items() if isinstance(value, str | int | float)},
        )


class EvidenceSpan(BaseModel):
    """Represent a cited span of evidence inside a chunk.

    Attributes:
        evidence_id: Stable evidence identifier.
        chunk_id: Source chunk identifier.
        char_start: Inclusive start offset in the chunk.
        char_end: Exclusive end offset in the chunk.
        line_start: 1-based start line for the span when known.
        line_end: 1-based end line for the span when known.
        excerpt: Short excerpt preserved for review and provenance.
    """

    evidence_id: str
    chunk_id: str
    char_start: int
    char_end: int
    line_start: int | None = None
    line_end: int | None = None
    excerpt: str


class MentionRecord(BaseModel):
    """Represent an extracted mention before canonicalization.

    Attributes:
        mention_id: Stable mention identifier.
        chunk_id: Source chunk identifier.
        text: Surface text from the source.
        normalized_text: Lowercased lexical normalization.
        type_guess: Open-world type guess such as ``python_function`` or ``concept``.
        role: Optional extraction-local role such as ``subject`` or ``object``.
        confidence: Extraction confidence score between 0 and 1.
        evidence: Evidence span supporting the mention.
        extraction_method: Method name used to produce the mention.
        extraction_model: Optional model identifier.
        status: Review lifecycle status.
        metadata: Additional extraction metadata.
    """

    mention_id: str
    chunk_id: str
    text: str
    normalized_text: str
    type_guess: str
    role: str | None = None
    confidence: float = 0.0
    evidence: EvidenceSpan
    extraction_method: str
    extraction_model: str | None = None
    status: ReviewStatus = ReviewStatus.PROPOSED
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


class CandidateEntityRecord(BaseModel):
    """Represent a candidate entity anchored to one or more mentions.

    Attributes:
        candidate_entity_id: Stable candidate entity identifier.
        chunk_id: Chunk where the candidate was first observed.
        display_name: Preferred surface label before canonicalization.
        normalized_name: Normalized lexical form for clustering.
        type_guess: Open-world type guess.
        mention_ids: Supporting mention identifiers.
        confidence: Aggregate confidence for the candidate.
        status: Review lifecycle status.
        metadata: Additional extraction metadata.
    """

    candidate_entity_id: str
    chunk_id: str
    display_name: str
    normalized_name: str
    type_guess: str
    mention_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    status: ReviewStatus = ReviewStatus.PROPOSED
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


class CandidateAssertionRecord(BaseModel):
    """Represent a reified candidate assertion before schema promotion.

    Attributes:
        assertion_id: Stable assertion identifier.
        chunk_id: Source chunk identifier.
        subject_mention_id: Mention identifier for the subject.
        object_mention_id: Mention identifier for the object.
        relation_guess: Free-text relation guess from extraction.
        statement: Review-friendly textual rendering of the claim.
        evidence_ids: Supporting evidence span identifiers.
        confidence: Extraction confidence score.
        extraction_method: Method that produced the assertion.
        extraction_model: Optional model name.
        status: Review lifecycle status.
        metadata: Additional extraction metadata.
    """

    assertion_id: str
    chunk_id: str
    subject_mention_id: str
    object_mention_id: str
    relation_guess: str
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    extraction_method: str
    extraction_model: str | None = None
    status: ReviewStatus = ReviewStatus.PROPOSED
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


class EventArgument(BaseModel):
    """Represent one role filler for a candidate event frame.

    Attributes:
        role: Event role name such as ``actor`` or ``theme``.
        mention_id: Mention identifier filling the role.
    """

    role: str
    mention_id: str


class CandidateEventRecord(BaseModel):
    """Represent a candidate event or frame extracted from a chunk.

    Attributes:
        candidate_event_id: Stable event identifier.
        chunk_id: Source chunk identifier.
        trigger_text: Surface trigger phrase.
        event_type_guess: Open-world event/frame guess.
        arguments: Role fillers linked to mentions.
        evidence_ids: Supporting evidence identifiers.
        confidence: Extraction confidence score.
        extraction_method: Method that produced the event.
        status: Review lifecycle status.
        metadata: Additional extraction metadata.
    """

    candidate_event_id: str
    chunk_id: str
    trigger_text: str
    event_type_guess: str
    arguments: list[EventArgument] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    extraction_method: str
    status: ReviewStatus = ReviewStatus.PROPOSED
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


class OpenChunkExtraction(BaseModel):
    """Bundle source-layer and extraction-layer outputs for one chunk.

    Attributes:
        source_document: Source-document record for the chunk.
        chunk: Chunk-layer record.
        mentions: Extracted mention records.
        candidate_entities: Candidate entities derived from mentions.
        candidate_assertions: Candidate assertions derived from chunk text.
        candidate_events: Candidate events derived from chunk text.
        metadata: Run metadata for the extractor.
    """

    source_document: SourceDocumentRecord
    chunk: ChunkRecord
    mentions: list[MentionRecord] = Field(default_factory=list)
    candidate_entities: list[CandidateEntityRecord] = Field(default_factory=list)
    candidate_assertions: list[CandidateAssertionRecord] = Field(default_factory=list)
    candidate_events: list[CandidateEventRecord] = Field(default_factory=list)
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


class CanonicalizationDecision(BaseModel):
    """Describe how a candidate was mapped into a canonical cluster.

    Attributes:
        decision_id: Stable decision identifier.
        source_kind: Input category such as ``entity`` or ``predicate``.
        source_id: Identifier of the source candidate.
        canonical_id: Identifier of the selected canonical target.
        decision_source: Mechanism used to make the decision.
        confidence: Confidence score for the decision.
        rationale: Short explanation suitable for review.
        alternatives: Alternative candidate IDs or labels considered.
        status: Review lifecycle status.
    """

    decision_id: str
    source_kind: str
    source_id: str
    canonical_id: str
    decision_source: DecisionSource
    confidence: float
    rationale: str
    alternatives: list[str] = Field(default_factory=list)
    status: ReviewStatus = ReviewStatus.PROPOSED


class CanonicalEntity(BaseModel):
    """Represent a canonical entity cluster in the semantic layer.

    Attributes:
        canonical_entity_id: Stable canonical identifier.
        preferred_label: Reviewed display label.
        normalized_label: Normalized lexical form.
        type_guess: Dominant open-world type guess.
        mention_ids: Member mention identifiers.
        candidate_entity_ids: Member candidate entity identifiers.
        provenance_chunk_ids: Chunks where the entity appears.
        confidence: Aggregate confidence for the cluster.
        status: Review lifecycle status.
        metadata: Additional canonicalization metadata.
    """

    canonical_entity_id: str
    preferred_label: str
    normalized_label: str
    type_guess: str
    mention_ids: list[str] = Field(default_factory=list)
    candidate_entity_ids: list[str] = Field(default_factory=list)
    provenance_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    status: ReviewStatus = ReviewStatus.PROPOSED
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


class CanonicalPredicate(BaseModel):
    """Represent a canonical predicate proposal in the semantic layer.

    Attributes:
        canonical_predicate_id: Stable canonical predicate identifier.
        preferred_label: Reviewed predicate label.
        normalized_label: Normalized lexical form.
        relation_guesses: Relation phrases clustered into the predicate.
        assertion_ids: Assertions using this predicate.
        confidence: Aggregate confidence for the cluster.
        status: Review lifecycle status.
    """

    canonical_predicate_id: str
    preferred_label: str
    normalized_label: str
    relation_guesses: list[str] = Field(default_factory=list)
    assertion_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    status: ReviewStatus = ReviewStatus.PROPOSED


class CanonicalEventType(BaseModel):
    """Represent a canonical event-type proposal in the semantic layer.

    Attributes:
        canonical_event_type_id: Stable canonical event-type identifier.
        preferred_label: Reviewed event type label.
        normalized_label: Normalized lexical form.
        event_ids: Candidate events using this type.
        confidence: Aggregate confidence for the cluster.
        status: Review lifecycle status.
    """

    canonical_event_type_id: str
    preferred_label: str
    normalized_label: str
    event_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    status: ReviewStatus = ReviewStatus.PROPOSED


class CanonicalizationResult(BaseModel):
    """Bundle canonicalization outputs across entities, predicates, and events."""

    canonical_entities: list[CanonicalEntity] = Field(default_factory=list)
    canonical_predicates: list[CanonicalPredicate] = Field(default_factory=list)
    canonical_event_types: list[CanonicalEventType] = Field(default_factory=list)
    decisions: list[CanonicalizationDecision] = Field(default_factory=list)


class SchemaInductionCandidate(BaseModel):
    """Represent a reviewable schema candidate aggregated from open extraction.

    Attributes:
        candidate_id: Stable schema candidate identifier.
        candidate_kind: Candidate category such as ``entity_type`` or ``relation_type``.
        proposed_label: Proposed canonical schema label.
        lexical_variants: Surface variants clustered under the label.
        supporting_ids: Supporting entity, assertion, or event identifiers.
        support_count: Number of supporting observations.
        confidence: Aggregate confidence score.
        status: Review lifecycle status.
        notes: Review-oriented notes.
    """

    candidate_id: str
    candidate_kind: str
    proposed_label: str
    lexical_variants: list[str] = Field(default_factory=list)
    supporting_ids: list[str] = Field(default_factory=list)
    support_count: int = 0
    confidence: float = 0.0
    status: ReviewStatus = ReviewStatus.PROPOSED
    notes: str = ""


class OntologyClass(BaseModel):
    """Represent a stabilized ontology class in the ontology layer.

    Attributes:
        ontology_class_id: Stable ontology class identifier.
        label: Canonical class label.
        definition: Short class description.
        source_candidate_ids: Schema candidates supporting the class.
        confidence: Confidence score for stabilization.
        status: Review lifecycle status.
    """

    ontology_class_id: str
    label: str
    definition: str
    source_candidate_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    status: ReviewStatus = ReviewStatus.PROPOSED


class OntologyProperty(BaseModel):
    """Represent a stabilized ontology property in the ontology layer.

    Attributes:
        ontology_property_id: Stable ontology property identifier.
        label: Canonical property label.
        definition: Short property description.
        source_candidate_ids: Schema candidates supporting the property.
        confidence: Confidence score for stabilization.
        status: Review lifecycle status.
    """

    ontology_property_id: str
    label: str
    definition: str
    source_candidate_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    status: ReviewStatus = ReviewStatus.PROPOSED


class ConceptScheme(BaseModel):
    """Represent an emergent SKOS concept scheme or taxonomy bucket.

    Attributes:
        concept_scheme_id: Stable concept-scheme identifier.
        label: Human-readable scheme label.
        concept_ids: Ontology class or property identifiers under the scheme.
        status: Review lifecycle status.
    """

    concept_scheme_id: str
    label: str
    concept_ids: list[str] = Field(default_factory=list)
    status: ReviewStatus = ReviewStatus.PROPOSED


class SchemaInductionResult(BaseModel):
    """Bundle reviewable schema proposals and stabilized ontology elements."""

    candidates: list[SchemaInductionCandidate] = Field(default_factory=list)
    ontology_classes: list[OntologyClass] = Field(default_factory=list)
    ontology_properties: list[OntologyProperty] = Field(default_factory=list)
    concept_schemes: list[ConceptScheme] = Field(default_factory=list)
    review_markdown: str = ""


class MaterializedNode(BaseModel):
    """Represent a final promoted property-graph node ready for Neo4j writes.

    Attributes:
        node_id: Stable node identifier.
        label: Single operational Neo4j label.
        properties: Node properties persisted to Neo4j.
    """

    node_id: str
    label: str
    properties: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)


class MaterializedEdge(BaseModel):
    """Represent a final promoted property-graph edge ready for Neo4j writes.

    Attributes:
        source_id: Source node identifier.
        target_id: Target node identifier.
        relationship_type: Neo4j relationship type.
        properties: Edge properties persisted to Neo4j.
    """

    source_id: str
    target_id: str
    relationship_type: str
    properties: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)


class GraphWritePlan(BaseModel):
    """Describe a Neo4j write plan and representative retrieval queries.

    Attributes:
        constraints: Constraint and index statements.
        nodes: Materialized nodes ready for writes.
        edges: Materialized edges ready for writes.
        retrieval_queries: Representative Cypher queries used by retrieval.
    """

    constraints: list[str] = Field(default_factory=list)
    nodes: list[MaterializedNode] = Field(default_factory=list)
    edges: list[MaterializedEdge] = Field(default_factory=list)
    retrieval_queries: dict[str, str] = Field(default_factory=dict)


class SemanticExportResult(BaseModel):
    """Represent semantic-web exports produced from the stabilized KG."""

    ontology_turtle: str
    instances_turtle: str
    summary: dict[str, int | str | float] = Field(default_factory=dict)


class GraphQualityReport(BaseModel):
    """Summarize graph-quality metrics over the redesigned pipeline."""

    num_chunks: int = 0
    num_mentions: int = 0
    num_candidate_entities: int = 0
    num_candidate_assertions: int = 0
    num_candidate_events: int = 0
    num_canonical_entities: int = 0
    num_canonical_predicates: int = 0
    num_ontology_classes: int = 0
    num_ontology_properties: int = 0
    promoted_assertion_ratio: float = 0.0
    entity_compression_ratio: float = 0.0
    mean_assertion_confidence: float = 0.0
    mean_canonical_entity_confidence: float = 0.0
    schema_support_ratio: float = 0.0


class KnowledgeGraphRunResult(BaseModel):
    """Aggregate the outputs of one end-to-end KG induction run.

    Attributes:
        extractions: Chunk-level extraction outputs.
        canonicalization: Canonicalization outputs.
        schema_induction: Schema induction outputs.
        write_plan: Final materialized graph write plan.
        semantic_export: Semantic export output.
        graph_quality: Graph-quality metrics.
        artifact_paths: Paths written during the run.
    """

    extractions: list[OpenChunkExtraction] = Field(default_factory=list)
    canonicalization: CanonicalizationResult = Field(default_factory=CanonicalizationResult)
    schema_induction: SchemaInductionResult = Field(default_factory=SchemaInductionResult)
    write_plan: GraphWritePlan = Field(default_factory=GraphWritePlan)
    semantic_export: SemanticExportResult | None = None
    graph_quality: GraphQualityReport = Field(default_factory=GraphQualityReport)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
