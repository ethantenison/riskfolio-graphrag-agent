"""Open extraction for chunks in the redesigned KG induction pipeline.

This module converts ingestion-layer `Document` chunks into source records,
mention records, candidate entities, candidate assertions, and candidate event
frames. The implementation intentionally keeps extraction open-world: it emits
free-text type and relation guesses rather than forcing early alignment to a
fixed ontology.

Inputs are `Document` chunks. Outputs are `OpenChunkExtraction` records with
evidence spans, confidence, and extraction metadata suitable for later review.

This module does not canonicalize entities, stabilize schema, or write Neo4j.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, Field

from riskfolio_graphrag_agent.ingestion.loader import Document
from riskfolio_graphrag_agent.kg_models import (
    CandidateAssertionRecord,
    CandidateEntityRecord,
    CandidateEventRecord,
    ChunkRecord,
    EventArgument,
    EvidenceSpan,
    MentionRecord,
    OpenChunkExtraction,
    ReviewStatus,
    SourceDocumentRecord,
    stable_id,
)

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_TITLE_CASE_PATTERN = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}|[A-Z]{2,}(?:-[A-Z]{2,})?)\b")
_MIXED_CASE_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9]{1,24}[A-Z][A-Za-z0-9]{0,24}\b")
_CODE_REF_PATTERN = re.compile(r"`([^`]{2,80})`")
_CLASS_PATTERN = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_FUNCTION_PATTERN = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", re.MULTILINE)
_RELATION_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"\b(?P<subject>[A-Z][\w\-. ]{1,80}?)\s+uses\s+(?P<object>[A-Z][\w\-. ]{1,80})\b"),
        "uses",
        "usage",
    ),
    (
        re.compile(r"\b(?P<subject>[A-Z][\w\-. ]{1,80}?)\s+supports\s+(?P<object>[A-Z][\w\-. ]{1,80})\b"),
        "supports",
        "support",
    ),
    (
        re.compile(r"\b(?P<subject>[A-Z][\w\-. ]{1,80}?)\s+implements\s+(?P<object>[A-Z][\w\-. ]{1,80})\b"),
        "implements",
        "implementation",
    ),
    (
        re.compile(r"\b(?P<subject>[A-Z][\w\-. ]{1,80}?)\s+is\s+(?:an?\s+)?(?P<object>[A-Za-z][\w\-. ]{1,80})\b"),
        "is",
        "typing",
    ),
)


class ChunkOpenExtractorProtocol(Protocol):
    """Protocol for chunk-level open extractors used by the KG pipeline."""

    def extract_documents(self, documents: Sequence[Document]) -> list[OpenChunkExtraction]: ...

    def extract_chunk(self, document: Document) -> OpenChunkExtraction: ...


class LLMOpenExtractProtocol(Protocol):
    """Callable contract for chunk-level LLM open extraction.

    Implementations must return a JSON-compatible dictionary matching the
    ``LLMExtractionPayload`` schema.
    """

    def __call__(
        self,
        *,
        content: str,
        source_type: str,
        model_name: str,
    ) -> dict[str, object]: ...


class LLMAssertionPayload(BaseModel):
    """Structured JSON schema for one LLM-backed candidate assertion.

    Attributes:
        subject_text: Surface text for the assertion subject mention.
        subject_type_guess: Optional open-world type guess for the subject.
        predicate_text: Surface relation phrase linking subject and object.
        object_text: Surface text for the assertion object mention.
        object_type_guess: Optional open-world type guess for the object.
        statement: Optional review-friendly textual rendering of the claim.
        evidence_text: Optional supporting excerpt from the chunk.
        confidence: Model confidence between 0 and 1.
        metadata: Optional scalar metadata for diagnostics.
    """

    subject_text: str
    subject_type_guess: str = ""
    predicate_text: str
    object_text: str
    object_type_guess: str = ""
    statement: str = ""
    evidence_text: str = ""
    confidence: float = 0.0
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class LLMEventArgumentPayload(BaseModel):
    """Structured JSON schema for one argument inside an LLM-backed event.

    Attributes:
        role: Event role such as ``subject`` or ``object``.
        text: Surface text for the argument mention.
        type_guess: Optional open-world type guess for the argument.
    """

    role: str
    text: str
    type_guess: str = ""


class LLMEventPayload(BaseModel):
    """Structured JSON schema for one LLM-backed candidate event.

    Attributes:
        trigger_text: Trigger phrase for the event or frame.
        event_type_guess: Open-world event/frame label.
        arguments: Role-filled arguments referencing surface text.
        evidence_text: Optional supporting excerpt from the chunk.
        confidence: Model confidence between 0 and 1.
        metadata: Optional scalar metadata for diagnostics.
    """

    trigger_text: str
    event_type_guess: str
    arguments: list[LLMEventArgumentPayload] = Field(default_factory=list)
    evidence_text: str = ""
    confidence: float = 0.0
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class LLMExtractionPayload(BaseModel):
    """Top-level JSON schema returned by the LLM open extractor.

    Attributes:
        candidate_assertions: Proposed evidence-grounded assertions.
        candidate_events: Proposed evidence-grounded event frames.
    """

    candidate_assertions: list[LLMAssertionPayload] = Field(default_factory=list)
    candidate_events: list[LLMEventPayload] = Field(default_factory=list)


class HeuristicOpenExtractor:
    """Produce open extraction artifacts without a predefined ontology.

    The current implementation is deliberately simple but structurally honest:
    it yields mention-level and assertion-level records with provenance and
    free-text semantic guesses. Higher-precision LLM or model-backed extractors
    can replace this class without changing downstream contracts.
    """

    def __init__(self, model_name: str = "heuristic-open-extractor") -> None:
        """Initialize the extractor.

        Args:
            model_name: Human-readable extraction model name recorded in metadata.
        """
        self._model_name = model_name

    def extract_documents(self, documents: Sequence[Document]) -> list[OpenChunkExtraction]:
        """Extract open KG artifacts from multiple chunks.

        Args:
            documents: Chunked ingestion documents.

        Returns:
            Chunk-level open extraction records.
        """
        return [self.extract_chunk(document) for document in documents]

    def extract_chunk(self, document: Document) -> OpenChunkExtraction:
        """Extract mentions, entities, assertions, and events from one chunk.

        Args:
            document: Chunked source document.

        Returns:
            A structured open extraction bundle.
        """
        source_document = SourceDocumentRecord.from_document(document)
        chunk = ChunkRecord.from_document(document)
        mentions = self._extract_mentions(chunk)
        mention_map = {mention.normalized_text: mention for mention in mentions}
        candidate_entities = [self._candidate_from_mention(mention) for mention in mentions]
        candidate_assertions, candidate_events = self._extract_assertions_and_events(chunk, mention_map)
        return OpenChunkExtraction(
            source_document=source_document,
            chunk=chunk,
            mentions=mentions,
            candidate_entities=candidate_entities,
            candidate_assertions=candidate_assertions,
            candidate_events=candidate_events,
            metadata={
                "extractor": self._model_name,
                "source_type": str(chunk.metadata.get("source_type", "unknown")),
            },
        )

    def _extract_mentions(self, chunk: ChunkRecord) -> list[MentionRecord]:
        mentions: list[MentionRecord] = []
        seen_offsets: set[tuple[int, int, str]] = set()
        content = chunk.content

        for match in _CLASS_PATTERN.finditer(content):
            mentions.append(self._build_mention(chunk, match.group(1), match.start(1), match.end(1), "python_class", 0.92))

        for match in _FUNCTION_PATTERN.finditer(content):
            mentions.append(self._build_mention(chunk, match.group(1), match.start(1), match.end(1), "python_function", 0.9))
            raw_params = [part.strip() for part in match.group(2).split(",") if part.strip()]
            for param in raw_params[:6]:
                param_name = param.split("=")[0].split(":")[0].strip()
                if not param_name:
                    continue
                offset = content.find(param_name, match.start(2), match.end(2))
                if offset >= 0:
                    mentions.append(
                        self._build_mention(
                            chunk,
                            param_name,
                            offset,
                            offset + len(param_name),
                            "python_parameter",
                            0.78,
                        )
                    )

        for match in _CODE_REF_PATTERN.finditer(content):
            text = match.group(1).strip()
            mention = self._build_mention(chunk, text, match.start(1), match.end(1), self._guess_type(text), 0.72)
            key = (mention.evidence.char_start, mention.evidence.char_end, mention.normalized_text)
            if key not in seen_offsets:
                seen_offsets.add(key)
                mentions.append(mention)

        for match in _TITLE_CASE_PATTERN.finditer(content):
            text = match.group(0).strip()
            if len(text) < 3:
                continue
            mention = self._build_mention(chunk, text, match.start(), match.end(), self._guess_type(text), 0.66)
            key = (mention.evidence.char_start, mention.evidence.char_end, mention.normalized_text)
            if key not in seen_offsets:
                seen_offsets.add(key)
                mentions.append(mention)

        for match in _MIXED_CASE_PATTERN.finditer(content):
            text = match.group(0).strip()
            mention = self._build_mention(chunk, text, match.start(), match.end(), self._guess_type(text), 0.7)
            key = (mention.evidence.char_start, mention.evidence.char_end, mention.normalized_text)
            if key not in seen_offsets:
                seen_offsets.add(key)
                mentions.append(mention)

        deduped: dict[str, MentionRecord] = {}
        for mention in mentions:
            deduped.setdefault(mention.mention_id, mention)
        return list(deduped.values())

    def _extract_assertions_and_events(
        self,
        chunk: ChunkRecord,
        mention_map: dict[str, MentionRecord],
    ) -> tuple[list[CandidateAssertionRecord], list[CandidateEventRecord]]:
        assertions: list[CandidateAssertionRecord] = []
        events: list[CandidateEventRecord] = []
        for sentence in _SENTENCE_SPLIT.split(chunk.content):
            sentence = sentence.strip()
            if not sentence:
                continue
            for pattern, relation_guess, event_type_guess in _RELATION_PATTERNS:
                match = pattern.search(sentence)
                if match is None:
                    continue
                subject_text = match.group("subject").strip()
                object_text = match.group("object").strip()
                subject_mention = mention_map.get(subject_text.casefold())
                object_mention = mention_map.get(object_text.casefold())
                if subject_mention is None or object_mention is None:
                    continue
                sentence_start = chunk.content.find(sentence)
                sentence_end = sentence_start + len(sentence)
                evidence = self._build_evidence(chunk, sentence, sentence_start, sentence_end)
                assertion_id = stable_id(
                    "assertion",
                    chunk.chunk_id,
                    subject_mention.mention_id,
                    relation_guess,
                    object_mention.mention_id,
                )
                assertions.append(
                    CandidateAssertionRecord(
                        assertion_id=assertion_id,
                        chunk_id=chunk.chunk_id,
                        subject_mention_id=subject_mention.mention_id,
                        object_mention_id=object_mention.mention_id,
                        relation_guess=relation_guess,
                        statement=f"{subject_text} {relation_guess} {object_text}",
                        evidence_ids=[evidence.evidence_id],
                        confidence=0.68,
                        extraction_method=self._model_name,
                        extraction_model=self._model_name,
                        status=ReviewStatus.PROPOSED,
                        metadata={"sentence": sentence},
                    )
                )
                events.append(
                    CandidateEventRecord(
                        candidate_event_id=stable_id("event", chunk.chunk_id, relation_guess, sentence),
                        chunk_id=chunk.chunk_id,
                        trigger_text=relation_guess,
                        event_type_guess=event_type_guess,
                        arguments=[
                            EventArgument(role="subject", mention_id=subject_mention.mention_id),
                            EventArgument(role="object", mention_id=object_mention.mention_id),
                        ],
                        evidence_ids=[evidence.evidence_id],
                        confidence=0.61,
                        extraction_method=self._model_name,
                        status=ReviewStatus.PROPOSED,
                        metadata={"sentence": sentence},
                    )
                )
        return assertions, events

    def _candidate_from_mention(self, mention: MentionRecord) -> CandidateEntityRecord:
        return CandidateEntityRecord(
            candidate_entity_id=stable_id("candidate-entity", mention.chunk_id, mention.normalized_text),
            chunk_id=mention.chunk_id,
            display_name=mention.text,
            normalized_name=mention.normalized_text,
            type_guess=mention.type_guess,
            mention_ids=[mention.mention_id],
            confidence=mention.confidence,
            status=ReviewStatus.PROPOSED,
            metadata={"extraction_method": mention.extraction_method},
        )

    def _build_mention(
        self,
        chunk: ChunkRecord,
        text: str,
        char_start: int,
        char_end: int,
        type_guess: str,
        confidence: float,
    ) -> MentionRecord:
        evidence = self._build_evidence(chunk, text, char_start, char_end)
        normalized_text = text.casefold().strip()
        return MentionRecord(
            mention_id=stable_id("mention", chunk.chunk_id, normalized_text, str(char_start), str(char_end)),
            chunk_id=chunk.chunk_id,
            text=text,
            normalized_text=normalized_text,
            type_guess=type_guess,
            confidence=confidence,
            evidence=evidence,
            extraction_method=self._model_name,
            extraction_model=self._model_name,
            status=ReviewStatus.PROPOSED,
            metadata={"section": chunk.section},
        )

    def _build_evidence(self, chunk: ChunkRecord, excerpt: str, char_start: int, char_end: int) -> EvidenceSpan:
        start = max(0, char_start)
        end = max(start, char_end)
        return EvidenceSpan(
            evidence_id=stable_id("evidence", chunk.chunk_id, str(start), str(end), excerpt[:48]),
            chunk_id=chunk.chunk_id,
            char_start=start,
            char_end=end,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            excerpt=excerpt[:280],
        )

    def _guess_type(self, text: str) -> str:
        lowered = text.casefold()
        if "model" in lowered:
            return "model"
        if "measure" in lowered or lowered.endswith("var"):
            return "risk_measure_like"
        if text.isupper() and len(text) <= 8:
            return "acronym_concept"
        if "." in text or "_" in text:
            return "api_symbol"
        return "concept"


class LLMOpenExtractor:
    """Enrich heuristic extraction with LLM-backed assertions and events.

    The extractor preserves the heuristic mention/entity base layer, then adds
    candidate assertions and event frames produced by an LLM under the strict
    ``LLMExtractionPayload`` JSON schema. This keeps the pipeline auditable:
    the LLM proposes evidence-grounded claims, while downstream stages still
    canonicalize, induce schema, and materialize the promoted graph.
    """

    def __init__(
        self,
        llm_extract: LLMOpenExtractProtocol,
        model_name: str = "llm-open-extractor",
        fallback_extractor: HeuristicOpenExtractor | None = None,
    ) -> None:
        """Initialize the hybrid extractor.

        Args:
            llm_extract: Callable that returns JSON matching
                ``LLMExtractionPayload``.
            model_name: Human-readable model identifier for metadata.
            fallback_extractor: Base extractor used for mentions/entities and as
                a safe fallback when the LLM call fails.
        """
        self._llm_extract = llm_extract
        self._model_name = model_name
        self._fallback_extractor = fallback_extractor or HeuristicOpenExtractor()

    def extract_documents(self, documents: Sequence[Document]) -> list[OpenChunkExtraction]:
        """Extract open KG artifacts from multiple chunks.

        Args:
            documents: Chunked ingestion documents.

        Returns:
            Chunk-level open extraction records.
        """
        return [self.extract_chunk(document) for document in documents]

    def extract_chunk(self, document: Document) -> OpenChunkExtraction:
        """Extract mentions, entities, assertions, and events from one chunk.

        Args:
            document: Chunked source document.

        Returns:
            A structured open extraction bundle enriched by the LLM.
        """
        extraction = self._fallback_extractor.extract_chunk(document)
        source_type = str(extraction.chunk.metadata.get("source_type", "unknown"))
        try:
            payload = self._llm_extract(
                content=extraction.chunk.content,
                source_type=source_type,
                model_name=self._model_name,
            )
            parsed = LLMExtractionPayload.model_validate(payload)
        except Exception as exc:
            logger.warning("LLM open extraction failed for chunk %s: %s", extraction.chunk.chunk_id, exc)
            return extraction

        mention_index = {mention.normalized_text: mention for mention in extraction.mentions}
        candidate_entity_index = {candidate.normalized_name: candidate for candidate in extraction.candidate_entities}
        existing_assertion_ids = {assertion.assertion_id for assertion in extraction.candidate_assertions}
        existing_assertion_keys = {
            (
                assertion.subject_mention_id,
                assertion.relation_guess.casefold(),
                assertion.object_mention_id,
            )
            for assertion in extraction.candidate_assertions
        }
        existing_event_ids = {event.candidate_event_id for event in extraction.candidate_events}
        existing_event_keys = {
            (
                event.event_type_guess.casefold(),
                tuple(sorted((argument.role, argument.mention_id) for argument in event.arguments)),
            )
            for event in extraction.candidate_events
        }

        for candidate_assertion in parsed.candidate_assertions:
            self._append_assertion(
                extraction=extraction,
                payload=candidate_assertion,
                mention_index=mention_index,
                candidate_entity_index=candidate_entity_index,
                existing_assertion_ids=existing_assertion_ids,
                existing_assertion_keys=existing_assertion_keys,
            )

        for candidate_event in parsed.candidate_events:
            self._append_event(
                extraction=extraction,
                payload=candidate_event,
                mention_index=mention_index,
                candidate_entity_index=candidate_entity_index,
                existing_event_ids=existing_event_ids,
                existing_event_keys=existing_event_keys,
            )

        extraction.metadata["extractor"] = self._model_name
        extraction.metadata["llm_enriched"] = 1
        return extraction

    def _append_assertion(
        self,
        *,
        extraction: OpenChunkExtraction,
        payload: LLMAssertionPayload,
        mention_index: dict[str, MentionRecord],
        candidate_entity_index: dict[str, CandidateEntityRecord],
        existing_assertion_ids: set[str],
        existing_assertion_keys: set[tuple[str, str, str]],
    ) -> None:
        relation_guess = payload.predicate_text.strip()
        subject_text = payload.subject_text.strip()
        object_text = payload.object_text.strip()
        if not relation_guess or not subject_text or not object_text:
            return

        subject_mention = self._ensure_mention(
            extraction=extraction,
            mention_index=mention_index,
            candidate_entity_index=candidate_entity_index,
            text=subject_text,
            type_guess=payload.subject_type_guess,
            confidence=payload.confidence,
        )
        object_mention = self._ensure_mention(
            extraction=extraction,
            mention_index=mention_index,
            candidate_entity_index=candidate_entity_index,
            text=object_text,
            type_guess=payload.object_type_guess,
            confidence=payload.confidence,
        )
        evidence_excerpt = (
            payload.evidence_text.strip()
            or payload.statement.strip()
            or (f"{subject_mention.text} {relation_guess} {object_mention.text}")
        )
        evidence = self._resolve_evidence(extraction.chunk, evidence_excerpt)
        assertion_key = (
            subject_mention.mention_id,
            relation_guess.casefold(),
            object_mention.mention_id,
        )
        if assertion_key in existing_assertion_keys:
            return
        assertion_id = stable_id(
            "assertion",
            extraction.chunk.chunk_id,
            subject_mention.mention_id,
            relation_guess,
            object_mention.mention_id,
            evidence.evidence_id,
        )
        if assertion_id in existing_assertion_ids:
            return

        extraction.candidate_assertions.append(
            CandidateAssertionRecord(
                assertion_id=assertion_id,
                chunk_id=extraction.chunk.chunk_id,
                subject_mention_id=subject_mention.mention_id,
                object_mention_id=object_mention.mention_id,
                relation_guess=relation_guess,
                statement=payload.statement.strip() or evidence_excerpt,
                evidence_ids=[evidence.evidence_id],
                confidence=round(max(0.0, min(1.0, payload.confidence or 0.0)), 3),
                extraction_method=self._model_name,
                extraction_model=self._model_name,
                status=ReviewStatus.PROPOSED,
                metadata=self._scalar_metadata(payload.metadata),
            )
        )
        existing_assertion_ids.add(assertion_id)
        existing_assertion_keys.add(assertion_key)

    def _append_event(
        self,
        *,
        extraction: OpenChunkExtraction,
        payload: LLMEventPayload,
        mention_index: dict[str, MentionRecord],
        candidate_entity_index: dict[str, CandidateEntityRecord],
        existing_event_ids: set[str],
        existing_event_keys: set[tuple[str, tuple[tuple[str, str], ...]]],
    ) -> None:
        trigger_text = payload.trigger_text.strip()
        event_type_guess = payload.event_type_guess.strip()
        if not trigger_text or not event_type_guess:
            return

        event_arguments: list[EventArgument] = []
        for argument in payload.arguments:
            argument_text = argument.text.strip()
            if not argument_text:
                continue
            mention = self._ensure_mention(
                extraction=extraction,
                mention_index=mention_index,
                candidate_entity_index=candidate_entity_index,
                text=argument_text,
                type_guess=argument.type_guess,
                confidence=payload.confidence,
            )
            event_arguments.append(EventArgument(role=argument.role.strip() or "argument", mention_id=mention.mention_id))

        if not event_arguments:
            return

        evidence_excerpt = payload.evidence_text.strip() or trigger_text
        evidence = self._resolve_evidence(extraction.chunk, evidence_excerpt)
        event_key = (
            event_type_guess.casefold(),
            tuple(sorted((argument.role, argument.mention_id) for argument in event_arguments)),
        )
        if event_key in existing_event_keys:
            return
        candidate_event_id = stable_id(
            "event",
            extraction.chunk.chunk_id,
            event_type_guess,
            trigger_text,
            evidence.evidence_id,
        )
        if candidate_event_id in existing_event_ids:
            return

        extraction.candidate_events.append(
            CandidateEventRecord(
                candidate_event_id=candidate_event_id,
                chunk_id=extraction.chunk.chunk_id,
                trigger_text=trigger_text,
                event_type_guess=event_type_guess,
                arguments=event_arguments,
                evidence_ids=[evidence.evidence_id],
                confidence=round(max(0.0, min(1.0, payload.confidence or 0.0)), 3),
                extraction_method=self._model_name,
                status=ReviewStatus.PROPOSED,
                metadata=self._scalar_metadata(payload.metadata),
            )
        )
        existing_event_ids.add(candidate_event_id)
        existing_event_keys.add(event_key)

    def _ensure_mention(
        self,
        *,
        extraction: OpenChunkExtraction,
        mention_index: dict[str, MentionRecord],
        candidate_entity_index: dict[str, CandidateEntityRecord],
        text: str,
        type_guess: str,
        confidence: float,
    ) -> MentionRecord:
        normalized_text = text.casefold().strip()
        existing = mention_index.get(normalized_text)
        if existing is not None:
            return existing

        content_lower = extraction.chunk.content.casefold()
        char_start = content_lower.find(normalized_text)
        if char_start < 0:
            char_start = 0
        char_end = min(len(extraction.chunk.content), char_start + len(text))
        mention = self._fallback_extractor._build_mention(
            extraction.chunk,
            text,
            char_start,
            char_end,
            type_guess or self._fallback_extractor._guess_type(text),
            max(0.55, confidence or 0.0),
        ).model_copy(
            update={
                "extraction_method": self._model_name,
                "extraction_model": self._model_name,
                "type_guess": type_guess or self._fallback_extractor._guess_type(text),
            }
        )
        extraction.mentions.append(mention)
        mention_index[normalized_text] = mention

        if normalized_text not in candidate_entity_index:
            candidate_entity = self._fallback_extractor._candidate_from_mention(mention).model_copy(
                update={
                    "type_guess": mention.type_guess,
                    "confidence": mention.confidence,
                    "metadata": {"extraction_method": self._model_name},
                }
            )
            extraction.candidate_entities.append(candidate_entity)
            candidate_entity_index[normalized_text] = candidate_entity

        return mention

    def _resolve_evidence(self, chunk: ChunkRecord, excerpt: str) -> EvidenceSpan:
        excerpt_text = excerpt.strip() or chunk.content[:120]
        content_lower = chunk.content.casefold()
        excerpt_lower = excerpt_text.casefold()
        char_start = content_lower.find(excerpt_lower)
        if char_start < 0:
            char_start = 0
        char_end = min(len(chunk.content), char_start + len(excerpt_text))
        return self._fallback_extractor._build_evidence(chunk, excerpt_text, char_start, char_end)

    def _scalar_metadata(self, metadata: dict[str, str | int | float | bool]) -> dict[str, str | int | float]:
        normalized: dict[str, str | int | float] = {}
        for key, value in metadata.items():
            if isinstance(value, bool):
                normalized[key] = int(value)
            elif isinstance(value, str | int | float):
                normalized[key] = value
        return normalized
