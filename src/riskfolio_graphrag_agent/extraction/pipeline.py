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

import re
from collections.abc import Sequence

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
