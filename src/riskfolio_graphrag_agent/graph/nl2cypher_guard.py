"""Provide a narrow, auditable natural-language to Cypher safety layer.

This module belongs to the graph layer but acts as a guardrail rather than a
general query engine. It translates a small allowlisted subset of natural
language prompts into read-only Cypher templates and records the resulting
decision for auditability.

Inputs are end-user questions and audit metadata. Outputs are structured guard
decisions and newline-delimited JSON audit records.

Key implementation decisions:
- the translator is template-based rather than generative so behavior remains
    deterministic and reviewable;
- unsafe write-oriented keywords are blocked before template selection;
- ambiguous questions are escalated instead of guessed.

This module does not execute Cypher, own graph retrieval ranking, or provide
unrestricted NL-to-Cypher generation.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

UNSAFE_KEYWORDS = ("create", "merge", "delete", "set ", "remove", "drop", "call dbms", "load csv")


@dataclass
class GuardDecision:
    """Describe the result of evaluating a natural-language Cypher request.

    Attributes:
        status: Decision outcome such as ``safe``, ``blocked``, or ``escalate``.
        reason: Stable machine-readable explanation for the outcome.
        cypher: Read-only Cypher statement when the request is allowed.
        params: Query parameters associated with ``cypher``.
        requires_human_review: Whether a human should review the request.
    """

    status: str
    reason: str
    cypher: str = ""
    params: dict[str, str] | None = None
    requires_human_review: bool = False


@dataclass
class QueryAuditRecord:
    """Capture one guard decision in the append-only audit log.

    Attributes:
        tenant_id: Tenant identifier associated with the request context.
        request_id: Request correlation identifier.
        question: Original natural-language question.
        decision: Structured guard outcome for the question.
        timestamp_utc: ISO 8601 UTC timestamp for the audit entry.
    """

    tenant_id: str
    request_id: str
    question: str
    decision: GuardDecision
    timestamp_utc: str


def guarded_nl_to_cypher(question: str) -> GuardDecision:
    """Translate a narrow allowlisted query into safe read-only Cypher.

    Args:
        question: Natural-language request to evaluate.

    Returns:
        A ``GuardDecision`` describing whether the request was blocked,
        escalated, or mapped to an allowlisted Cypher template.
    """
    text = question.strip()
    if not text:
        return GuardDecision(status="blocked", reason="empty_question", requires_human_review=True)

    lowered = text.lower()
    if any(keyword in lowered for keyword in UNSAFE_KEYWORDS):
        return GuardDecision(status="blocked", reason="unsafe_intent_detected", requires_human_review=True)

    count_match = re.search(r"count\s+(\w+)", lowered)
    if count_match:
        label = _safe_label(count_match.group(1))
        cypher = f"MATCH (n:{label}) RETURN count(n) AS count"
        if _is_safe_read_only(cypher):
            return GuardDecision(status="safe", reason="allowlisted_count_template", cypher=cypher, params={})

    mentions_match = re.search(r"mentions?\s+([a-zA-Z0-9_\-\s]+)", lowered)
    if mentions_match:
        entity_name = mentions_match.group(1).strip()
        cypher = (
            "MATCH (c:Chunk)-[:MENTIONS]->(e) "
            "WHERE toLower(e.name) CONTAINS toLower($entity_name) "
            "RETURN c.name AS chunk_id, c.source_path AS source_path, e.name AS entity_name LIMIT 25"
        )
        if _is_safe_read_only(cypher):
            return GuardDecision(
                status="safe",
                reason="allowlisted_mentions_template",
                cypher=cypher,
                params={"entity_name": entity_name},
            )

    return GuardDecision(status="escalate", reason="ambiguous_query", requires_human_review=True)


def append_query_audit(
    *,
    tenant_id: str,
    request_id: str,
    question: str,
    decision: GuardDecision,
    audit_path: str | Path,
) -> None:
    """Append a guard decision to the newline-delimited audit log.

    Args:
        tenant_id: Tenant identifier for multi-tenant audit partitioning.
        request_id: Correlation identifier for the originating request.
        question: Original natural-language question.
        decision: Guard outcome to persist.
        audit_path: Destination JSONL file path.
    """
    record = QueryAuditRecord(
        tenant_id=tenant_id,
        request_id=request_id,
        question=question,
        decision=decision,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )

    target = Path(audit_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_record_dict(record), sort_keys=True) + "\n")


def _is_safe_read_only(cypher: str) -> bool:
    stripped = cypher.strip().lower()
    return stripped.startswith("match") and all(keyword not in stripped for keyword in UNSAFE_KEYWORDS)


def _safe_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", label)
    if not cleaned:
        return "Concept"
    return cleaned[0].upper() + cleaned[1:]


def _record_dict(record: QueryAuditRecord) -> dict[str, object]:
    payload = asdict(record)
    payload["decision"] = asdict(record.decision)
    return payload
