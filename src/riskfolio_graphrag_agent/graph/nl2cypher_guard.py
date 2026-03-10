"""Guarded natural-language to Cypher translation with audit logging."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

UNSAFE_KEYWORDS = ("create", "merge", "delete", "set ", "remove", "drop", "call dbms", "load csv")


@dataclass
class GuardDecision:
    status: str
    reason: str
    cypher: str = ""
    params: dict[str, str] | None = None
    requires_human_review: bool = False


@dataclass
class QueryAuditRecord:
    tenant_id: str
    request_id: str
    question: str
    decision: GuardDecision
    timestamp_utc: str


def guarded_nl_to_cypher(question: str) -> GuardDecision:
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
