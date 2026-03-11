"""Tests for NL-to-Cypher guarded translation."""

from __future__ import annotations

import json

from riskfolio_graphrag_agent.graph.nl2cypher_guard import append_query_audit, guarded_nl_to_cypher


def test_guarded_nl_to_cypher_allows_safe_count():
    decision = guarded_nl_to_cypher("count chunk")
    assert decision.status == "safe"
    assert decision.cypher.lower().startswith("match")


def test_guarded_nl_to_cypher_blocks_mutation():
    decision = guarded_nl_to_cypher("delete all nodes")
    assert decision.status == "blocked"
    assert decision.requires_human_review is True


def test_append_query_audit_writes_jsonl(tmp_path):
    decision = guarded_nl_to_cypher("count chunk")
    path = tmp_path / "audit.jsonl"
    append_query_audit(
        tenant_id="tenant-a",
        request_id="req-1",
        question="count chunk",
        decision=decision,
        audit_path=path,
    )

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["tenant_id"] == "tenant-a"
