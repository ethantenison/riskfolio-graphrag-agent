"""Tests for entity resolution pipeline and metrics."""

from __future__ import annotations

from riskfolio_graphrag_agent.er.pipeline import EntityRecord, evaluate_er, run_er_pipeline


def _entities() -> list[EntityRecord]:
    return [
        EntityRecord(entity_id="e1", name="Hierarchical Risk Parity", source="docs"),
        EntityRecord(entity_id="e2", name="hierarchical-risk-parity", source="code"),
        EntityRecord(entity_id="e3", name="CVaR", source="docs"),
    ]


def test_run_er_pipeline_groups_duplicates(tmp_path):
    result = run_er_pipeline(
        _entities(),
        gold_pairs={("e1", "e2")},
        audit_dir=tmp_path,
    )

    assert len(result.canonical_entities) == 2
    assert ("e1", "e2") in result.predicted_pairs
    assert result.metrics is not None
    assert result.metrics.precision >= 0.5
    assert (tmp_path / "er_audit.json").exists()


def test_evaluate_er_zero_case():
    metrics = evaluate_er(predicted_pairs=set(), gold_pairs={("a", "b")})
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
