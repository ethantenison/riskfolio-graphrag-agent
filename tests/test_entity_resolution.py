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


def test_cvar_alias_resolves_to_same_canonical():
    """CVaR and 'conditional value at risk' must resolve to the same canonical bucket (gold pair e3/e4)."""
    entities = [
        EntityRecord(entity_id="e3", name="CVaR", source="docs"),
        EntityRecord(entity_id="e4", name="conditional value at risk", source="code"),
    ]
    gold_pairs: set[tuple[str, str]] = {("e3", "e4")}
    result = run_er_pipeline(entities, gold_pairs=gold_pairs)

    assert ("e3", "e4") in result.predicted_pairs, (
        "CVaR and 'conditional value at risk' were not merged into the same canonical entity. "
        f"Canonical entities: {[c.canonical_name for c in result.canonical_entities]}"
    )
    assert result.metrics is not None
    assert result.metrics.recall == 1.0, f"ER recall should be 1.0, got {result.metrics.recall}"


def test_expected_shortfall_alias_resolves_to_cvar():
    """'expected shortfall' is an alias for CVaR — it must group with CVaR."""
    entities = [
        EntityRecord(entity_id="e5", name="CVaR", source="docs"),
        EntityRecord(entity_id="e6", name="expected shortfall", source="code"),
    ]
    result = run_er_pipeline(entities, gold_pairs={("e5", "e6")})
    assert ("e5", "e6") in result.predicted_pairs
