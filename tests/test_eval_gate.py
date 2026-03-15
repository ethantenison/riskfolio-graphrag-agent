"""Tests for evaluation regression gate."""

from __future__ import annotations

import json

import pytest

from riskfolio_graphrag_agent.eval.regression_gate import RegressionGateError, run_regression_gate


def test_eval_gate_passes(tmp_path):
    report = {
        "context_recall": 0.6,
        "answer_faithfulness": 0.45,
        "answer_relevance": 0.9,
        "grounding": 0.5,
        "multi_hop_accuracy": 0.5,
        "avg_latency_ms": 100.0,
        "estimated_cost_usd": 0.001,
    }
    path = tmp_path / "eval.json"
    trend_path = tmp_path / "trend.json"
    path.write_text(json.dumps(report))

    run_regression_gate(path, min_faithfulness=0.35, min_relevance=0.8, min_context_recall=0.45, trend_path=trend_path)

    history = json.loads(trend_path.read_text())
    assert len(history) == 1
    assert history[0]["passed"] is True
    assert history[0]["drift_flagged"] is False
    assert history[0]["failed_checks"] == []


def test_eval_gate_fails_on_regression(tmp_path):
    baseline = {
        "context_recall": 0.6,
        "answer_faithfulness": 0.45,
        "answer_relevance": 0.9,
        "grounding": 0.5,
        "multi_hop_accuracy": 0.5,
        "avg_latency_ms": 100.0,
        "estimated_cost_usd": 0.001,
    }
    report = {
        "context_recall": 0.1,
        "answer_faithfulness": 0.2,
        "answer_relevance": 0.4,
        "grounding": 0.1,
        "multi_hop_accuracy": 0.0,
        "avg_latency_ms": 9000.0,
        "estimated_cost_usd": 0.5,
    }
    path = tmp_path / "eval.json"
    baseline_path = tmp_path / "baseline.json"
    trend_path = tmp_path / "trend.json"
    baseline_path.write_text(json.dumps(baseline))
    path.write_text(json.dumps(report))

    run_regression_gate(
        baseline_path,
        min_faithfulness=0.35,
        min_relevance=0.8,
        min_context_recall=0.45,
        trend_path=trend_path,
    )

    with pytest.raises(RegressionGateError):
        run_regression_gate(
            path,
            min_faithfulness=0.35,
            min_relevance=0.8,
            min_context_recall=0.45,
            trend_path=trend_path,
        )

    history = json.loads(trend_path.read_text())
    assert len(history) == 2
    latest = history[-1]
    assert latest["passed"] is False
    assert latest["drift_flagged"] is True
    assert latest["metric_deltas"]["context_recall"] == -0.5
    assert any("answer_faithfulness" in check for check in latest["failed_checks"])
