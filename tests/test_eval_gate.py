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
    }
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(report))

    run_regression_gate(path, min_faithfulness=0.35, min_relevance=0.8, min_context_recall=0.45)


def test_eval_gate_fails_on_regression(tmp_path):
    report = {
        "context_recall": 0.1,
        "answer_faithfulness": 0.2,
        "answer_relevance": 0.4,
    }
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(report))

    with pytest.raises(RegressionGateError):
        run_regression_gate(path, min_faithfulness=0.35, min_relevance=0.8, min_context_recall=0.45)
