"""Tests for riskfolio_graphrag_agent.eval.evaluator."""

from __future__ import annotations

import json

from riskfolio_graphrag_agent.eval.evaluator import EvalReport, EvalSample, Evaluator


def _make_samples(n: int = 3) -> list[EvalSample]:
    return [
        EvalSample(
            question=f"Q{i}",
            reference_answer=f"A{i}",
            generated_answer=f"GA{i}",
        )
        for i in range(n)
    ]


def test_eval_report_defaults():
    report = EvalReport()
    assert report.num_samples == 0
    assert report.context_recall == 0.0


def test_evaluator_run_stub():
    """Evaluator.run stub should return an EvalReport with correct num_samples."""
    samples = _make_samples(5)
    evaluator = Evaluator(samples)
    report = evaluator.run()
    assert isinstance(report, EvalReport)
    assert report.num_samples == 5


def test_evaluator_save(tmp_path):
    """Evaluator.save should write valid JSON to the given path."""
    samples = _make_samples(2)
    evaluator = Evaluator(samples)
    output = tmp_path / "results.json"
    evaluator.save(output)
    assert output.exists()
    data = json.loads(output.read_text())
    assert data["num_samples"] == 2
