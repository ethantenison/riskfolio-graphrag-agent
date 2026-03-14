"""Tests for riskfolio_graphrag_agent.eval.evaluator."""

from __future__ import annotations

import json

from riskfolio_graphrag_agent.eval.evaluator import (
    EvalReport,
    EvalSample,
    Evaluator,
    _answer_faithfulness,
    _grounding_score,
)
from riskfolio_graphrag_agent.retrieval.retriever import RetrievalResult


class _StubRetriever:
    def retrieve(self, query: str) -> list[RetrievalResult]:
        _ = query
        return [
            RetrievalResult(
                content=("Hierarchical Risk Parity uses clustering and risk parity to allocate portfolio weights."),
                source_path="docs/hrp.md",
                score=0.9,
                related_entities=["HRP", "clustering", "risk parity"],
                metadata={"chunk_id": "c1", "section": "HRP"},
            )
        ]


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


def test_evaluator_default_profile_is_ragas_style():
    samples = [
        EvalSample(
            question="What is Hierarchical Risk Parity?",
            reference_answer="HRP uses clustering and risk parity.",
            expected_context_terms=["hierarchical", "risk parity", "clustering"],
        )
    ]

    evaluator = Evaluator(samples=samples, retriever=_StubRetriever())
    report = evaluator.run()

    assert report.metric_profile == "ragas-style"
    assert report.num_samples == 1
    assert 0.0 <= report.context_recall <= 1.0
    assert 0.0 <= report.context_precision <= 1.0
    assert 0.0 <= report.answer_faithfulness <= 1.0
    assert 0.0 <= report.answer_relevance <= 1.0


def test_evaluator_accepts_heuristic_profile():
    samples = [
        EvalSample(
            question="What is Hierarchical Risk Parity?",
            reference_answer="HRP uses clustering and risk parity.",
            expected_context_terms=["hierarchical", "risk parity", "clustering"],
        )
    ]

    evaluator = Evaluator(
        samples=samples,
        retriever=_StubRetriever(),
        metric_profile="heuristic",
    )
    report = evaluator.run()

    assert report.metric_profile == "heuristic"
    assert report.num_samples == 1


def test_evaluator_save(tmp_path):
    """Evaluator.save should write valid JSON to the given path."""
    samples = _make_samples(2)
    evaluator = Evaluator(samples)
    output = tmp_path / "results.json"
    evaluator.save(output)
    assert output.exists()
    data = json.loads(output.read_text())
    assert data["num_samples"] == 2


def test_grounding_is_distinct_from_faithfulness():
    answer = "Regarding HRP: clustering organizes assets. Key entities: HRP, clustering."
    contexts = [
        "clustering organizes assets for HRP portfolios",
        "tail context with extra repeated HRP clustering terms and unrelated filler",
        "more unrelated filler terms for retrieval depth",
    ]

    faithfulness = _answer_faithfulness(answer, contexts)
    grounding = _grounding_score(answer, contexts)

    assert 0.0 <= grounding <= 1.0
    assert grounding != faithfulness


def test_evaluator_reports_failure_reasons_when_no_contexts():
    samples = [
        EvalSample(
            question="What is HRP?",
            reference_answer="HRP is a portfolio method.",
            expected_context_terms=["hrp", "portfolio"],
        )
    ]

    evaluator = Evaluator(samples=samples, retriever=None)
    report = evaluator.run()

    assert report.num_samples == 1
    failure_reasons = report.per_sample[0]["failure_reasons"]
    assert isinstance(failure_reasons, list)
    assert "low_context_recall" in failure_reasons
    assert "low_grounding" in failure_reasons
