"""Tests for riskfolio_graphrag_agent.eval.evaluator."""

from __future__ import annotations

import json

from riskfolio_graphrag_agent.eval.evaluator import (
    ContrastiveEvalReport,
    EvalReport,
    EvalSample,
    Evaluator,
    _answer_faithfulness,
    _grounding_score,
    _multi_hop_accuracy,
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


class _CandidateRetriever:
    def retrieve(self, query: str) -> list[RetrievalResult]:
        _ = query
        return [
            RetrievalResult(
                content=(
                    "Hierarchical Risk Parity uses clustering"
                    " and risk parity to allocate portfolio weights with strong grounding."
                ),
                source_path="docs/hrp.md",
                score=0.95,
                related_entities=["Hierarchical Risk Parity", "clustering", "risk parity"],
                graph_neighbours=["allocation workflow", "portfolio weights", "docs/hrp.md::chunk:2"],
                metadata={"chunk_id": "c2", "section": "HRP"},
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


def test_multi_hop_accuracy_rewards_coherent_graph_support():
    coherent_results = [
        RetrievalResult(
            content="Hierarchical Risk Parity uses clustering to allocate diversified portfolios.",
            source_path="docs/hrp.md",
            related_entities=["Hierarchical Risk Parity", "clustering"],
            graph_neighbours=["risk parity", "allocation workflow", "docs/cluster_walkthrough::chunk:2"],
        ),
        RetrievalResult(
            content="Risk parity allocation connects clustering outputs to portfolio weights.",
            source_path="docs/risk_parity.md",
            related_entities=["risk parity", "portfolio weights"],
            graph_neighbours=["Hierarchical Risk Parity", "clustering tree", "docs/hrp.md::chunk:1"],
        ),
    ]
    shallow_results = [
        RetrievalResult(
            content="General documentation about plotting.",
            source_path="docs/plots.md",
            related_entities=["plots"],
            graph_neighbours=[],
        )
    ]

    coherent_score = _multi_hop_accuracy("How does HRP connect clustering to allocation?", coherent_results)
    shallow_score = _multi_hop_accuracy("How does HRP connect clustering to allocation?", shallow_results)

    assert 0.0 <= coherent_score <= 1.0
    assert 0.0 <= shallow_score <= 1.0
    assert coherent_score > shallow_score


def test_evaluator_run_contrastive_returns_comparison_artifact():
    samples = [
        EvalSample(
            question="What is Hierarchical Risk Parity?",
            reference_answer="HRP uses clustering and risk parity.",
            expected_context_terms=["hierarchical", "risk parity", "clustering"],
        )
    ]

    evaluator = Evaluator(samples=samples, retriever=_StubRetriever())
    report = evaluator.run_contrastive(
        baseline_retriever=_StubRetriever(),
        candidate_retriever=_CandidateRetriever(),
        baseline_label="baseline-v1",
        candidate_label="candidate-v2",
    )

    assert isinstance(report, ContrastiveEvalReport)
    assert report.baseline_label == "baseline-v1"
    assert report.candidate_label == "candidate-v2"
    assert "context_recall" in report.metric_deltas
    assert report.winner in {"baseline", "candidate", "tie"}
    assert len(report.per_sample_deltas) == 1
    assert report.per_sample_deltas[0]["question"] == "What is Hierarchical Risk Parity?"


def test_evaluator_save_contrastive_writes_json(tmp_path):
    samples = [
        EvalSample(
            question="What is Hierarchical Risk Parity?",
            reference_answer="HRP uses clustering and risk parity.",
            expected_context_terms=["hierarchical", "risk parity", "clustering"],
        )
    ]

    evaluator = Evaluator(samples=samples, retriever=_StubRetriever())
    output = tmp_path / "contrastive.json"
    evaluator.save_contrastive(
        output,
        baseline_retriever=_StubRetriever(),
        candidate_retriever=_CandidateRetriever(),
    )

    assert output.exists()
    data = json.loads(output.read_text())
    assert data["baseline_label"] == "baseline"
    assert data["candidate_label"] == "candidate"
    assert "metric_deltas" in data
    assert "per_sample_deltas" in data
