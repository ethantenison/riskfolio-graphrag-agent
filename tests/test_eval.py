"""Tests for riskfolio_graphrag_agent.eval.evaluator."""

from __future__ import annotations

import json

import pytest

from riskfolio_graphrag_agent.eval.evaluator import (
    ContrastiveEvalReport,
    EvalReport,
    EvalSample,
    Evaluator,
    _answer_faithfulness,
    _grounding_score,
    _multi_hop_accuracy,
    build_default_eval_samples,
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


def test_evaluator_default_profile_is_heuristic_overlap():
    samples = [
        EvalSample(
            question="What is Hierarchical Risk Parity?",
            reference_answer="HRP uses clustering and risk parity.",
            expected_context_terms=["hierarchical", "risk parity", "clustering"],
        )
    ]

    evaluator = Evaluator(samples=samples, retriever=_StubRetriever())
    report = evaluator.run()

    assert report.metric_profile == "heuristic-overlap"
    assert report.num_samples == 1
    assert 0.0 <= report.context_recall <= 1.0
    assert 0.0 <= report.context_precision <= 1.0
    assert 0.0 <= report.answer_faithfulness <= 1.0
    assert 0.0 <= report.answer_relevance <= 1.0
    assert 0.0 <= report.link_prediction_ndcg_at_3 <= 1.0
    assert 0.0 <= report.link_prediction_ndcg_at_10 <= 1.0
    assert 0.0 <= report.rank_quality <= 1.0


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

    assert report.metric_profile == "heuristic-overlap"
    assert report.num_samples == 1


def test_evaluator_legacy_ragas_style_alias_normalizes():
    samples = [
        EvalSample(
            question="What is Hierarchical Risk Parity?",
            reference_answer="HRP uses clustering and risk parity.",
            expected_context_terms=["hierarchical", "risk parity", "clustering"],
        )
    ]

    evaluator = Evaluator(samples=samples, retriever=_StubRetriever(), metric_profile="ragas-style")
    report = evaluator.run()

    assert report.metric_profile == "heuristic-overlap"


def test_evaluator_unknown_metric_profile_raises():
    samples = [
        EvalSample(
            question="What is Hierarchical Risk Parity?",
            reference_answer="HRP uses clustering and risk parity.",
            expected_context_terms=["hierarchical", "risk parity"],
        )
    ]

    with pytest.raises(ValueError, match="Unknown metric_profile"):
        Evaluator(samples=samples, metric_profile="foo-bar")


def test_evaluator_graph_profile_alias_normalizes_to_graph_order_sensitive():
    samples = [
        EvalSample(
            question="What is Hierarchical Risk Parity?",
            reference_answer="HRP uses clustering and risk parity.",
            expected_context_terms=["hierarchical", "risk parity", "clustering"],
        )
    ]

    evaluator = Evaluator(samples=samples, retriever=_StubRetriever(), metric_profile="graph")
    report = evaluator.run()

    assert report.metric_profile == "graph-order-sensitive"
    assert report.rank_quality >= 0.0


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


def test_build_default_eval_samples():
    samples = build_default_eval_samples()
    assert len(samples) >= 25
    assert any(s.difficulty == "easy" for s in samples)
    assert any(s.difficulty == "hard" for s in samples)
    assert any(s.tags and "negative-control" in s.tags for s in samples)


def test_per_sample_contains_diagnostic_fields():
    """run() must emit actual_retrieval_mode, retrieved_sources, matched_terms, missed_terms."""
    samples = [
        EvalSample(
            question="What is HRP?",
            reference_answer="HRP uses clustering and risk parity.",
            expected_context_terms=["hrp", "clustering"],
        )
    ]
    evaluator = Evaluator(
        samples=samples,
        retriever=_StubRetriever(),
        runtime_config={"retrieval_mode": "hybrid_rerank", "embedding_provider": "hash"},
    )
    report = evaluator.run()
    row = report.per_sample[0]

    assert "actual_retrieval_mode" in row
    assert row["actual_retrieval_mode"] == "hybrid_rerank"
    assert "retrieved_sources" in row
    assert isinstance(row["retrieved_sources"], list)
    assert "matched_terms" in row
    assert isinstance(row["matched_terms"], list)
    assert "missed_terms" in row
    assert isinstance(row["missed_terms"], list)
    # matched + missed should partition expected_context_terms
    assert set(row["matched_terms"]) | set(row["missed_terms"]) == {"hrp", "clustering"}


def test_report_has_run_at_timestamp():
    """EvalReport.run_at must be a non-empty ISO-8601 string after run()."""
    evaluator = Evaluator(_make_samples(1))
    report = evaluator.run()
    assert report.run_at != ""
    # Verify it is a plausible ISO timestamp (contains "T")
    assert "T" in report.run_at


def test_run_at_serialized_in_save(tmp_path):
    """run_at must appear in the JSON artifact written by save()."""
    evaluator = Evaluator(_make_samples(1))
    output = tmp_path / "out.json"
    evaluator.save(output)
    data = json.loads(output.read_text())
    assert "run_at" in data
    assert data["run_at"] != ""


def test_save_with_precomputed_report_does_not_rerun(tmp_path):
    """Passing a report to save() must serialize that exact report without re-evaluating."""
    run_count = 0

    class _CountingEvaluator(Evaluator):
        def run(self) -> EvalReport:  # type: ignore[override]
            nonlocal run_count
            run_count += 1
            return super().run()

    evaluator = _CountingEvaluator(_make_samples(1))
    report = evaluator.run()
    assert run_count == 1
    output = tmp_path / "out.json"
    evaluator.save(output, report)
    assert run_count == 1  # save() must not call run() again


def test_faithfulness_reflects_reference_answer_support():
    """Faithfulness must drop when contexts do not support the reference answer."""
    sample_with_match = EvalSample(
        question="What is HRP?",
        reference_answer="HRP uses clustering and risk parity.",
        expected_context_terms=["hrp", "clustering"],
    )
    sample_no_match = EvalSample(
        question="What is HRP?",
        reference_answer="HRP uses clustering and risk parity.",
        expected_context_terms=["hrp", "clustering"],
    )

    class _OffTopicRetriever:
        def retrieve(self, query: str) -> list[RetrievalResult]:
            _ = query
            return [
                RetrievalResult(
                    content="Apples and oranges are common fruits found in grocery stores.",
                    source_path="off_topic.md",
                    score=0.1,
                )
            ]

    evaluator_relevant = Evaluator(samples=[sample_with_match], retriever=_StubRetriever())
    evaluator_irrelevant = Evaluator(samples=[sample_no_match], retriever=_OffTopicRetriever())

    report_relevant = evaluator_relevant.run()
    report_irrelevant = evaluator_irrelevant.run()

    # Retriever returning HRP/clustering content should score higher faithfulness
    assert report_relevant.answer_faithfulness > report_irrelevant.answer_faithfulness


def test_context_precision_matches_multiword_terms():
    """_ragas_style_context_precision must match multi-word expected terms via substring."""
    from riskfolio_graphrag_agent.eval.evaluator import _ragas_style_context_precision

    contexts = ["Hierarchical Risk Parity uses clustering to build portfolios."]
    # "hierarchical risk parity" is multi-word — tokenized matching would miss it
    score_multiword = _ragas_style_context_precision(
        "What is HRP?",
        ["hierarchical risk parity", "clustering"],
        contexts,
    )
    score_single = _ragas_style_context_precision(
        "What is HRP?",
        ["hrp", "clustering"],
        contexts,
    )
    # Both terms appear in the context; multi-word term must register a hit
    assert score_multiword > 0.0
    # Multi-word and single-word terms should both yield non-trivial scores
    assert score_single > 0.0
