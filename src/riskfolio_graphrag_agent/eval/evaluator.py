"""Retrieval-quality and answer-faithfulness evaluation harness."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from riskfolio_graphrag_agent.retrieval.retriever import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """A single question/answer/context evaluation sample."""

    question: str
    reference_answer: str
    expected_context_terms: list[str] = field(default_factory=list)
    generated_answer: str = ""
    retrieved_contexts: list[str] = field(default_factory=list)
    retrieved_sources: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """Aggregated evaluation results."""

    num_samples: int = 0
    context_recall: float = 0.0
    context_precision: float = 0.0
    answer_faithfulness: float = 0.0
    answer_relevance: float = 0.0
    per_sample: list[dict[str, str | float | int]] = field(default_factory=list)


class Evaluator:
    """Run deterministic retrieval/faithfulness metrics over evaluation samples."""

    def __init__(
        self,
        samples: list[EvalSample],
        retriever: HybridRetriever | None = None,
    ) -> None:
        self._samples = samples
        self._retriever = retriever

    def run(self) -> EvalReport:
        logger.info("Running evaluation over %d samples.", len(self._samples))

        recall_scores: list[float] = []
        precision_scores: list[float] = []
        faithfulness_scores: list[float] = []
        relevance_scores: list[float] = []
        per_sample: list[dict[str, str | float | int]] = []

        for sample in self._samples:
            results = self._retrieve(sample.question)
            sample.retrieved_contexts = [result.content for result in results]
            sample.retrieved_sources = [result.source_path for result in results]
            sample.generated_answer = _synthesize_answer(sample.question, results)

            expected_terms = sample.expected_context_terms or _fallback_expected_terms(sample)
            recall = _context_recall(expected_terms, sample.retrieved_contexts)
            precision = _context_precision(expected_terms, sample.retrieved_contexts)
            faithfulness = _answer_faithfulness(sample.generated_answer, sample.retrieved_contexts)
            relevance = _answer_relevance(sample.question, sample.generated_answer)

            recall_scores.append(recall)
            precision_scores.append(precision)
            faithfulness_scores.append(faithfulness)
            relevance_scores.append(relevance)

            per_sample.append(
                {
                    "question": sample.question,
                    "context_recall": round(recall, 4),
                    "context_precision": round(precision, 4),
                    "answer_faithfulness": round(faithfulness, 4),
                    "answer_relevance": round(relevance, 4),
                    "retrieved_contexts": len(sample.retrieved_contexts),
                }
            )

        return EvalReport(
            num_samples=len(self._samples),
            context_recall=_mean(recall_scores),
            context_precision=_mean(precision_scores),
            answer_faithfulness=_mean(faithfulness_scores),
            answer_relevance=_mean(relevance_scores),
            per_sample=per_sample,
        )

    def save(self, output_path: str | Path) -> None:
        report = self.run()
        path = Path(output_path)
        path.write_text(json.dumps(asdict(report), indent=2))
        logger.info("Evaluation results written to %s", path)

    def _retrieve(self, question: str) -> list[RetrievalResult]:
        if self._retriever is None:
            return []
        try:
            return self._retriever.retrieve(question)
        except Exception as exc:
            logger.warning("Retriever failed for %r: %s", question, exc)
            return []


DEFAULT_EVAL_SAMPLES: list[EvalSample] = [
    EvalSample(
        question="What is Hierarchical Risk Parity (HRP)?",
        reference_answer="HRP is a hierarchical portfolio allocation method based on clustering and risk budgeting.",
        expected_context_terms=["hrp", "hierarchical risk parity", "clustering", "risk parity"],
    ),
    EvalSample(
        question="How does Riskfolio compute CVaR-based optimization?",
        reference_answer="Riskfolio supports CVaR as a risk measure and can optimize portfolios under CVaR objectives/constraints.",
        expected_context_terms=["cvar", "value at risk", "risk measure", "optimization"],
    ),
    EvalSample(
        question="Which estimators are used for covariance in Riskfolio?",
        reference_answer="Riskfolio documents multiple estimators, including historical and shrinkage-style estimators.",
        expected_context_terms=["historical", "ledoit", "shrinkage", "covariance"],
    ),
    EvalSample(
        question="What constraints can be applied in Riskfolio optimization?",
        reference_answer="Riskfolio supports multiple portfolio constraints such as budget, leverage, and risk-related constraints.",
        expected_context_terms=["constraint", "budget", "leverage", "risk contribution"],
    ),
    EvalSample(
        question="How do examples demonstrate portfolio reports and plots?",
        reference_answer="Examples and docs show plotting/reporting workflows for frontier, allocations, and risk contributions.",
        expected_context_terms=["examples", "report", "plot", "efficient frontier"],
    ),
]


def build_default_eval_samples() -> list[EvalSample]:
    return [
        EvalSample(
            question=sample.question,
            reference_answer=sample.reference_answer,
            expected_context_terms=list(sample.expected_context_terms),
        )
        for sample in DEFAULT_EVAL_SAMPLES
    ]


def _fallback_expected_terms(sample: EvalSample) -> list[str]:
    return _tokens(sample.question + " " + sample.reference_answer)[:8]


def _context_recall(expected_terms: list[str], contexts: list[str]) -> float:
    if not expected_terms:
        return 0.0
    corpus = "\n".join(contexts).lower()
    matched = sum(1 for term in expected_terms if term.lower() in corpus)
    return matched / len(expected_terms)


def _context_precision(expected_terms: list[str], contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    expected_lower = [term.lower() for term in expected_terms]
    relevant = 0
    for context in contexts:
        lowered = context.lower()
        if any(term in lowered for term in expected_lower):
            relevant += 1
    return relevant / len(contexts)


def _answer_faithfulness(answer: str, contexts: list[str]) -> float:
    answer_tokens = set(_tokens(answer))
    if not answer_tokens:
        return 0.0
    context_tokens = set(_tokens(" ".join(contexts)))
    if not context_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def _answer_relevance(question: str, answer: str) -> float:
    q_tokens = set(_tokens(question))
    a_tokens = set(_tokens(answer))
    if not q_tokens or not a_tokens:
        return 0.0
    return len(q_tokens & a_tokens) / len(q_tokens)


def _synthesize_answer(question: str, results: list[RetrievalResult]) -> str:
    if not results:
        return "No supporting contexts were retrieved."

    top = results[0]
    entities = top.related_entities[:5]
    if entities:
        return (
            f"For '{question}', top evidence comes from {top.source_path}. "
            f"Key related entities: {', '.join(entities)}."
        )
    return f"For '{question}', top evidence comes from {top.source_path}."


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
