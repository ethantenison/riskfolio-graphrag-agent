"""Retrieval-quality and answer-faithfulness evaluation suite.

Metrics (planned)
-----------------
- **Context Recall** – fraction of ground-truth evidence chunks retrieved.
- **Context Precision** – fraction of retrieved chunks that are relevant.
- **Answer Faithfulness** – whether the answer is entailed by the context.
- **Answer Relevance** – semantic similarity of answer to question.

This module currently provides **stub** implementations.  Wire in a RAGAS
or custom evaluation harness once the retrieval and agent layers are ready.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """A single question/answer/context evaluation sample.

    Attributes:
        question: The input question.
        reference_answer: Gold-standard answer for comparison.
        generated_answer: Answer produced by the agent.
        retrieved_contexts: Context chunks used to produce the answer.
    """

    question: str
    reference_answer: str
    generated_answer: str = ""
    retrieved_contexts: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """Aggregated evaluation results.

    Attributes:
        num_samples: Total number of evaluated samples.
        context_recall: Mean context recall score (0–1).
        context_precision: Mean context precision score (0–1).
        answer_faithfulness: Mean faithfulness score (0–1).
        answer_relevance: Mean answer-relevance score (0–1).
    """

    num_samples: int = 0
    context_recall: float = 0.0
    context_precision: float = 0.0
    answer_faithfulness: float = 0.0
    answer_relevance: float = 0.0


class Evaluator:
    """Runs the evaluation suite over a list of :class:`EvalSample` objects.

    Args:
        samples: Evaluation samples to score.
    """

    def __init__(self, samples: list[EvalSample]) -> None:
        self._samples = samples

    def run(self) -> EvalReport:
        """Execute all evaluation metrics and return an :class:`EvalReport`.

        Returns:
            Aggregated metric scores across all samples.
        """
        logger.info("Evaluator.run called with %d samples (stub).", len(self._samples))
        # TODO: implement per-sample metric computation and aggregation
        return EvalReport(num_samples=len(self._samples))

    def save(self, output_path: str | Path) -> None:
        """Run evaluation and write results to *output_path* as JSON.

        Args:
            output_path: File path for the JSON results file.
        """
        report = self.run()
        path = Path(output_path)
        path.write_text(json.dumps(asdict(report), indent=2))
        logger.info("Evaluation results written to %s", path)
