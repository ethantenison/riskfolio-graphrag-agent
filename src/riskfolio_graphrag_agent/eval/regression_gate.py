"""Evaluation regression gate utilities and CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class RegressionGateError(RuntimeError):
    """Raised when evaluation metrics violate configured thresholds."""


def run_regression_gate(
    report_path: str | Path,
    min_faithfulness: float = 0.35,
    min_relevance: float = 0.8,
    min_context_recall: float = 0.45,
) -> None:
    report = json.loads(Path(report_path).read_text())

    faithfulness = float(report.get("answer_faithfulness", 0.0))
    relevance = float(report.get("answer_relevance", 0.0))
    recall = float(report.get("context_recall", 0.0))

    failures: list[str] = []
    if faithfulness < min_faithfulness:
        failures.append(
            f"answer_faithfulness={faithfulness:.4f} < min_faithfulness={min_faithfulness:.4f}"
        )
    if relevance < min_relevance:
        failures.append(f"answer_relevance={relevance:.4f} < min_relevance={min_relevance:.4f}")
    if recall < min_context_recall:
        failures.append(f"context_recall={recall:.4f} < min_context_recall={min_context_recall:.4f}")

    if failures:
        raise RegressionGateError("; ".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluation regression gate")
    parser.add_argument("--report", default="eval_results.json", help="Path to eval report JSON")
    parser.add_argument("--min-faithfulness", type=float, default=0.35)
    parser.add_argument("--min-relevance", type=float, default=0.8)
    parser.add_argument("--min-context-recall", type=float, default=0.45)
    args = parser.parse_args()

    try:
        run_regression_gate(
            report_path=args.report,
            min_faithfulness=args.min_faithfulness,
            min_relevance=args.min_relevance,
            min_context_recall=args.min_context_recall,
        )
    except RegressionGateError as exc:
        print(f"Eval regression gate failed: {exc}")
        return 1

    print("Eval regression gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
