"""Apply CI-style threshold checks to evaluation reports and track trends.

This module is the eval-layer policy gate described in the architecture map. It
consumes JSON reports produced by the evaluator, applies configurable quality
thresholds, appends a compact trend history artifact, and exposes a small CLI
for local use or CI integration.

Inputs are evaluation report paths and threshold values. Outputs are raised
errors for failed gates, terminal exit codes for the CLI, and a rolling JSON
trend file with additive drift diagnostics.

Key implementation decisions:
- threshold enforcement is deterministic and file-based so it can run in CI
    without service dependencies;
- trend tracking is appended before failure is raised so regressions are still
    recorded historically;
- trend entries include additive metric deltas and a simple drift flag so the
    history is useful for diagnostics, not just pass/fail bookkeeping;
- the CLI stays thin and delegates all policy logic to ``run_regression_gate``.

This module does not compute metrics itself, execute retrieval, or own broader
observability reporting.

Usage:
        python -m riskfolio_graphrag_agent.eval.regression_gate --report eval_results.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


class RegressionGateError(RuntimeError):
    """Raised when evaluation metrics violate configured thresholds."""


def run_regression_gate(
    report_path: str | Path,
    min_faithfulness: float = 0.35,
    min_relevance: float = 0.8,
    min_context_recall: float = 0.45,
    min_grounding: float = 0.35,
    min_multi_hop_accuracy: float = 0.25,
    max_latency_ms: float = 5000.0,
    max_estimated_cost_usd: float = 0.02,
    trend_path: str | Path = "artifacts/eval/eval_trend.json",
) -> None:
    """Validate an evaluation report against configured thresholds.

    Args:
        report_path: Path to the JSON report produced by the evaluator.
        min_faithfulness: Minimum acceptable answer faithfulness.
        min_relevance: Minimum acceptable answer relevance.
        min_context_recall: Minimum acceptable context recall.
        min_grounding: Minimum acceptable grounding score.
        min_multi_hop_accuracy: Minimum acceptable multi-hop support score.
        max_latency_ms: Maximum acceptable mean latency in milliseconds.
        max_estimated_cost_usd: Maximum acceptable estimated cost.
        trend_path: Path to the rolling JSON trend artifact.

    Raises:
        RegressionGateError: If any configured threshold is violated.
    """
    report = json.loads(Path(report_path).read_text())

    faithfulness = float(report.get("answer_faithfulness", 0.0))
    relevance = float(report.get("answer_relevance", 0.0))
    recall = float(report.get("context_recall", 0.0))
    grounding = float(report.get("grounding", 0.0))
    multi_hop_accuracy = float(report.get("multi_hop_accuracy", 0.0))
    latency_ms = float(report.get("avg_latency_ms", 0.0))
    estimated_cost_usd = float(report.get("estimated_cost_usd", 0.0))

    failures: list[str] = []
    if faithfulness < min_faithfulness:
        failures.append(f"answer_faithfulness={faithfulness:.4f} < min_faithfulness={min_faithfulness:.4f}")
    if relevance < min_relevance:
        failures.append(f"answer_relevance={relevance:.4f} < min_relevance={min_relevance:.4f}")
    if recall < min_context_recall:
        failures.append(f"context_recall={recall:.4f} < min_context_recall={min_context_recall:.4f}")
    if grounding < min_grounding:
        failures.append(f"grounding={grounding:.4f} < min_grounding={min_grounding:.4f}")
    if multi_hop_accuracy < min_multi_hop_accuracy:
        failures.append(f"multi_hop_accuracy={multi_hop_accuracy:.4f} < min_multi_hop_accuracy={min_multi_hop_accuracy:.4f}")
    if latency_ms > max_latency_ms:
        failures.append(f"avg_latency_ms={latency_ms:.2f} > max_latency_ms={max_latency_ms:.2f}")
    if estimated_cost_usd > max_estimated_cost_usd:
        failures.append(f"estimated_cost_usd={estimated_cost_usd:.6f} > max_estimated_cost_usd={max_estimated_cost_usd:.6f}")

    _append_trend(
        trend_path=trend_path,
        report=report,
        passed=not failures,
        failed_checks=failures,
    )

    if failures:
        raise RegressionGateError("; ".join(failures))


def main() -> int:
    """Run the regression gate CLI.

    Returns:
        ``0`` when the report passes all thresholds, otherwise ``1``.
    """
    parser = argparse.ArgumentParser(description="Evaluation regression gate")
    parser.add_argument("--report", default="eval_results.json", help="Path to eval report JSON")
    parser.add_argument("--min-faithfulness", type=float, default=0.35)
    parser.add_argument("--min-relevance", type=float, default=0.8)
    parser.add_argument("--min-context-recall", type=float, default=0.45)
    parser.add_argument("--min-grounding", type=float, default=0.35)
    parser.add_argument("--min-multi-hop-accuracy", type=float, default=0.25)
    parser.add_argument("--max-latency-ms", type=float, default=5000.0)
    parser.add_argument("--max-estimated-cost-usd", type=float, default=0.02)
    parser.add_argument("--trend-path", default="artifacts/eval/eval_trend.json")
    args = parser.parse_args()

    try:
        run_regression_gate(
            report_path=args.report,
            min_faithfulness=args.min_faithfulness,
            min_relevance=args.min_relevance,
            min_context_recall=args.min_context_recall,
            min_grounding=args.min_grounding,
            min_multi_hop_accuracy=args.min_multi_hop_accuracy,
            max_latency_ms=args.max_latency_ms,
            max_estimated_cost_usd=args.max_estimated_cost_usd,
            trend_path=args.trend_path,
        )
    except RegressionGateError as exc:
        print(f"Eval regression gate failed: {exc}")
        return 1

    print("Eval regression gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _append_trend(
    *,
    trend_path: str | Path,
    report: dict[str, object],
    passed: bool,
    failed_checks: list[str] | None = None,
) -> None:
    target = Path(trend_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, object]] = []
    if target.exists():
        try:
            history = json.loads(target.read_text())
            if not isinstance(history, list):
                history = []
        except json.JSONDecodeError:
            history = []

    previous_entry = history[-1] if history else None
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "context_recall": float(report.get("context_recall", 0.0)),
        "answer_faithfulness": float(report.get("answer_faithfulness", 0.0)),
        "answer_relevance": float(report.get("answer_relevance", 0.0)),
        "grounding": float(report.get("grounding", 0.0)),
        "multi_hop_accuracy": float(report.get("multi_hop_accuracy", 0.0)),
        "avg_latency_ms": float(report.get("avg_latency_ms", 0.0)),
        "estimated_cost_usd": float(report.get("estimated_cost_usd", 0.0)),
        "failed_checks": list(failed_checks or []),
    }
    entry["metric_deltas"] = _metric_deltas(previous_entry=previous_entry, current_entry=entry)
    entry["drift_flagged"] = _drift_flagged(entry["metric_deltas"])

    history.append(entry)

    target.write_text(json.dumps(history[-30:], indent=2))


def _metric_deltas(
    *,
    previous_entry: dict[str, object] | None,
    current_entry: dict[str, object],
) -> dict[str, float]:
    """Return additive metric deltas between the current and previous trend entries."""
    tracked_metrics = (
        "context_recall",
        "answer_faithfulness",
        "answer_relevance",
        "grounding",
        "multi_hop_accuracy",
        "avg_latency_ms",
        "estimated_cost_usd",
    )
    if previous_entry is None:
        return {metric: 0.0 for metric in tracked_metrics}

    deltas: dict[str, float] = {}
    for metric in tracked_metrics:
        current_value = float(current_entry.get(metric, 0.0))
        previous_value = float(previous_entry.get(metric, 0.0))
        deltas[metric] = round(current_value - previous_value, 6)
    return deltas


def _drift_flagged(metric_deltas: dict[str, float]) -> bool:
    """Return whether recent metric deltas indicate meaningful regression drift."""
    score_metrics = (
        "context_recall",
        "answer_faithfulness",
        "answer_relevance",
        "grounding",
        "multi_hop_accuracy",
    )
    if any(metric_deltas.get(metric, 0.0) <= -0.15 for metric in score_metrics):
        return True
    if metric_deltas.get("avg_latency_ms", 0.0) >= 500.0:
        return True
    if metric_deltas.get("estimated_cost_usd", 0.0) >= 0.005:
        return True
    return False
