"""Service-level indicators, SLOs, and drift/freshness reporting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class SLOTargets:
    min_answer_faithfulness: float = 0.4
    min_grounding: float = 0.4
    max_avg_latency_ms: float = 3000.0
    max_estimated_cost_usd: float = 0.02


def build_observability_report(
    *,
    eval_report_path: str | Path,
    output_path: str | Path,
    data_last_updated_utc: str,
    drift_score: float,
    freshness_budget_hours: int,
    drift_threshold: float,
    slo_targets: SLOTargets | None = None,
) -> dict[str, object]:
    targets = slo_targets or SLOTargets()
    report = json.loads(Path(eval_report_path).read_text())

    faithfulness = float(report.get("answer_faithfulness", 0.0))
    grounding = float(report.get("grounding", 0.0))
    latency_ms = float(report.get("avg_latency_ms", 0.0))
    estimated_cost_usd = float(report.get("estimated_cost_usd", 0.0))

    updated_dt = datetime.fromisoformat(data_last_updated_utc)
    now = datetime.now(timezone.utc)
    age_hours = (now - updated_dt).total_seconds() / 3600.0

    sli = {
        "answer_faithfulness": faithfulness,
        "grounding": grounding,
        "avg_latency_ms": latency_ms,
        "estimated_cost_usd": estimated_cost_usd,
        "drift_score": drift_score,
        "data_freshness_hours": round(age_hours, 2),
    }

    slo = {
        "faithfulness_met": faithfulness >= targets.min_answer_faithfulness,
        "grounding_met": grounding >= targets.min_grounding,
        "latency_met": latency_ms <= targets.max_avg_latency_ms,
        "cost_met": estimated_cost_usd <= targets.max_estimated_cost_usd,
        "drift_met": drift_score <= drift_threshold,
        "freshness_met": age_hours <= freshness_budget_hours,
    }

    status = "healthy" if all(bool(value) for value in slo.values()) else "degraded"
    payload = {
        "status": status,
        "generated_at_utc": now.isoformat(),
        "sli": sli,
        "slo": slo,
        "targets": {
            "min_answer_faithfulness": targets.min_answer_faithfulness,
            "min_grounding": targets.min_grounding,
            "max_avg_latency_ms": targets.max_avg_latency_ms,
            "max_estimated_cost_usd": targets.max_estimated_cost_usd,
            "drift_threshold": drift_threshold,
            "freshness_budget_hours": freshness_budget_hours,
        },
    }

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2))
    return payload
