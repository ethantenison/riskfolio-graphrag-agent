"""Tests for observability report generation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from riskfolio_graphrag_agent.observability.reporting import build_observability_report


def test_build_observability_report(tmp_path):
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "answer_faithfulness": 0.6,
                "grounding": 0.7,
                "avg_latency_ms": 200.0,
                "estimated_cost_usd": 0.001,
            }
        )
    )

    out = tmp_path / "sli.json"
    report = build_observability_report(
        eval_report_path=eval_path,
        output_path=out,
        data_last_updated_utc=(datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
        drift_score=0.05,
        freshness_budget_hours=24,
        drift_threshold=0.2,
    )

    assert out.exists()
    assert report["status"] == "healthy"
