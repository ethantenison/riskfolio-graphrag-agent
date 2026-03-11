#!/usr/bin/env python3
"""Generate observability report from evaluation artifacts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from riskfolio_graphrag_agent.config.settings import Settings
from riskfolio_graphrag_agent.observability.reporting import build_observability_report


def main() -> int:
    settings = Settings()
    now = datetime.now(timezone.utc)
    data_last_updated = (now - timedelta(hours=6)).isoformat()

    report = build_observability_report(
        eval_report_path="eval_results.json",
        output_path=settings.observability_sli_path,
        data_last_updated_utc=data_last_updated,
        drift_score=0.08,
        freshness_budget_hours=settings.observability_freshness_hours,
        drift_threshold=settings.observability_drift_threshold,
    )

    print(f"status={report['status']}")
    print(f"output={settings.observability_sli_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
