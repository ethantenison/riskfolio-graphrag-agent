"""Tests for integration profile artifacts and commands."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_integration_compose_has_required_services():
    compose = (ROOT / "docker-compose.integration.yml").read_text().lower()
    assert "neo4j" in compose
    assert "chromadb" in compose


def test_smoke_script_contains_end_to_end_steps():
    script = (ROOT / "scripts" / "run_integration_smoke.sh").read_text().lower()
    assert "riskfolio-agent ingest" in script
    assert "riskfolio-agent build-graph" in script
    assert "agentworkflow" in script
    assert "riskfolio-agent eval" in script
    assert "riskfolio-agent eval-gate" in script
