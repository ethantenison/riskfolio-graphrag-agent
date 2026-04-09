"""Tests for integration profile artifacts and commands."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

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


def test_quickstart_eval_samples_path_points_to_existing_file():
    quickstart = (ROOT / "docs" / "quickstart.md").read_text()
    expected_fragment = "riskfolio-agent eval --samples benchmarks/eval_samples_v1.json"
    assert expected_fragment in quickstart
    assert (ROOT / "benchmarks" / "eval_samples_v1.json").exists()


@pytest.mark.integration
def test_quickstart_option_b_end_to_end_runs_when_enabled():
    """RUN_OPTION_B_INTEGRATION=1 poetry run pytest -q -m integration"""
    if os.environ.get("RUN_OPTION_B_INTEGRATION") != "1":
        pytest.skip("Set RUN_OPTION_B_INTEGRATION=1 to run the full Option B integration flow.")

    base_env = os.environ.copy()
    command_plan: list[tuple[list[str], dict[str, str] | None]] = [
        (["poetry", "install"], None),
        (["poetry", "run", "pytest", "-q"], {"RUN_OPTION_B_INTEGRATION": "0"}),
        (
            [
                "poetry",
                "run",
                "riskfolio-agent",
                "eval",
                "--samples",
                "benchmarks/eval_samples_v1.json",
            ],
            None,
        ),
        (["poetry", "run", "python", "scripts/benchmark_retrieval_ablation.py"], None),
        (["poetry", "run", "riskfolio-agent", "er-eval"], None),
        (["poetry", "run", "python", "scripts/report_observability.py"], None),
    ]

    for command, env_override in command_plan:
        command_env = base_env.copy()
        if env_override:
            command_env.update(env_override)
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=command_env,
            text=True,
            capture_output=True,
            check=False,
            timeout=1800,
        )
        assert result.returncode == 0, (
            f"Command failed: {' '.join(command)}\nstdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
        )
