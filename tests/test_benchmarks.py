"""Tests for benchmark scripts used in Quickstart workflows."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def test_retrieval_ablation_benchmark_script_writes_expected_artifacts(tmp_path, monkeypatch):
    """Run the retrieval ablation benchmark script and validate output payloads."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "benchmark_retrieval_ablation.py"

    spec = importlib.util.spec_from_file_location("benchmark_retrieval_ablation", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.chdir(tmp_path)
    exit_code = module.main()
    assert exit_code == 0

    json_output = tmp_path / "benchmarks" / "retrieval_ablation_results.json"
    md_output = tmp_path / "benchmarks" / "retrieval_ablation_results.md"
    assert json_output.exists()
    assert md_output.exists()

    payload = json.loads(json_output.read_text())
    assert payload["fixed_eval_set"] == "riskfolio_graphrag_agent.eval.evaluator.DEFAULT_EVAL_SAMPLES"
    assert payload["winner"] in {"dense", "sparse", "graph", "hybrid_rerank"}
    assert len(payload["results"]) == 4
    assert {row["mode"] for row in payload["results"]} == {"dense", "sparse", "graph", "hybrid_rerank"}
