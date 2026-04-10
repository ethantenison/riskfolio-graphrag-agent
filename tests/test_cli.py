"""Tests for riskfolio_graphrag_agent.cli helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from riskfolio_graphrag_agent import cli
from riskfolio_graphrag_agent.cli import _resolve_eval_samples, _select_documents_for_build
from riskfolio_graphrag_agent.ingestion.loader import Document

runner = CliRunner()


def _make_docs(count: int) -> list[Document]:
    return [Document(content=f"chunk-{index}", source_path=f"/tmp/file_{index}.py", chunk_index=index) for index in range(count)]


def test_select_documents_for_build_all_when_unbounded():
    docs = _make_docs(4)
    selected = _select_documents_for_build(docs)
    assert [doc.chunk_index for doc in selected] == [0, 1, 2, 3]


def test_select_documents_for_build_window():
    docs = _make_docs(6)
    selected = _select_documents_for_build(docs, chunk_offset=2, max_chunks=2)
    assert [doc.chunk_index for doc in selected] == [2, 3]


def test_select_documents_for_build_empty_when_offset_too_large():
    docs = _make_docs(2)
    selected = _select_documents_for_build(docs, chunk_offset=10, max_chunks=1)
    assert selected == []


def test_resolve_eval_samples_uses_custom_file(tmp_path):
    sample_file = tmp_path / "samples.json"
    sample_file.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "question": "What is CVaR?",
                        "reference_answer": "CVaR is a tail-risk measure.",
                        "expected_context_terms": ["cvar", "tail risk"],
                        "domain": "risk-measures",
                        "difficulty": "easy",
                        "retrieval_type": "dense",
                        "tags": ["cvar"],
                    }
                ]
            }
        )
    )

    samples = _resolve_eval_samples(str(sample_file))

    assert len(samples) == 1
    assert samples[0].question == "What is CVaR?"
    assert samples[0].domain == "risk-measures"


def test_eval_gate_cli_passes_all_thresholds(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def _fake_run_regression_gate(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "run_regression_gate", _fake_run_regression_gate)

    report_file = tmp_path / "eval.json"
    report_file.write_text("{}")
    trend_file = tmp_path / "trend.json"

    result = runner.invoke(
        cli.app,
        [
            "eval-gate",
            "--report",
            str(report_file),
            "--min-faithfulness",
            "0.4",
            "--min-relevance",
            "0.85",
            "--min-context-recall",
            "0.5",
            "--min-grounding",
            "0.45",
            "--min-multi-hop-accuracy",
            "0.3",
            "--max-latency-ms",
            "2500",
            "--max-estimated-cost-usd",
            "0.01",
            "--trend-path",
            str(trend_file),
        ],
    )

    assert result.exit_code == 0
    assert captured["report_path"] == str(report_file)
    assert captured["min_grounding"] == 0.45
    assert captured["min_multi_hop_accuracy"] == 0.3
    assert captured["max_latency_ms"] == 2500.0
    assert captured["max_estimated_cost_usd"] == 0.01
    assert captured["trend_path"] == str(trend_file)


def test_eval_cli_accepts_benchmark_samples_file(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    samples_path = repo_root / "benchmarks" / "eval_samples_v1.json"

    class _StubRetriever:
        def retrieve(self, query: str):
            _ = query
            return []

        def close(self):
            return None

    monkeypatch.setattr(
        cli,
        "_resolve_embedding",
        lambda settings: SimpleNamespace(
            provider=None,
            selected_provider="stub",
            fallback_reason=None,
        ),
    )
    monkeypatch.setattr(cli, "HybridRetriever", lambda **kwargs: _StubRetriever())
    monkeypatch.setattr(cli, "run_er_pipeline", lambda *args, **kwargs: SimpleNamespace(metrics=None))

    output_path = tmp_path / "eval_results.json"
    result = runner.invoke(
        cli.app,
        [
            "eval",
            "--samples",
            str(samples_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert payload["num_samples"] > 0


def test_kg_run_cli_writes_summary(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class _FakePipeline:
        def run(self, **kwargs):
            captured.update(kwargs)
            output_dir = tmp_path / "kg"
            output_dir.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(
                graph_quality=SimpleNamespace(
                    num_chunks=2,
                    num_mentions=4,
                    num_candidate_assertions=1,
                    num_canonical_entities=2,
                    num_ontology_classes=1,
                    num_ontology_properties=1,
                ),
                artifact_paths={
                    "ontology_ttl": str(output_dir / "ontology.ttl"),
                    "instances_ttl": str(output_dir / "instances.ttl"),
                },
            )

    monkeypatch.setattr(cli, "_make_kg_pipeline", lambda settings=None: _FakePipeline())
    monkeypatch.setattr(cli, "_resolve_focus_directories", lambda source_dir, settings: [tmp_path])
    monkeypatch.setattr(
        cli,
        "_load_from_directories",
        lambda directories: [
            Document(
                content="Hierarchical Risk Parity uses CVaR.",
                source_path=str(tmp_path / "doc.md"),
                chunk_index=0,
                chunk_id="doc.md::chunk:0",
                content_hash="abc123",
                line_start=1,
                line_end=1,
                metadata={"source_type": "docs", "relative_path": "doc.md"},
            )
        ],
    )

    result = runner.invoke(
        cli.app,
        [
            "kg-run",
            "--artifact-dir",
            str(tmp_path / "kg"),
            "--source-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["artifact_dir"] == str(tmp_path / "kg")
    assert captured["persist_neo4j"] is False


def test_make_kg_pipeline_uses_llm_open_extractor_when_enabled(monkeypatch):
    captured: dict[str, object] = {}

    class _FakePipeline:
        def __init__(self, extractor=None):
            captured["extractor"] = extractor

    class _FakeLLMOpenExtractor:
        def __init__(self, llm_extract, model_name, fallback_extractor=None):
            captured["llm_extract"] = llm_extract
            captured["model_name"] = model_name
            captured["fallback_extractor"] = fallback_extractor

    import riskfolio_graphrag_agent.extraction.pipeline as extraction_pipeline
    import riskfolio_graphrag_agent.kg_pipeline as kg_pipeline_module

    monkeypatch.setattr(extraction_pipeline, "LLMOpenExtractor", _FakeLLMOpenExtractor)
    monkeypatch.setattr(kg_pipeline_module, "KnowledgeGraphPipeline", _FakePipeline)
    monkeypatch.setattr(cli, "_make_openai_open_extractor", lambda settings: object())

    settings = SimpleNamespace(
        openai_enable_graph_extraction=True,
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
    )

    cli._make_kg_pipeline(settings)

    assert captured["extractor"] is not None
    assert captured["model_name"] == "gpt-4o-mini"
