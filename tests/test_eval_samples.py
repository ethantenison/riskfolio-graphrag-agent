"""Tests for eval sample loading and saving."""

from __future__ import annotations

import json

import pytest

from riskfolio_graphrag_agent.eval.evaluator import EvalSample
from riskfolio_graphrag_agent.eval.samples import load_eval_samples, save_eval_samples


def test_load_eval_samples_reads_mapping_with_samples(tmp_path):
    sample_file = tmp_path / "samples.json"
    sample_file.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "question": "How does HRP work?",
                        "reference_answer": "HRP uses clustering.",
                        "expected_context_terms": ["hrp", "clustering"],
                        "domain": "portfolio-construction",
                        "difficulty": "easy",
                        "retrieval_type": "hybrid",
                        "tags": ["hrp", "allocation"],
                    }
                ]
            }
        )
    )

    samples = load_eval_samples(sample_file)

    assert len(samples) == 1
    assert samples[0].question == "How does HRP work?"
    assert samples[0].tags == ["hrp", "allocation"]


def test_save_eval_samples_round_trips_metadata(tmp_path):
    sample_file = tmp_path / "samples.json"
    samples = [
        EvalSample(
            question="What is CVaR?",
            reference_answer="CVaR is a downside risk measure.",
            expected_context_terms=["cvar", "risk"],
            domain="risk-measures",
            difficulty="easy",
            retrieval_type="dense",
            tags=["cvar"],
        )
    ]

    save_eval_samples(samples, sample_file)
    reloaded = load_eval_samples(sample_file)

    assert reloaded == samples


def test_load_eval_samples_rejects_invalid_payload(tmp_path):
    sample_file = tmp_path / "samples.json"
    sample_file.write_text(json.dumps({"samples": [{"question": "Incomplete"}]}))

    with pytest.raises(ValueError):
        load_eval_samples(sample_file)
