"""Load and save evaluation samples for the eval layer.

This module provides the eval package's configuration-facing sample I/O. It is
responsible for translating committed or user-supplied JSON files into typed
``EvalSample`` records without moving scoring logic out of the evaluator.

Inputs are JSON files containing either a top-level list of sample objects or a
mapping with a ``samples`` key. Outputs are validated ``EvalSample`` instances
and JSON artifacts that preserve optional sample metadata used for segmented
reporting.

This module does not compute metrics, run retrieval, or apply regression-gate
policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from riskfolio_graphrag_agent.eval.evaluator import EvalSample


def load_eval_samples(path: str | Path) -> list[EvalSample]:
    """Load evaluation samples from a JSON file.

    The file may contain either a top-level JSON array or an object with a
    ``samples`` field containing the array.

    Args:
        path: Path to the JSON sample file.

    Returns:
        Parsed evaluation samples.

    Raises:
        ValueError: If the file format or sample payload is invalid.
    """
    target = Path(path)
    payload = json.loads(target.read_text())

    if isinstance(payload, dict):
        raw_samples = payload.get("samples")
    else:
        raw_samples = payload

    if not isinstance(raw_samples, list):
        raise ValueError("Evaluation sample file must contain a list or a mapping with a 'samples' list.")

    return [_sample_from_dict(item, index=index) for index, item in enumerate(raw_samples, start=1)]


def save_eval_samples(samples: list[EvalSample], path: str | Path) -> None:
    """Write evaluation samples to a JSON file.

    Args:
        samples: Sample records to serialize.
        path: Destination path for the JSON file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"samples": [_sample_to_dict(sample) for sample in samples]}, indent=2))


def _sample_from_dict(payload: Any, *, index: int) -> EvalSample:
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation sample #{index} must be a JSON object.")

    question = str(payload.get("question", "")).strip()
    reference_answer = str(payload.get("reference_answer", "")).strip()
    if not question or not reference_answer:
        raise ValueError(f"Evaluation sample #{index} must include non-empty 'question' and 'reference_answer'.")

    expected_context_terms = _string_list(
        payload.get("expected_context_terms", []), field_name="expected_context_terms", index=index
    )
    retrieved_contexts = _string_list(payload.get("retrieved_contexts", []), field_name="retrieved_contexts", index=index)
    retrieved_sources = _string_list(payload.get("retrieved_sources", []), field_name="retrieved_sources", index=index)
    tags = _string_list(payload.get("tags", []), field_name="tags", index=index)

    return EvalSample(
        question=question,
        reference_answer=reference_answer,
        expected_context_terms=expected_context_terms,
        generated_answer=str(payload.get("generated_answer", "")),
        retrieved_contexts=retrieved_contexts,
        retrieved_sources=retrieved_sources,
        domain=str(payload.get("domain", "")).strip(),
        difficulty=str(payload.get("difficulty", "")).strip(),
        retrieval_type=str(payload.get("retrieval_type", "")).strip(),
        tags=tags,
    )


def _sample_to_dict(sample: EvalSample) -> dict[str, Any]:
    return {
        "question": sample.question,
        "reference_answer": sample.reference_answer,
        "expected_context_terms": list(sample.expected_context_terms),
        "generated_answer": sample.generated_answer,
        "retrieved_contexts": list(sample.retrieved_contexts),
        "retrieved_sources": list(sample.retrieved_sources),
        "domain": sample.domain,
        "difficulty": sample.difficulty,
        "retrieval_type": sample.retrieval_type,
        "tags": list(sample.tags),
    }


def _string_list(value: Any, *, field_name: str, index: int) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Evaluation sample #{index} field '{field_name}' must be a list of strings.")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"Evaluation sample #{index} field '{field_name}' must contain only strings.")
        text = item.strip()
        if text:
            normalized.append(text)
    return normalized
