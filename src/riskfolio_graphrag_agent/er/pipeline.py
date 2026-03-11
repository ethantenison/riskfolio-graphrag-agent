"""Entity resolution workflow with deterministic and optional model-assisted stages."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable

from riskfolio_graphrag_agent.graph.builder import DOMAIN_ALIASES


def _build_alias_lookup() -> dict[str, str]:
    """Build a reverse map from normalised alias text → normalised canonical key."""
    lookup: dict[str, str] = {}
    for _, concepts in DOMAIN_ALIASES.items():
        for canonical_name, aliases in concepts.items():
            canon_key = _canonical_key_raw(canonical_name)
            # The canonical name itself maps to its own key.
            lookup[canon_key] = canon_key
            for alias in aliases:
                lookup[_canonical_key_raw(alias)] = canon_key
    return lookup


def _canonical_key_raw(name: str) -> str:
    """Normalise a name to a sortable key without alias substitution."""
    normalised = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return normalised.replace(" ", "_") or "unknown"


# Module-level reverse-alias lookup built once at import time.
_ALIAS_LOOKUP: dict[str, str] = _build_alias_lookup()


def _jaccard_tokens(name: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", name.lower()))


def _jaccard_similarity(a: str, b: str) -> float:
    ta = _jaccard_tokens(a)
    tb = _jaccard_tokens(b)
    union = len(ta | tb)
    if not union:
        return 0.0
    return len(ta & tb) / union


@dataclass
class EntityRecord:
    entity_id: str
    name: str
    source: str = ""
    entity_type: str = "Concept"


@dataclass
class CanonicalEntity:
    canonical_id: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    source_entity_ids: list[str] = field(default_factory=list)


@dataclass
class ERMetrics:
    precision: float
    recall: float
    f1: float


@dataclass
class ERPipelineResult:
    canonical_entities: list[CanonicalEntity]
    predicted_pairs: set[tuple[str, str]]
    metrics: ERMetrics | None = None


def run_er_pipeline(
    entities: list[EntityRecord],
    *,
    gold_pairs: set[tuple[str, str]] | None = None,
    model_assist: Callable[[EntityRecord, EntityRecord], bool] | None = None,
    audit_dir: str | Path | None = None,
) -> ERPipelineResult:
    grouped: dict[str, list[EntityRecord]] = {}
    for entity in entities:
        key = _canonical_key(entity.name)
        grouped.setdefault(key, []).append(entity)

    grouped = _apply_jaccard_merge(grouped)

    if model_assist is not None:
        grouped = _apply_model_assist(grouped, model_assist)

    canonical_entities: list[CanonicalEntity] = []
    predicted_pairs: set[tuple[str, str]] = set()

    for key, records in grouped.items():
        canonical = sorted(records, key=lambda row: (len(row.name), row.name))[0]
        aliases = sorted({record.name for record in records if record.name != canonical.name})
        ids = sorted({record.entity_id for record in records})
        canonical_entities.append(
            CanonicalEntity(
                canonical_id=f"canon::{key}",
                canonical_name=canonical.name,
                aliases=aliases,
                source_entity_ids=ids,
            )
        )

        for left, right in combinations(ids, 2):
            predicted_pairs.add(tuple(sorted((left, right))))

    metrics = evaluate_er(predicted_pairs=predicted_pairs, gold_pairs=gold_pairs) if gold_pairs is not None else None
    result = ERPipelineResult(
        canonical_entities=sorted(canonical_entities, key=lambda row: row.canonical_id),
        predicted_pairs=predicted_pairs,
        metrics=metrics,
    )

    if audit_dir is not None:
        _write_audit(result=result, audit_dir=audit_dir, input_entities=entities, gold_pairs=gold_pairs)

    return result


def evaluate_er(*, predicted_pairs: set[tuple[str, str]], gold_pairs: set[tuple[str, str]] | None) -> ERMetrics:
    if gold_pairs is None:
        return ERMetrics(precision=0.0, recall=0.0, f1=0.0)

    normalized_pred = {tuple(sorted(pair)) for pair in predicted_pairs}
    normalized_gold = {tuple(sorted(pair)) for pair in gold_pairs}

    true_positive = len(normalized_pred & normalized_gold)
    false_positive = len(normalized_pred - normalized_gold)
    false_negative = len(normalized_gold - normalized_pred)

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

    return ERMetrics(precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4))


def _canonical_key(name: str) -> str:
    """Normalise *name* to a canonical bucket key.

    Lookup order:
    1. Check ``_ALIAS_LOOKUP`` built from ``DOMAIN_ALIASES`` — e.g. both
       ``"CVaR"`` and ``"conditional value at risk"`` resolve to the same key.
    2. Apply legacy hard-coded substitutions for backward compatibility.
    """
    raw_key = _canonical_key_raw(name)
    # DOMAIN_ALIASES lookup: if the normalised form (or any known alias) is
    # already indexed, return the canonical key directly.
    if raw_key in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[raw_key]
    # Legacy substitutions kept for backward compatibility.
    normalised = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    normalised = normalised.replace("risk parity", "riskparity")
    normalised = normalised.replace("value at risk", "var")
    return normalised.replace(" ", "_") or "unknown"


def _apply_jaccard_merge(
    grouped: dict[str, list[EntityRecord]],
    threshold: float = 0.70,
) -> dict[str, list[EntityRecord]]:
    """Merge groups whose representative names share Jaccard similarity ≥ *threshold*."""
    keys = sorted(grouped.keys())
    merged = dict(grouped)

    for i, left_key in enumerate(keys):
        if left_key not in merged:
            continue
        rep_left = sorted(merged[left_key], key=lambda r: (len(r.name), r.name))[0]
        for right_key in keys[i + 1 :]:
            if right_key not in merged:
                continue
            rep_right = sorted(merged[right_key], key=lambda r: (len(r.name), r.name))[0]
            if _jaccard_similarity(rep_left.name, rep_right.name) >= threshold:
                merged[left_key] = merged[left_key] + merged[right_key]
                del merged[right_key]

    return merged


def _apply_model_assist(
    grouped: dict[str, list[EntityRecord]],
    model_assist: Callable[[EntityRecord, EntityRecord], bool],
) -> dict[str, list[EntityRecord]]:
    keys = sorted(grouped.keys())
    merged = dict(grouped)

    for i, left_key in enumerate(keys):
        if left_key not in merged:
            continue
        left_records = merged[left_key]
        for right_key in keys[i + 1 :]:
            if right_key not in merged:
                continue
            right_records = merged[right_key]
            if not left_records or not right_records:
                continue

            if model_assist(left_records[0], right_records[0]):
                merged[left_key] = left_records + right_records
                del merged[right_key]

    return merged


def _write_audit(
    *,
    result: ERPipelineResult,
    audit_dir: str | Path,
    input_entities: list[EntityRecord],
    gold_pairs: set[tuple[str, str]] | None,
) -> None:
    target = Path(audit_dir)
    target.mkdir(parents=True, exist_ok=True)

    payload = {
        "input_entities": [asdict(entity) for entity in input_entities],
        "canonical_entities": [asdict(entity) for entity in result.canonical_entities],
        "predicted_pairs": sorted([list(pair) for pair in result.predicted_pairs]),
        "gold_pairs": sorted([list(pair) for pair in (gold_pairs or set())]),
        "metrics": asdict(result.metrics) if result.metrics is not None else None,
    }

    (target / "er_audit.json").write_text(json.dumps(payload, indent=2))
