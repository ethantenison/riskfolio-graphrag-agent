"""Shared heuristics for promoted graph semantic quality.

This module centralizes lightweight policies that distinguish reviewable raw
assertions from relations that are useful in the promoted retrieval graph.
It sits between open extraction and retrieval-time graph traversal so schema
induction, graph materialization, and retrieval can apply the same filtering
rules.

Inputs are relation labels plus optional subject/object type guesses. Outputs
are normalized labels and boolean promotion decisions.

This module does not perform extraction, canonicalization, or graph writes.
"""

from __future__ import annotations

import re

from riskfolio_graphrag_agent.kg_models import slugify

STRUCTURAL_RELATION_LABELS = frozenset({"accepts-parameter", "defines", "relates-to", "returns"})
CODE_LIKE_ENTITY_TYPES = frozenset({"api_symbol", "python_class", "python_function", "python_module", "python_parameter"})
CODE_LIKE_TYPE_TOKENS = frozenset(
    {
        "api",
        "attribute",
        "class",
        "code",
        "decorator",
        "import",
        "module",
        "namespace",
        "package",
        "parameter",
        "symbol",
        "variable",
    }
)


def normalize_relation_label(label: str) -> str:
    """Normalize a free-text relation label into a stable comparison key.

    Args:
        label: Raw relation label or ontology-property label.

    Returns:
        A slugified, lowercase comparison form.
    """
    return slugify(label)


def is_semantic_relation_label(label: str) -> bool:
    """Return whether a relation label is eligible for promoted schema use.

    Args:
        label: Raw relation label or ontology-property label.

    Returns:
        True when the label is not a structural code relation.
    """
    return normalize_relation_label(label) not in STRUCTURAL_RELATION_LABELS


def is_code_like_entity_type(entity_type: str | None) -> bool:
    """Return whether an entity type appears to describe source-code structure.

    Args:
        entity_type: Optional entity type guess from extraction/canonicalization.

    Returns:
        True when the type indicates a code symbol or code-level construct.
    """
    normalized = (entity_type or "").strip().casefold().replace("-", "_")
    if not normalized:
        return False
    if normalized in CODE_LIKE_ENTITY_TYPES:
        return True
    if normalized.startswith("python_"):
        return True
    tokens = {token for token in re.split(r"[^a-z0-9]+", normalized) if token}
    return not CODE_LIKE_TYPE_TOKENS.isdisjoint(tokens)


def is_semantic_assertion(
    relation_label: str,
    subject_type: str | None = None,
    object_type: str | None = None,
) -> bool:
    """Return whether a candidate assertion should enter the promoted graph.

    Args:
        relation_label: Raw relation guess for the assertion.
        subject_type: Optional subject entity type guess.
        object_type: Optional object entity type guess.

    Returns:
        True when the assertion is semantically useful for retrieval.
    """
    normalized_label = normalize_relation_label(relation_label)
    if normalized_label in STRUCTURAL_RELATION_LABELS:
        return False

    if normalized_label == "is" and (is_code_like_entity_type(subject_type) or is_code_like_entity_type(object_type)):
        return False
    return True
