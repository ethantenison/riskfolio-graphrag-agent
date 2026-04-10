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

from riskfolio_graphrag_agent.kg_models import slugify

STRUCTURAL_RELATION_LABELS = frozenset({"accepts-parameter", "defines", "relates-to", "returns"})
CODE_LIKE_ENTITY_TYPES = frozenset({"api_symbol", "python_class", "python_function", "python_module", "python_parameter"})


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

    subject_kind = (subject_type or "").casefold()
    object_kind = (object_type or "").casefold()
    if normalized_label == "is" and (subject_kind in CODE_LIKE_ENTITY_TYPES or object_kind in CODE_LIKE_ENTITY_TYPES):
        return False
    return True
