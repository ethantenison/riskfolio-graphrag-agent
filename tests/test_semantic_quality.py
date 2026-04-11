"""Tests for semantic assertion quality filtering policies."""

from __future__ import annotations

from riskfolio_graphrag_agent.semantic_quality import is_code_like_entity_type, is_semantic_assertion


def test_is_code_like_entity_type_handles_python_prefix_and_tokens() -> None:
    assert is_code_like_entity_type("python_function")
    assert is_code_like_entity_type("python-method")
    assert is_code_like_entity_type("api_symbol")
    assert is_code_like_entity_type("code_symbol")
    assert not is_code_like_entity_type("custom_function_like")
    assert not is_code_like_entity_type("risk_measure_like")
    assert not is_code_like_entity_type("portfolio_methodology")


def test_is_semantic_assertion_filters_code_level_is_relations() -> None:
    assert not is_semantic_assertion("is", "python_function", "concept")
    assert not is_semantic_assertion("is", "portfolio_method", "python_class")
    assert not is_semantic_assertion("is", "python-method", "concept")


def test_is_semantic_assertion_keeps_domain_level_is_relations() -> None:
    assert is_semantic_assertion("is", "portfolio_method", "risk_measure_family")
    assert is_semantic_assertion("supports", "python_function", "risk_measure_family")
