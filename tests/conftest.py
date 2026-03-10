"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_test_env(monkeypatch, tmp_path):
    """Isolate tests from local shell env/.env and provide deterministic defaults."""
    settings_keys = (
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIM",
        "VECTOR_STORE_BACKEND",
        "CHROMA_PERSIST_DIR",
        "LOG_LEVEL",
        "RISKFOLIO_SOURCE_DIR",
    )
    for key in settings_keys:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / ".chroma"))
    monkeypatch.setenv("RISKFOLIO_SOURCE_DIR", str(tmp_path / "riskfolio-lib"))
    monkeypatch.chdir(tmp_path)


@pytest.fixture()
def smoke_source_dir(tmp_path):
    """Create a small deterministic source tree for smoke/e2e tests."""
    source_root = tmp_path / "riskfolio-lib"
    source_root.mkdir(parents=True, exist_ok=True)

    py_file = source_root / "portfolio.py"
    py_file.write_text("def hrp_allocation():\n    return 'ok'\n")

    docs_dir = source_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    md_file = docs_dir / "hrp.md"
    md_file.write_text("# HRP\n\nHierarchical Risk Parity overview.\n")

    return source_root


@pytest.fixture()
def tmp_source_dir(tmp_path):
    """Create a temporary directory with sample source files for ingestion tests."""
    py_file = tmp_path / "sample.py"
    py_file.write_text("def hello():\n    return 'world'\n")
    md_file = tmp_path / "README.md"
    md_file.write_text("# Sample\n\nThis is a test.\n")
    return tmp_path
