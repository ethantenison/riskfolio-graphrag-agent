"""Tests for riskfolio_graphrag_agent.ingestion.loader."""

from __future__ import annotations

import pytest

from riskfolio_graphrag_agent.ingestion.loader import Document, load_directory


def test_load_directory_returns_documents(tmp_source_dir):
    """load_directory should return at least one Document for each supported file."""
    docs = load_directory(tmp_source_dir)
    assert isinstance(docs, list)
    assert len(docs) >= 2  # one for .py, one for .md


def test_load_directory_document_fields(tmp_source_dir):
    """Each Document should have non-empty content and a valid source_path."""
    docs = load_directory(tmp_source_dir)
    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.content
        assert doc.source_path
        assert doc.chunk_index >= 0


def test_load_directory_missing_path():
    """load_directory should raise FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_directory("/tmp/_nonexistent_riskfolio_test_dir")


def test_load_directory_chunk_size(tmp_source_dir):
    """Chunks should not exceed chunk_size characters."""
    chunk_size = 20
    docs = load_directory(tmp_source_dir, chunk_size=chunk_size, overlap=0)
    for doc in docs:
        assert len(doc.content) <= chunk_size


def test_document_metadata(tmp_source_dir):
    """Documents should carry extension and filename in their metadata."""
    docs = load_directory(tmp_source_dir)
    for doc in docs:
        assert "extension" in doc.metadata
        assert "filename" in doc.metadata
