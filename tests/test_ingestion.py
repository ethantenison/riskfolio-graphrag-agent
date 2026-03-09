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


def test_document_identity_and_provenance_metadata(tmp_source_dir):
    """Documents should include stable chunk identity and source provenance fields."""
    docs = load_directory(tmp_source_dir)
    assert docs

    for doc in docs:
        relative_path = str(doc.metadata["relative_path"])
        assert doc.chunk_id == f"{relative_path}::chunk:{doc.chunk_index}"
        assert len(doc.content_hash) == 64
        assert doc.section
        assert doc.line_start >= 1
        assert doc.line_end >= doc.line_start

        assert doc.metadata["source_path"] == doc.source_path
        assert doc.metadata["section"] == doc.section
        assert doc.metadata["line_start"] == doc.line_start
        assert doc.metadata["line_end"] == doc.line_end


def test_load_directory_is_deterministic_and_idempotent(tmp_source_dir):
    """Repeated loads of same corpus should produce identical chunk identity/metadata."""
    docs_first = load_directory(tmp_source_dir)
    docs_second = load_directory(tmp_source_dir)

    def _snapshot(docs: list[Document]) -> list[tuple[str, str, str, int, int, str]]:
        return [
            (
                doc.chunk_id,
                doc.content_hash,
                doc.section,
                doc.line_start,
                doc.line_end,
                doc.source_path,
            )
            for doc in docs
        ]

    assert _snapshot(docs_first) == _snapshot(docs_second)


def test_rst_section_titles_are_human_readable(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    rst_file = docs_dir / "risk.rst"
    rst_file.write_text(
        "##############\n"
        "Risk Measures\n"
        "##############\n\n"
        "The risk module includes CVaR and EVaR.\n"
    )

    docs = load_directory(docs_dir)
    assert docs

    section_titles = [doc.section for doc in docs]
    assert "Risk Measures" in section_titles
    assert not any(title and set(title) <= set("#=-~^`:*") for title in section_titles)
