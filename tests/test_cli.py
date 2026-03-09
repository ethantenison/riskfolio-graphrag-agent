"""Tests for riskfolio_graphrag_agent.cli helpers."""

from __future__ import annotations

from riskfolio_graphrag_agent.cli import _select_documents_for_build
from riskfolio_graphrag_agent.ingestion.loader import Document


def _make_docs(count: int) -> list[Document]:
    return [
        Document(content=f"chunk-{index}", source_path=f"/tmp/file_{index}.py", chunk_index=index)
        for index in range(count)
    ]


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
