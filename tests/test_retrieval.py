"""Tests for riskfolio_graphrag_agent.retrieval.retriever."""

from __future__ import annotations

from typing import Any

from riskfolio_graphrag_agent.ingestion.loader import Document
from riskfolio_graphrag_agent.retrieval.retriever import (
    ChromaVectorStore,
    HybridRetriever,
    RetrievalResult,
    VectorHit,
    _graph_expand,
    _hash_embedding,
    _vector_search,
)


class _FakeCollection:
    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for index, chunk_id in enumerate(ids):
            self._records[chunk_id] = {
                "id": chunk_id,
                "document": documents[index],
                "embedding": embeddings[index],
                "metadata": metadatas[index],
            }

    def query(self, query_embeddings: list[list[float]], n_results: int, include: list[str]):
        _ = include
        query_embedding = query_embeddings[0]

        scored: list[tuple[float, dict[str, Any]]] = []
        for record in self._records.values():
            distance = _l2_distance(query_embedding, record["embedding"])
            scored.append((distance, record))
        scored.sort(key=lambda item: item[0])
        top = scored[:n_results]

        return {
            "ids": [[item[1]["id"] for item in top]],
            "documents": [[item[1]["document"] for item in top]],
            "metadatas": [[item[1]["metadata"] for item in top]],
            "distances": [[item[0] for item in top]],
        }


class _FakeChromaClient:
    def __init__(self) -> None:
        self._collections: dict[str, _FakeCollection] = {}

    def get_or_create_collection(self, name: str) -> _FakeCollection:
        if name not in self._collections:
            self._collections[name] = _FakeCollection()
        return self._collections[name]


class _FakeVectorStore:
    def upsert(self, documents: list[Document]) -> int:
        return len(documents)

    def search(self, query: str, top_k: int) -> list[VectorHit]:
        _ = query, top_k
        return []


def _l2_distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=False)) ** 0.5


def _make_doc(content: str, chunk_index: int, section: str) -> Document:
    return Document(
        content=content,
        source_path="/tmp/sample.py",
        chunk_index=chunk_index,
        chunk_id=f"sample.py::chunk:{chunk_index}",
        content_hash=f"h{chunk_index}",
        section=section,
        line_start=1,
        line_end=3,
        metadata={
            "relative_path": "sample.py",
            "chunk_kind": "function",
            "section": section,
            "line_start": 1,
            "line_end": 3,
        },
    )


def test_retrieval_result_dataclass():
    result = RetrievalResult(content="some text", source_path="/a/b.py", score=0.9)
    assert result.score == 0.9
    assert result.graph_neighbours == []


def test_vector_search_stub():
    """_vector_search stub should return an empty list without raising."""
    results = _vector_search("test query", top_k=3)
    assert isinstance(results, list)


def test_chroma_vector_store_upsert_and_query():
    client = _FakeChromaClient()
    store = ChromaVectorStore(
        persist_dir=".ignored",
        collection_name="test_chunks",
        embedding_dim=64,
        client=client,
    )

    docs = [
        _make_doc("Hierarchical Risk Parity allocation method", 0, "hrp_allocation"),
        _make_doc("Conditional Value at Risk explanation", 1, "risk_metrics"),
    ]

    inserted = store.upsert(docs)
    assert inserted == 2

    hits = store.search("HRP allocation", top_k=2)
    assert len(hits) == 2
    assert hits[0].chunk_id == "sample.py::chunk:0"
    assert 0.0 <= hits[0].score <= 1.0
    assert hits[0].metadata["section"] == "hrp_allocation"
    assert hits[0].metadata["line_start"] == 1
    assert hits[0].metadata["line_end"] == 3


def test_hybrid_retriever_upsert_documents_delegates():
    retriever = HybridRetriever(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        top_k=3,
        vector_store=_FakeVectorStore(),
    )
    try:
        count = retriever.upsert_documents([_make_doc("x", 0, "s")])
    finally:
        retriever.close()
    assert count == 1


def test_graph_expand_stub():
    """_graph_expand stub should return the result unchanged."""
    r = RetrievalResult(content="x", source_path="/a.py")
    expanded = _graph_expand(r)
    assert expanded is r


def test_hybrid_retriever_retrieve_stub():
    """HybridRetriever.retrieve should return a list (empty in stub mode)."""
    retriever = HybridRetriever(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        top_k=3,
        vector_store=_FakeVectorStore(),
    )
    try:
        results = retriever.retrieve("What is portfolio optimisation?")
        assert isinstance(results, list)
    finally:
        retriever.close()


def test_hash_embedding_is_deterministic():
    first = _hash_embedding("risk parity", dim=64)
    second = _hash_embedding("risk parity", dim=64)
    assert first == second
