"""Tests for riskfolio_graphrag_agent.retrieval.retriever."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from riskfolio_graphrag_agent.ingestion.loader import Document
from riskfolio_graphrag_agent.retrieval.embeddings import HashEmbeddingProvider
from riskfolio_graphrag_agent.retrieval.retriever import (
    ChromaVectorStore,
    HybridRetriever,
    RetrievalResult,
    VectorHit,
    _dense_query_variants,
    _graph_expand,
    _merge_hits,
    _promoted_graph_bridge_hits,
    _promoted_graph_hop_expansion,
    _promoted_graph_seed_hits,
    _query_tokens_for_lexical_graph,
    _sparse_query_hits,
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


class _Row(dict):
    pass


class _SessionCtx:
    def __init__(self, rows: list[_Row]) -> None:
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _ = exc_type, exc_val, exc_tb
        return False

    def run(self, cypher: str, **params):
        _ = cypher, params
        return self._rows


class _DriverCtx:
    def __init__(self, rows: list[_Row]) -> None:
        self._rows = rows

    def session(self):
        return _SessionCtx(self._rows)


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


def test_chroma_vector_store_upsert_and_query():
    client = _FakeChromaClient()
    store = ChromaVectorStore(
        persist_dir=".ignored",
        collection_name="test_chunks",
        embedding_provider=HashEmbeddingProvider(dimension=64),
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
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = iter([])

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    with patch(
        "riskfolio_graphrag_agent.retrieval.retriever.GraphDatabase.driver",
        return_value=mock_driver,
    ):
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


def test_sparse_query_hits_supports_node_id_chunks():
    rows = [
        _Row(
            {
                "chunk_id": "chunk:abc",
                "content": "CVaR can be used for tail-risk control.",
                "source_path": "docs/risk.md",
                "relative_path": "docs/risk.md",
                "chunk_index": 4,
                "chunk_kind": "section",
                "line_start": 11,
                "line_end": 24,
                "score": 2,
            }
        )
    ]
    driver = _DriverCtx(rows)

    hits = _sparse_query_hits(driver, "cvar tail risk", top_k=3)

    assert len(hits) == 1
    assert hits[0].chunk_id == "chunk:abc"
    assert hits[0].metadata["relative_path"] == "docs/risk.md"


def test_query_tokens_for_lexical_graph_expands_synonyms():
    tokens = _query_tokens_for_lexical_graph("Compare CVaR report plot")

    assert "cvar" in tokens
    assert "tail" in tokens
    assert "risk" in tokens
    assert "reporting" in tokens
    assert "chart" in tokens


def test_dense_query_variants_adds_augmented_variant_for_synonyms():
    variants = _dense_query_variants("Compare CVaR report plot")

    assert variants[0] == "Compare CVaR report plot"
    assert len(variants) == 2
    assert "tail" in variants[1].lower()
    assert "reporting" in variants[1].lower()


def test_promoted_graph_seed_hits_reads_assertion_backed_chunks():
    rows = [
        _Row(
            {
                "chunk_id": "chunk:seed",
                "content": "Hierarchical Risk Parity uses CVaR.",
                "source_path": "docs/risk.md",
                "relative_path": "docs/risk.md",
                "chunk_index": 0,
                "chunk_kind": "section",
                "line_start": 1,
                "line_end": 1,
                "score": 0.72,
            }
        )
    ]
    driver = _DriverCtx(rows)

    hits = _promoted_graph_seed_hits(driver, "hierarchical risk parity", top_k=5)

    assert len(hits) == 1
    assert hits[0].chunk_id == "chunk:seed"
    assert hits[0].score == 0.72


def test_promoted_graph_hop_expansion_returns_hits():
    rows = [
        _Row(
            {
                "chunk_id": "chunk:hop",
                "content": "CVaR belongs to tail-risk measures.",
                "source_path": "docs/risk.md",
                "relative_path": "docs/risk.md",
                "chunk_index": 1,
                "chunk_kind": "section",
                "line_start": 2,
                "line_end": 3,
                "score": 0.61,
            }
        )
    ]
    driver = _DriverCtx(rows)

    hits = _promoted_graph_hop_expansion(driver, "tail risk class", top_k=5)

    assert len(hits) == 1
    assert hits[0].chunk_id == "chunk:hop"


def test_promoted_graph_bridge_hits_returns_hits():
    rows = [
        _Row(
            {
                "chunk_id": "chunk:bridge",
                "content": "Tail-risk peers connect CVaR and CDaR assertions.",
                "source_path": "docs/risk.md",
                "relative_path": "docs/risk.md",
                "chunk_index": 7,
                "chunk_kind": "section",
                "line_start": 3,
                "line_end": 6,
                "score": 0.59,
            }
        )
    ]
    driver = _DriverCtx(rows)

    hits = _promoted_graph_bridge_hits(driver, "tail risk", top_k=5)

    assert len(hits) == 1
    assert hits[0].chunk_id == "chunk:bridge"


def test_merge_hits_prefers_cross_channel_overlap():
    dense_hits = [
        VectorHit(chunk_id="a", content="a", source_path="a.py", score=0.95),
        VectorHit(chunk_id="b", content="b", source_path="b.py", score=0.9),
    ]
    sparse_hits = [
        VectorHit(chunk_id="c", content="c", source_path="c.py", score=4.0),
        VectorHit(chunk_id="a", content="a", source_path="a.py", score=3.0),
    ]

    merged = _merge_hits(dense_hits, sparse_hits, top_k=3)

    assert merged
    assert merged[0].chunk_id == "a"


def test_merge_hits_returns_empty_when_top_k_non_positive():
    dense_hits = [VectorHit(chunk_id="a", content="a", source_path="a.py", score=0.5)]
    sparse_hits = [VectorHit(chunk_id="a", content="a", source_path="a.py", score=1.0)]

    assert _merge_hits(dense_hits, sparse_hits, top_k=0) == []


def test_hybrid_retriever_uses_wider_candidate_pool_in_hybrid_mode():
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = iter([])

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    class _DenseOnlyStore:
        def __init__(self) -> None:
            self.search_calls: list[int] = []

        def upsert(self, documents: list[Document]) -> int:
            return len(documents)

        def search(self, query: str, top_k: int) -> list[VectorHit]:
            _ = query
            self.search_calls.append(top_k)
            return [VectorHit(chunk_id="dense-1", content="dense", source_path="dense.py", score=0.9)]

    vector_store = _DenseOnlyStore()

    with (
        patch("riskfolio_graphrag_agent.retrieval.retriever.GraphDatabase.driver", return_value=mock_driver),
        patch(
            "riskfolio_graphrag_agent.retrieval.retriever._sparse_query_hits",
            return_value=[VectorHit(chunk_id="sparse-1", content="sparse", source_path="sparse.py", score=2.0)],
        ),
        patch(
            "riskfolio_graphrag_agent.retrieval.retriever._graph_expand",
            side_effect=lambda hit, session: RetrievalResult(
                content=hit.content,
                source_path=hit.source_path,
                score=hit.score,
                graph_neighbours=[],
                related_entities=[],
                metadata={},
            ),
        ),
    ):
        retriever = HybridRetriever(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            top_k=3,
            vector_store=vector_store,
            retrieval_mode="hybrid_rerank",
        )
        try:
            results = retriever.retrieve("How does HRP relate to CVaR?")
        finally:
            retriever.close()

    assert results
    assert vector_store.search_calls == [6]


def test_dense_retriever_uses_query_variants_and_wider_candidate_pool():
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = iter([])

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    class _DenseOnlyStore:
        def __init__(self) -> None:
            self.search_calls: list[tuple[str, int]] = []

        def upsert(self, documents: list[Document]) -> int:
            return len(documents)

        def search(self, query: str, top_k: int) -> list[VectorHit]:
            self.search_calls.append((query, top_k))
            return [VectorHit(chunk_id=f"dense-{len(self.search_calls)}", content=query, source_path="dense.py", score=0.9)]

    vector_store = _DenseOnlyStore()

    with (
        patch("riskfolio_graphrag_agent.retrieval.retriever.GraphDatabase.driver", return_value=mock_driver),
        patch(
            "riskfolio_graphrag_agent.retrieval.retriever._graph_expand",
            side_effect=lambda hit, session: RetrievalResult(
                content=hit.content,
                source_path=hit.source_path,
                score=hit.score,
                graph_neighbours=[],
                related_entities=[],
                metadata={},
            ),
        ),
    ):
        retriever = HybridRetriever(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            top_k=3,
            vector_store=vector_store,
            retrieval_mode="dense",
        )
        try:
            results = retriever.retrieve("How does CVaR report plotting work?")
        finally:
            retriever.close()

    assert results
    assert len(vector_store.search_calls) == 2
    assert all(call_top_k == 6 for _, call_top_k in vector_store.search_calls)


def test_graph_retriever_uses_wider_candidate_pool_and_sparse_backfill():
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = iter([])

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    with (
        patch("riskfolio_graphrag_agent.retrieval.retriever.GraphDatabase.driver", return_value=mock_driver),
        patch(
            "riskfolio_graphrag_agent.retrieval.retriever._promoted_graph_seed_hits",
            return_value=[VectorHit(chunk_id="seed-1", content="seed", source_path="seed.py", score=0.9)],
        ) as seed_mock,
        patch(
            "riskfolio_graphrag_agent.retrieval.retriever._promoted_graph_hop_expansion",
            return_value=[VectorHit(chunk_id="hop-1", content="hop", source_path="hop.py", score=0.7)],
        ) as hop_mock,
        patch(
            "riskfolio_graphrag_agent.retrieval.retriever._promoted_graph_bridge_hits",
            return_value=[VectorHit(chunk_id="bridge-1", content="bridge", source_path="bridge.py", score=0.8)],
        ) as bridge_mock,
        patch(
            "riskfolio_graphrag_agent.retrieval.retriever._sparse_query_hits",
            return_value=[VectorHit(chunk_id="sparse-1", content="sparse", source_path="sparse.py", score=2.0)],
        ) as sparse_mock,
        patch(
            "riskfolio_graphrag_agent.retrieval.retriever._graph_expand",
            side_effect=lambda hit, session: RetrievalResult(
                content=hit.content,
                source_path=hit.source_path,
                score=hit.score,
                graph_neighbours=[],
                related_entities=[],
                metadata={"chunk_id": hit.chunk_id},
            ),
        ),
    ):
        retriever = HybridRetriever(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            top_k=3,
            vector_store=_FakeVectorStore(),
            retrieval_mode="graph",
        )
        try:
            results = retriever.retrieve("How does HRP connect to CVaR?")
        finally:
            retriever.close()

    assert results
    assert seed_mock.call_args.kwargs["top_k"] == 9
    assert hop_mock.call_args.kwargs["top_k"] == 9
    assert bridge_mock.call_args.kwargs["top_k"] == 9
    assert sparse_mock.call_args.kwargs["top_k"] == 4
