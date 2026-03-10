"""Hybrid retriever combining chunk search and graph expansion."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from neo4j import Driver, GraphDatabase

from riskfolio_graphrag_agent.ingestion.loader import Document as IngestDocument

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A retrieved evidence chunk plus graph context."""

    content: str
    source_path: str
    score: float = 0.0
    graph_neighbours: list[str] = field(default_factory=list)
    related_entities: list[str] = field(default_factory=list)
    metadata: dict[str, str | int | list[str]] = field(default_factory=dict)


@dataclass
class VectorHit:
    """A vector store hit for a chunk."""

    chunk_id: str
    content: str
    source_path: str
    score: float
    metadata: dict[str, str | int | list[str]] = field(default_factory=dict)


class VectorStore(Protocol):
    """Vector store interface for chunk retrieval."""

    def upsert(self, documents: list[IngestDocument]) -> int:
        """Upsert chunk documents into the vector store and return count."""
        ...

    def search(self, query: str, top_k: int) -> list[VectorHit]:
        """Return top-k chunk hits for query."""
        ...


class ChromaVectorStore:
    """Chroma-backed vector store for ingestion-time upserts and query-time search."""

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "riskfolio_chunks",
        embedding_dim: int = 256,
        client: Any | None = None,
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._embedding_dim = embedding_dim

        if client is not None:
            self._client = client
        else:
            try:
                import chromadb  # type: ignore[import-not-found]
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Chroma backend selected but chromadb is not installed. Install dependencies and retry."
                ) from exc
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_dir)

        self._collection = self._client.get_or_create_collection(name=collection_name)

    def close(self) -> None:
        return None

    def upsert(self, documents: list[IngestDocument]) -> int:
        if not documents:
            return 0

        ids: list[str] = []
        texts: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, str | int | float | bool]] = []
        for doc in documents:
            ids.append(doc.chunk_id)
            texts.append(doc.content)
            embeddings.append(_hash_embedding(doc.content, dim=self._embedding_dim))
            metadatas.append(_sanitize_metadata_for_chroma(doc))

        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(ids)

    def search(self, query: str, top_k: int) -> list[VectorHit]:
        if top_k <= 0:
            return []

        query_embedding = _hash_embedding(query, dim=self._embedding_dim)
        response = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        hits: list[VectorHit] = []
        for index, chunk_id in enumerate(ids):
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 0.0
            score = 1.0 / (1.0 + max(0.0, distance))

            source_path = str(metadata.get("source_path", ""))
            if not source_path:
                source_path = str(metadata.get("relative_path", ""))

            hits.append(
                VectorHit(
                    chunk_id=str(chunk_id),
                    content=str(documents[index]) if index < len(documents) else "",
                    source_path=source_path,
                    score=score,
                    metadata={
                        "relative_path": str(metadata.get("relative_path", "")),
                        "chunk_index": int(metadata.get("chunk_index", 0)),
                        "chunk_kind": str(metadata.get("chunk_kind", "")),
                        "section": str(metadata.get("section", "")),
                        "line_start": int(metadata.get("line_start", 1)),
                        "line_end": int(metadata.get("line_end", 1)),
                        "content_hash": str(metadata.get("content_hash", "")),
                    },
                )
            )

        return hits


class Neo4jChunkVectorStore:
    """Neo4j-backed lexical proxy for vector search over Chunk nodes.

    This is intentionally backend-agnostic at the retriever layer. A future
    embedding backend can implement :class:`VectorStore` and be swapped in.
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str) -> None:
        self._driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def close(self) -> None:
        self._driver.close()

    def upsert(self, documents: list[IngestDocument]) -> int:
        _ = documents
        return 0

    def search(self, query: str, top_k: int) -> list[VectorHit]:
        tokens = _query_tokens(query)
        if not tokens:
            return []

        cypher = (
            "MATCH (c:Chunk) "
            "WITH c, [t IN $tokens WHERE toLower(c.content) CONTAINS t] AS matched "
            "WITH c, matched, size(matched) AS score "
            "WHERE score > 0 "
            "RETURN c.name AS chunk_id, c.content AS content, c.source_path AS source_path, "
            "c.relative_path AS relative_path, c.chunk_index AS chunk_index, "
            "c.chunk_kind AS chunk_kind, c.line_start AS line_start, c.line_end AS line_end, score "
            "ORDER BY score DESC LIMIT $top_k"
        )

        with self._driver.session() as session:
            rows = list(session.run(cypher, tokens=tokens, top_k=top_k))

        hits: list[VectorHit] = []
        for row in rows:
            hits.append(
                VectorHit(
                    chunk_id=str(row["chunk_id"]),
                    content=str(row["content"]),
                    source_path=str(row["source_path"]),
                    score=float(row["score"]),
                    metadata={
                        "relative_path": str(row["relative_path"]),
                        "chunk_index": int(row["chunk_index"]),
                        "chunk_kind": str(row["chunk_kind"]),
                        "section": "",
                        "line_start": int(row["line_start"] or 1),
                        "line_end": int(row["line_end"] or 1),
                        "content_hash": "",
                    },
                )
            )
        return hits


class HybridRetriever:
    """Combines vector search with graph-guided expansion around chunks."""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        top_k: int = 5,
        vector_store: VectorStore | None = None,
        vector_store_backend: str = "neo4j",
        chroma_persist_dir: str = ".chroma",
        embedding_dim: int = 256,
    ) -> None:
        self._uri = neo4j_uri
        self._user = neo4j_user
        self._password = neo4j_password
        self._top_k = top_k
        self._vector_store = vector_store or _build_default_vector_store(
            backend=vector_store_backend,
            chroma_persist_dir=chroma_persist_dir,
            embedding_dim=embedding_dim,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
        )
        self._driver: Driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def upsert_documents(self, documents: list[IngestDocument]) -> int:
        return self._vector_store.upsert(documents)

    def close(self) -> None:
        if hasattr(self._vector_store, "close"):
            self._vector_store.close()  # type: ignore[call-arg]
        self._driver.close()

    def __enter__(self) -> HybridRetriever:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def retrieve(self, query: str) -> list[RetrievalResult]:
        logger.info("Retrieving for query: %r (top_k=%d)", query, self._top_k)

        hits = self._vector_store.search(query, top_k=self._top_k)
        if not hits:
            return []
        results: list[RetrievalResult] = []
        with self._driver.session() as session:
            for hit in hits:
                results.append(_graph_expand(hit, session))

        results.sort(key=lambda item: item.score, reverse=True)
        return results


def _query_tokens(query: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query.lower())
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped[:12]


def _vector_search(query: str, top_k: int) -> list[RetrievalResult]:
    """Compatibility helper retained for existing tests."""
    logger.debug("_vector_search helper called for %r (top_k=%d)", query, top_k)
    return []


def _build_default_vector_store(
    backend: str,
    chroma_persist_dir: str,
    embedding_dim: int,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
) -> VectorStore:
    normalized = backend.strip().lower()
    if normalized == "chroma":
        try:
            return ChromaVectorStore(
                persist_dir=chroma_persist_dir,
                embedding_dim=embedding_dim,
            )
        except Exception as exc:
            logger.warning(
                "Falling back to Neo4jChunkVectorStore because Chroma is unavailable: %s",
                exc,
            )

    return Neo4jChunkVectorStore(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
    )


def _sanitize_metadata_for_chroma(doc: IngestDocument) -> dict[str, str | int | float | bool]:
    return {
        "source_path": doc.source_path,
        "relative_path": str(doc.metadata.get("relative_path", "")),
        "chunk_index": doc.chunk_index,
        "chunk_kind": str(doc.metadata.get("chunk_kind", "")),
        "section": doc.section,
        "line_start": doc.line_start,
        "line_end": doc.line_end,
        "content_hash": doc.content_hash,
    }


def _hash_embedding(text: str, dim: int = 256) -> list[float]:
    vector = [0.0] * max(8, dim)
    tokens = _query_tokens(text)
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % len(vector)
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _graph_expand(hit_or_result, session=None):
    """Expand a vector hit with neighboring entities/chunks.

    This function supports legacy test usage where it receives a
    :class:`RetrievalResult` and no session.
    """
    if isinstance(hit_or_result, RetrievalResult) and session is None:
        logger.debug("_graph_expand compatibility path for %s", hit_or_result.source_path)
        return hit_or_result

    hit = cast(VectorHit, hit_or_result)
    if session is None:
        raise ValueError("session is required when expanding a VectorHit")

    neighbour_cypher = (
        "MATCH (c:Chunk {name: $chunk_id}) "
        "OPTIONAL MATCH (c)-[:MENTIONS]->(e) "
        "OPTIONAL MATCH (src)-[:HAS_CHUNK]->(c) "
        "OPTIONAL MATCH (src)-[:HAS_CHUNK]->(near:Chunk) WHERE near.name <> c.name "
        "RETURN collect(DISTINCT e.name)[0..20] AS entities, "
        "collect(DISTINCT near.name)[0..10] AS neighbour_chunks"
    )

    row = session.run(neighbour_cypher, chunk_id=hit.chunk_id).single()
    entities = [str(item) for item in (row["entities"] if row else []) if item]
    neighbour_chunks = [str(item) for item in (row["neighbour_chunks"] if row else []) if item]

    combined_neighbours = sorted(set(entities + neighbour_chunks))
    score = hit.score + (0.05 * len(entities)) + (0.02 * len(neighbour_chunks))

    return RetrievalResult(
        content=hit.content,
        source_path=hit.source_path,
        score=score,
        related_entities=entities,
        graph_neighbours=combined_neighbours,
        metadata={
            **hit.metadata,
            "chunk_id": hit.chunk_id,
            "graph_neighbor_chunks": neighbour_chunks,
        },
    )
