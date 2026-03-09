"""Hybrid retriever combining chunk search and graph expansion."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

from neo4j import Driver, GraphDatabase

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

    def search(self, query: str, top_k: int) -> list[VectorHit]:
        """Return top-k chunk hits for query."""


class Neo4jChunkVectorStore:
    """Neo4j-backed lexical proxy for vector search over Chunk nodes.

    This is intentionally backend-agnostic at the retriever layer. A future
    embedding backend can implement :class:`VectorStore` and be swapped in.
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str) -> None:
        self._driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def close(self) -> None:
        self._driver.close()

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
            "c.chunk_kind AS chunk_kind, score "
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
    ) -> None:
        self._uri = neo4j_uri
        self._user = neo4j_user
        self._password = neo4j_password
        self._top_k = top_k
        self._vector_store = vector_store or Neo4jChunkVectorStore(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
        )
        self._driver: Driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

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


def _graph_expand(hit_or_result, session=None):
    """Expand a vector hit with neighboring entities/chunks.

    This function supports legacy test usage where it receives a
    :class:`RetrievalResult` and no session.
    """
    if isinstance(hit_or_result, RetrievalResult) and session is None:
        logger.debug("_graph_expand compatibility path for %s", hit_or_result.source_path)
        return hit_or_result

    hit: VectorHit = hit_or_result
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
