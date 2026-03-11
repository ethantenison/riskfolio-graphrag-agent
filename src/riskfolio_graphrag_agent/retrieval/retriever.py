"""Hybrid retriever combining dense, sparse, and graph retrieval modes."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from neo4j import Driver, GraphDatabase

from riskfolio_graphrag_agent.graph.builder import DOMAIN_ALIASES
from riskfolio_graphrag_agent.ingestion.loader import Document as IngestDocument
from riskfolio_graphrag_agent.retrieval.embeddings import EmbeddingProvider, HashEmbeddingProvider, hash_embedding

logger = logging.getLogger(__name__)

RetrievalMode = Literal["dense", "sparse", "graph", "hybrid_rerank"]


@dataclass
class RetrievalResult:
    """A retrieved evidence chunk plus graph context."""

    content: str
    source_path: str
    score: float = 0.0
    graph_neighbours: list[str] = field(default_factory=list)
    related_entities: list[str] = field(default_factory=list)
    metadata: dict[str, str | int | list[str] | float] = field(default_factory=dict)


@dataclass
class VectorHit:
    """A retrieval hit for a chunk."""

    chunk_id: str
    content: str
    source_path: str
    score: float
    metadata: dict[str, str | int | list[str] | float] = field(default_factory=dict)


class VectorStore(Protocol):
    """Vector store interface for chunk retrieval."""

    def upsert(self, documents: list[IngestDocument]) -> int: ...

    def search(self, query: str, top_k: int) -> list[VectorHit]: ...


class ChromaVectorStore:
    """Chroma-backed vector store for upsert and dense search."""

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "riskfolio_chunks",
        embedding_provider: EmbeddingProvider | None = None,
        client: Any | None = None,
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._embedding_provider = embedding_provider or HashEmbeddingProvider(dimension=256)

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
        metadatas: list[dict[str, str | int | float | bool]] = []
        for doc in documents:
            ids.append(doc.chunk_id)
            texts.append(doc.content)
            metadatas.append(_sanitize_metadata_for_chroma(doc))

        embeddings = self._embedding_provider.embed_texts(texts)
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

        query_embedding = self._embedding_provider.embed_texts([query])[0]
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

            source_path = str(metadata.get("source_path", "")) or str(metadata.get("relative_path", ""))

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
    """Neo4j-backed lexical retrieval over Chunk nodes."""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str) -> None:
        self._driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def close(self) -> None:
        self._driver.close()

    def upsert(self, documents: list[IngestDocument]) -> int:
        _ = documents
        return 0

    def search(self, query: str, top_k: int) -> list[VectorHit]:
        return _sparse_query_hits(self._driver, query, top_k=top_k)


class HybridRetriever:
    """Retrieval orchestrator supporting ablation modes."""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        top_k: int = 5,
        vector_store: VectorStore | None = None,
        vector_store_backend: str = "neo4j",
        chroma_persist_dir: str = ".chroma",
        embedding_provider: EmbeddingProvider | None = None,
        retrieval_mode: RetrievalMode = "hybrid_rerank",
    ) -> None:
        self._uri = neo4j_uri
        self._user = neo4j_user
        self._password = neo4j_password
        self._top_k = top_k
        self._retrieval_mode: RetrievalMode = retrieval_mode
        self._embedding_provider = embedding_provider or HashEmbeddingProvider(dimension=256)
        self._vector_store = vector_store or _build_default_vector_store(
            backend=vector_store_backend,
            chroma_persist_dir=chroma_persist_dir,
            embedding_provider=self._embedding_provider,
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

    def retrieve(self, query: str, mode_override: RetrievalMode | None = None) -> list[RetrievalResult]:
        retrieval_mode = mode_override or self._retrieval_mode
        logger.info("Retrieving for query: %r (top_k=%d mode=%s)", query, self._top_k, retrieval_mode)

        if retrieval_mode == "dense":
            hits = self._vector_store.search(query, top_k=self._top_k)
        elif retrieval_mode == "sparse":
            hits = _sparse_query_hits(self._driver, query, top_k=self._top_k)
        elif retrieval_mode == "graph":
            hits = _graph_seed_hits(self._driver, query, top_k=self._top_k)
            # Domain-concept graph-hop expansion: traverse IS_SUBTYPE_OF, ALTERNATIVE_TO, REQUIRES.
            hop_hits = _graph_hop_expansion(self._driver, query, top_k=self._top_k)
            if hop_hits:
                hits = _merge_hits(hits, hop_hits, top_k=self._top_k * 2)[: self._top_k]
        else:
            dense_hits = self._vector_store.search(query, top_k=self._top_k)
            sparse_hits = _sparse_query_hits(self._driver, query, top_k=self._top_k)
            hits = _merge_hits(dense_hits, sparse_hits, top_k=self._top_k)

        if not hits:
            return []

        results: list[RetrievalResult] = []
        with self._driver.session() as session:
            for hit in hits:
                results.append(_graph_expand(hit, session))

        if retrieval_mode == "hybrid_rerank":
            for result in results:
                graph_boost = 0.03 * len(result.related_entities) + 0.015 * len(result.graph_neighbours)
                result.score = round(float(result.score) + graph_boost, 6)

        results.sort(key=lambda item: item.score, reverse=True)
        return results[: self._top_k]


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
    embedding_provider: EmbeddingProvider,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
) -> VectorStore:
    normalized = backend.strip().lower()
    if normalized == "chroma":
        try:
            return ChromaVectorStore(
                persist_dir=chroma_persist_dir,
                embedding_provider=embedding_provider,
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
    """Compatibility wrapper kept for existing tests."""
    return hash_embedding(text, dim=dim)


def _graph_expand(hit_or_result, session=None):
    """Expand a hit with neighboring entities/chunks."""
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


def _sparse_query_hits(driver: Driver, query: str, top_k: int) -> list[VectorHit]:
    if top_k <= 0:
        return []
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

    with driver.session() as session:
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


def _graph_seed_hits(driver: Driver, query: str, top_k: int) -> list[VectorHit]:
    tokens = _query_tokens(query)
    if not tokens:
        return []

    cypher = (
        "MATCH (e) "
        "WHERE e.name IS NOT NULL "
        "AND any(t IN $tokens WHERE toLower(e.name) CONTAINS t) "
        "WITH e, size([t IN $tokens WHERE toLower(e.name) CONTAINS t]) AS entity_score "
        "OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e) "
        "WHERE c.name IS NOT NULL "
        "WITH c, max(entity_score) AS score "
        "WHERE c IS NOT NULL "
        "RETURN c.name AS chunk_id, c.content AS content, c.source_path AS source_path, "
        "c.relative_path AS relative_path, c.chunk_index AS chunk_index, "
        "c.chunk_kind AS chunk_kind, c.line_start AS line_start, c.line_end AS line_end, score "
        "ORDER BY score DESC LIMIT $top_k"
    )

    with driver.session() as session:
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


# ---------------------------------------------------------------------------
# Build a flat set of known domain-concept names from DOMAIN_ALIASES for fast
# membership checks (lowercased).
# ---------------------------------------------------------------------------
_DOMAIN_CONCEPT_NAMES: frozenset[str] = frozenset(name.lower() for concepts in DOMAIN_ALIASES.values() for name in concepts)


def _find_domain_concepts(query: str) -> list[str]:
    """Return canonical concept names from DOMAIN_ALIASES that appear in *query*."""
    lowered = query.lower()
    matched: list[str] = []
    for concepts in DOMAIN_ALIASES.values():
        for canonical_name in concepts:
            if canonical_name.lower() in lowered:
                matched.append(canonical_name)
    return matched


def _graph_hop_expansion(driver: Driver, query: str, top_k: int) -> list[VectorHit]:
    """Expand graph retrieval by traversing IS_SUBTYPE_OF, ALTERNATIVE_TO, and REQUIRES hops.

    For each domain concept found in *query*, retrieve chunks linked to:
    1. The concept's parent class (one hop up via IS_SUBTYPE_OF).
    2. Alternative concepts (one hop across via ALTERNATIVE_TO).
    3. Required prerequisites (one hop down via REQUIRES).
    """
    concepts = _find_domain_concepts(query)
    if not concepts:
        return []

    cypher = (
        "UNWIND $concepts AS concept_name "
        "MATCH (e) WHERE toLower(e.name) = toLower(concept_name) "
        "OPTIONAL MATCH (e)-[:IS_SUBTYPE_OF]->(parent) "
        "OPTIONAL MATCH (e)-[:ALTERNATIVE_TO]-(alt) "
        "OPTIONAL MATCH (e)-[:REQUIRES]->(req) "
        "WITH collect(DISTINCT parent) + collect(DISTINCT alt) + collect(DISTINCT req) AS related_nodes "
        "UNWIND related_nodes AS rn "
        "WHERE rn IS NOT NULL "
        "OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(rn) WHERE c.name IS NOT NULL "
        "WITH c, max(1) AS score WHERE c IS NOT NULL "
        "RETURN c.name AS chunk_id, c.content AS content, "
        "c.source_path AS source_path, c.relative_path AS relative_path, "
        "c.chunk_index AS chunk_index, c.chunk_kind AS chunk_kind, "
        "c.line_start AS line_start, c.line_end AS line_end, score "
        "ORDER BY score DESC LIMIT $top_k"
    )

    try:
        with driver.session() as session:
            rows = list(session.run(cypher, concepts=concepts, top_k=top_k))
    except Exception as exc:
        logger.debug("Graph-hop expansion query failed: %s", exc)
        return []

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


def _merge_hits(dense_hits: list[VectorHit], sparse_hits: list[VectorHit], top_k: int) -> list[VectorHit]:
    merged: dict[str, VectorHit] = {}
    for hit in dense_hits:
        key = hit.chunk_id
        merged[key] = hit

    for hit in sparse_hits:
        key = hit.chunk_id
        existing = merged.get(key)
        if existing is None:
            merged[key] = VectorHit(
                chunk_id=hit.chunk_id,
                content=hit.content,
                source_path=hit.source_path,
                score=0.0,
                metadata=dict(hit.metadata),
            )
            existing = merged[key]

        existing.score = (0.65 * float(existing.score)) + (0.35 * float(hit.score))
        for field_name in ("content", "source_path"):
            if not getattr(existing, field_name):
                setattr(existing, field_name, getattr(hit, field_name))

    ordered = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    return ordered[:top_k]
