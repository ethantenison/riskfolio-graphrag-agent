"""Retrieval orchestration for the GraphRAG package.

This module implements the retrieval layer for the package. It is responsible
for turning a user query into ranked evidence chunks that can be consumed by
the agent workflow. In the overall architecture, it sits between:

- ingestion, which produces chunked `Document` records
- graph construction, which stores `Chunk` and related entity nodes in Neo4j
- the agent workflow, which requests grounded context before answer generation

Package fit:
        - `riskfolio_graphrag_agent.ingestion.loader` creates the chunked documents
            that can be upserted into a vector backend.
        - `riskfolio_graphrag_agent.kg_pipeline` and
            `riskfolio_graphrag_agent.graph_materialization` provide the promoted
            graph shape consumed by graph retrieval.
        - `riskfolio_graphrag_agent.retrieval.router` may choose which retrieval
            mode to use per query.
        - `riskfolio_graphrag_agent.agent.workflow` consumes `RetrievalResult`
            objects as evidence for reasoning and citation building.

This module supports four retrieval modes:

- `dense`: embedding-based similarity over a vector store
- `sparse`: lexical token matching over Neo4j `Chunk` nodes
- `graph`: entity-seeded retrieval plus one-hop domain-concept expansion
- `hybrid_rerank`: dense and sparse retrieval merged from a broader candidate
    pool, then lightly boosted using graph neighbourhood and query coverage
    evidence

Minimal working example:
    >>> from riskfolio_graphrag_agent.retrieval.retriever import HybridRetriever
    >>> retriever = HybridRetriever(
    ...     neo4j_uri="bolt://localhost:7687",
    ...     neo4j_user="neo4j",
    ...     neo4j_password="password",
    ...     top_k=3,
    ...     vector_store_backend="neo4j",
    ... )
    >>> try:
    ...     results = retriever.retrieve("What is Hierarchical Risk Parity?")
    ...     for item in results:
    ...         print(item.source_path, round(item.score, 3))
    ... finally:
    ...     retriever.close()

Non-obvious design decisions:
    - The vector backend is abstracted behind `VectorStore` so the retrieval
      flow can work with either Chroma or a Neo4j-backed fallback.
        - Hybrid merging uses reciprocal-rank fusion with score calibration to keep
            the system deterministic, lightweight, and less sensitive to score-scale
            mismatch across backends.
        - Graph expansion is intentionally shallow. It augments retrieval with local
            entity and chunk context, but does not attempt deep graph reasoning.
        - Retrieval now prefers the promoted assertion-aware graph shape and falls
            back to legacy graph patterns when promoted nodes are unavailable.

What this module does not do:
    - It does not chunk source files or create `Document` objects.
    - It does not build the Neo4j knowledge graph schema or extract entities.
    - It does not generate final natural-language answers.
    - It does not choose retrieval mode by itself unless the caller passes a
      mode override or configures a router elsewhere.
    - It does not run a cross-encoder, learned reranker, or approximate nearest
      neighbour index tuning pipeline.
    - It does not translate arbitrary natural language into Cypher queries.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from neo4j import Driver, GraphDatabase

from riskfolio_graphrag_agent.ingestion.loader import Document as IngestDocument
from riskfolio_graphrag_agent.retrieval.embeddings import EmbeddingProvider, HashEmbeddingProvider, hash_embedding

logger = logging.getLogger(__name__)

_QUERY_ONLY_CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "MV": ("mv",),
    "KT": ("kt",),
    "WR": ("wr",),
    "RG": ("rg", "vrg"),
    "ADD": ("add",),
}

# Lightweight lexical expansions for sparse and graph retrieval seeding.
_LEXICAL_TOKEN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "cvar": ("conditional", "tail", "risk"),
    "cdar": ("drawdown", "risk"),
    "ledoit": ("shrinkage",),
    "lasso": ("graphical",),
    "report": ("reports", "reporting", "summary"),
    "plot": ("chart", "figure", "visualization"),
}


def _load_legacy_domain_aliases() -> dict[str, dict[str, tuple[str, ...]]]:
    """Load legacy domain aliases lazily for compatibility fallback.

    Returns:
        Legacy domain alias registry, or an empty dictionary when unavailable.
    """
    try:
        from riskfolio_graphrag_agent.graph.builder import DOMAIN_ALIASES

        return DOMAIN_ALIASES
    except Exception:  # pragma: no cover - defensive import fallback
        return {}


_LEGACY_DOMAIN_ALIASES = _load_legacy_domain_aliases()

RetrievalMode = Literal["dense", "sparse", "graph", "hybrid_rerank"]


@dataclass
class RetrievalResult:
    """Retrieved evidence chunk enriched with graph context.

    Instances of this class are the main output of `HybridRetriever.retrieve`.
    They contain the text content returned to the agent layer, a retrieval
    score, and lightweight graph-derived context for explainability.

    Attributes:
        content: The retrieved chunk text.
        source_path: Original file path associated with the chunk.
        score: Ranking score after retrieval and optional reranking.
        graph_neighbours: Nearby chunk or entity names collected during graph
            expansion.
        related_entities: Entity names directly mentioned by the chunk.
        metadata: Chunk metadata such as relative path, line numbers, and chunk
            identifiers.
    """

    content: str
    source_path: str
    score: float = 0.0
    graph_neighbours: list[str] = field(default_factory=list)
    related_entities: list[str] = field(default_factory=list)
    metadata: dict[str, str | int | list[str] | float] = field(default_factory=dict)


@dataclass
class VectorHit:
    """Internal retrieval hit before graph expansion.

    This is the lower-level representation returned by dense, sparse, and graph
    search backends. It is later converted into `RetrievalResult`.

    Attributes:
        chunk_id: Stable identifier for the chunk node or vector record.
        content: Retrieved chunk text.
        source_path: Source file path for the chunk.
        score: Backend-specific relevance score.
        metadata: Additional chunk metadata used during expansion and display.
    """

    chunk_id: str
    content: str
    source_path: str
    score: float
    metadata: dict[str, str | int | list[str] | float] = field(default_factory=dict)


class VectorStore(Protocol):
    """Protocol for pluggable dense retrieval backends.

    Implementations are expected to support document upsert and query-time
    search over chunk embeddings.
    """

    def upsert(self, documents: list[IngestDocument]) -> int: ...

    def search(self, query: str, top_k: int) -> list[VectorHit]: ...


class ChromaVectorStore:
    """Chroma-backed dense retrieval implementation.

    This backend stores chunk embeddings in a local persistent Chroma
    collection. It is the preferred dense retrieval path when `chromadb` is
    available.

    Args:
        persist_dir: Directory used by Chroma for local persistence.
        collection_name: Collection name for chunk records.
        embedding_provider: Embedding provider used to encode documents and
            queries. Defaults to `HashEmbeddingProvider`.
        client: Optional prebuilt Chroma client, primarily useful for tests.

    Raises:
        RuntimeError: If Chroma is requested but `chromadb` is not installed.
    """

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
            embeddings=cast(Any, embeddings),
            metadatas=cast(Any, metadatas),
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
                        "chunk_index": _as_int(metadata.get("chunk_index", 0), default=0),
                        "chunk_kind": str(metadata.get("chunk_kind", "")),
                        "section": str(metadata.get("section", "")),
                        "line_start": _as_int(metadata.get("line_start", 1), default=1),
                        "line_end": _as_int(metadata.get("line_end", 1), default=1),
                        "content_hash": str(metadata.get("content_hash", "")),
                    },
                )
            )
        return hits


def _as_int(value: object, *, default: int) -> int:
    """Convert mixed metadata values into an integer with fallback.

    Args:
        value: Metadata value from a backend payload.
        default: Default integer used when conversion fails.

    Returns:
        Coerced integer value.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return int(float(stripped))
        except ValueError:
            return default
    return default


class Neo4jLexicalStore:
    """Neo4j-backed fallback retrieval implementation.

    Despite the name, this class currently behaves as a lexical fallback over
    Neo4j `Chunk` nodes rather than a true vector index. It satisfies the
    `VectorStore` protocol so the rest of the retrieval pipeline can stay
    backend-agnostic.

    Args:
        neo4j_uri: Neo4j connection URI.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str) -> None:
        self._driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def close(self) -> None:
        self._driver.close()

    def upsert(self, documents: list[IngestDocument]) -> int:
        _ = documents
        return 0

    def search(self, query: str, top_k: int) -> list[VectorHit]:
        return _sparse_query_hits(self._driver, query, top_k=top_k)


Neo4jChunkVectorStore = Neo4jLexicalStore


class HybridRetriever:
    """Orchestrates dense, sparse, graph, and hybrid retrieval.

    `HybridRetriever` is the main entry point for callers. It coordinates the
    selected backend, executes retrieval for the requested mode, expands hits
    with local graph context, and returns ranked `RetrievalResult` objects.

    Args:
        neo4j_uri: Neo4j connection URI.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.
        top_k: Maximum number of final results to return.
        vector_store: Optional preconfigured vector store implementation.
        vector_store_backend: Backend name used when `vector_store` is not
            supplied. Supported values are currently `neo4j` and `chroma`.
        chroma_persist_dir: Persistence directory for the Chroma backend.
        embedding_provider: Embedding provider used by dense retrieval.
        retrieval_mode: Default retrieval mode used by `retrieve`.

    Notes:
        This class owns both the vector backend and a Neo4j driver. Call
        `close()` when you are done with it, or use it as a context manager.
    """

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
        """Index chunk documents in the configured vector backend.

        Args:
            documents: Chunked ingestion documents to upsert.

        Returns:
            Number of indexed documents.
        """
        return self._vector_store.upsert(documents)

    def close(self) -> None:
        """Release backend resources held by the retriever."""
        if hasattr(self._vector_store, "close"):
            self._vector_store.close()  # type: ignore[call-arg]
        self._driver.close()

    def __enter__(self) -> HybridRetriever:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def retrieve(self, query: str, mode_override: RetrievalMode | None = None) -> list[RetrievalResult]:
        """Retrieve ranked evidence for a user query.

        Args:
            query: Natural-language question or search string.
            mode_override: Optional retrieval mode to use for this call instead
                of the instance default.

        Returns:
            A ranked list of retrieval results enriched with graph context.

        Notes:
            Retrieval happens in two stages:
            1. Collect initial hits from the selected mode.
            2. Expand each hit with local graph evidence and optionally apply a
               lightweight rerank boost.

            In `graph` mode, the retriever seeds from matching entities and then
            performs one-hop expansion through selected domain relationships.
            In `hybrid_rerank` mode, dense and sparse hits are merged from a
            wider candidate set before graph-context and query-coverage boosts.
        """
        retrieval_mode = mode_override or self._retrieval_mode
        logger.info("Retrieving for query: %r (top_k=%d mode=%s)", query, self._top_k, retrieval_mode)

        if retrieval_mode == "dense":
            hits = self._vector_store.search(query, top_k=self._top_k)
        elif retrieval_mode == "sparse":
            hits = _sparse_query_hits(self._driver, query, top_k=self._top_k)
        elif retrieval_mode == "graph":
            # Prefer promoted assertion-aware traversal, fallback to legacy graph seeds.
            hits = _promoted_graph_seed_hits(self._driver, query, top_k=self._top_k)
            if not hits:
                hits = _graph_seed_hits(self._driver, query, top_k=self._top_k)

            hop_hits = _promoted_graph_hop_expansion(self._driver, query, top_k=self._top_k)
            if not hop_hits:
                hop_hits = _graph_hop_expansion(self._driver, query, top_k=self._top_k)
            if hop_hits:
                hits = _merge_hits(hits, hop_hits, top_k=self._top_k * 2)[: self._top_k]
        else:
            # Use a broader candidate pool in hybrid mode before final rerank.
            candidate_k = max(self._top_k * 2, self._top_k)
            dense_hits = self._vector_store.search(query, top_k=candidate_k)
            sparse_hits = _sparse_query_hits(self._driver, query, top_k=candidate_k)
            hits = _merge_hits(dense_hits, sparse_hits, top_k=candidate_k)

        if not hits:
            return []

        results: list[RetrievalResult] = []
        with self._driver.session() as session:
            for hit in hits:
                results.append(_graph_expand(hit, session))

        if retrieval_mode == "hybrid_rerank":
            query_tokens = set(_query_tokens(query))
            for result in results:
                entity_count = len(result.related_entities)
                neighbour_count = len(result.graph_neighbours)
                entity_signal = min(1.0, math.log1p(entity_count) / math.log(6.0))
                neighbour_signal = min(1.0, math.log1p(neighbour_count) / math.log(8.0))
                coverage_signal = _query_coverage_signal(query_tokens, result)
                evidence_boost = (0.11 * entity_signal) + (0.07 * neighbour_signal) + (0.09 * coverage_signal)
                result.score = round((0.85 * float(result.score)) + evidence_boost, 6)

        results.sort(key=lambda item: item.score, reverse=True)
        return results[: self._top_k]


def _query_tokens(query: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", query.lower())
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped[:12]


def _query_tokens_for_lexical_graph(query: str) -> list[str]:
    """Return sparse/graph tokens with lightweight synonym expansion.

    Args:
        query: Natural-language query text.

    Returns:
        Deduplicated lexical tokens plus a bounded set of synonym tokens.
    """
    base_tokens = _query_tokens(query)
    expanded: list[str] = []
    seen: set[str] = set()

    for token in base_tokens:
        if token not in seen:
            expanded.append(token)
            seen.add(token)

        for synonym in _LEXICAL_TOKEN_SYNONYMS.get(token, ()):
            if synonym in seen:
                continue
            expanded.append(synonym)
            seen.add(synonym)

    return expanded[:24]


def _query_coverage_signal(query_tokens: set[str], result: RetrievalResult) -> float:
    """Estimate how much a result directly covers query terms.

    Args:
        query_tokens: Deduplicated query tokens.
        result: Retrieved and graph-expanded result.

    Returns:
        Ratio in ``[0.0, 1.0]`` for token overlap between query and result.
    """
    if not query_tokens:
        return 0.0
    content_tokens = set(_query_tokens(result.content))
    entity_tokens = {token for entity in result.related_entities for token in _query_tokens(entity)}
    support_tokens = content_tokens | entity_tokens
    if not support_tokens:
        return 0.0
    overlap = len(query_tokens & support_tokens)
    return min(1.0, overlap / len(query_tokens))


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
                "Falling back to Neo4jLexicalStore because Chroma is unavailable: %s",
                exc,
            )

    return Neo4jLexicalStore(
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
    """Attach lightweight local graph context to a hit.

    Expansion is intentionally shallow: it gathers directly mentioned entities
    and nearby chunks from the same source node. The goal is explainability and
    mild reranking support, not multi-hop graph reasoning.
    """
    if isinstance(hit_or_result, RetrievalResult) and session is None:
        logger.debug("_graph_expand compatibility path for %s", hit_or_result.source_path)
        return hit_or_result

    hit = cast(VectorHit, hit_or_result)
    if session is None:
        raise ValueError("session is required when expanding a VectorHit")

    promoted_cypher = (
        "MATCH (c:Chunk) "
        "WHERE coalesce(c.node_id, c.name) = $chunk_id "
        "OPTIONAL MATCH (a:Assertion)-[:SUPPORTED_BY]->(c) "
        "OPTIONAL MATCH (a)-[:ASSERTS_SUBJECT]->(s:CanonicalEntity) "
        "OPTIONAL MATCH (a)-[:ASSERTS_OBJECT]->(o:CanonicalEntity) "
        "OPTIONAL MATCH (d:SourceDocument)-[:HAS_CHUNK]->(c) "
        "OPTIONAL MATCH (d)-[:HAS_CHUNK]->(near:Chunk) "
        "WHERE coalesce(near.node_id, near.name) <> coalesce(c.node_id, c.name) "
        "RETURN collect(DISTINCT s.preferred_label)[0..20] AS subject_entities, "
        "collect(DISTINCT o.preferred_label)[0..20] AS object_entities, "
        "collect(DISTINCT coalesce(near.node_id, near.name))[0..10] AS neighbour_chunks"
    )
    legacy_cypher = (
        "MATCH (c:Chunk {name: $chunk_id}) "
        "OPTIONAL MATCH (c)-[:MENTIONS]->(e) "
        "OPTIONAL MATCH (src)-[:HAS_CHUNK]->(c) "
        "OPTIONAL MATCH (src)-[:HAS_CHUNK]->(near:Chunk) WHERE near.name <> c.name "
        "RETURN collect(DISTINCT e.name)[0..20] AS entities, "
        "collect(DISTINCT near.name)[0..10] AS neighbour_chunks"
    )

    row = session.run(promoted_cypher, chunk_id=hit.chunk_id).single()
    subject_entities = [str(item) for item in (row["subject_entities"] if row else []) if item]
    object_entities = [str(item) for item in (row["object_entities"] if row else []) if item]
    neighbour_chunks = [str(item) for item in (row["neighbour_chunks"] if row else []) if item]

    entities = sorted(set(subject_entities + object_entities))
    if not entities and not neighbour_chunks:
        row = session.run(legacy_cypher, chunk_id=hit.chunk_id).single()
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
    tokens = _query_tokens_for_lexical_graph(query)
    if not tokens:
        return []

    cypher = (
        "MATCH (c:Chunk) "
        "WITH c, "
        "toLower(coalesce(c.content, '')) + ' ' + "
        "toLower(coalesce(c.section, '')) + ' ' + "
        "toLower(coalesce(c.title, '')) AS haystack "
        "WITH c, [t IN $tokens WHERE haystack CONTAINS t] AS matched "
        "WITH c, matched, size(matched) AS score "
        "WHERE score > 0 "
        "RETURN coalesce(c.node_id, c.name) AS chunk_id, "
        "coalesce(c.content, '') AS content, "
        "coalesce(c.source_path, '') AS source_path, "
        "coalesce(c.relative_path, c.source_path, '') AS relative_path, "
        "coalesce(c.chunk_index, 0) AS chunk_index, "
        "coalesce(c.chunk_kind, c.section, '') AS chunk_kind, "
        "coalesce(c.line_start, 1) AS line_start, "
        "coalesce(c.line_end, 1) AS line_end, score "
        "ORDER BY score DESC LIMIT $top_k"
    )

    with driver.session() as session:
        rows = list(session.run(cypher, tokens=tokens, top_k=top_k))

    hits: list[VectorHit] = []
    for row in rows:
        hits.append(
            VectorHit(
                chunk_id=str(row["chunk_id"] or ""),
                content=str(row["content"] or ""),
                source_path=str(row["source_path"] or ""),
                score=float(row["score"]),
                metadata={
                    "relative_path": str(row["relative_path"] or ""),
                    "chunk_index": int(row["chunk_index"] or 0),
                    "chunk_kind": str(row["chunk_kind"] or ""),
                    "section": "",
                    "line_start": int(row["line_start"] or 1),
                    "line_end": int(row["line_end"] or 1),
                    "content_hash": "",
                },
            )
        )
    return hits


def _graph_seed_hits(driver: Driver, query: str, top_k: int) -> list[VectorHit]:
    tokens = _query_tokens_for_lexical_graph(query)
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


def _find_domain_concepts(query: str) -> list[str]:
    """Return canonical concept names whose aliases appear in *query*."""
    lowered = query.lower()
    matched: list[str] = []
    for concepts in _LEGACY_DOMAIN_ALIASES.values():
        for canonical_name, aliases in concepts.items():
            candidate_aliases = tuple(alias.lower() for alias in aliases) + _QUERY_ONLY_CONCEPT_ALIASES.get(
                canonical_name,
                (),
            )
            if canonical_name.lower() in lowered or any(alias in lowered for alias in candidate_aliases):
                matched.append(canonical_name)
    return matched


def _promoted_graph_seed_hits(driver: Driver, query: str, top_k: int) -> list[VectorHit]:
    """Seed graph hits from promoted canonical entities and assertions."""
    tokens = _query_tokens_for_lexical_graph(query)
    if not tokens:
        return []

    cypher = (
        "MATCH (e:CanonicalEntity) "
        "WHERE any(t IN $tokens WHERE toLower(e.preferred_label) CONTAINS t "
        "OR toLower(coalesce(e.normalized_label, '')) CONTAINS t) "
        "OPTIONAL MATCH (a:Assertion)-[:ASSERTS_SUBJECT|ASSERTS_OBJECT]->(e) "
        "OPTIONAL MATCH (a)-[:SUPPORTED_BY]->(c:Chunk) "
        "WITH c, max(coalesce(a.confidence, 0.0)) AS score "
        "WHERE c IS NOT NULL "
        "RETURN coalesce(c.node_id, c.name) AS chunk_id, "
        "coalesce(c.content, '') AS content, "
        "coalesce(c.source_path, '') AS source_path, "
        "coalesce(c.relative_path, c.source_path, '') AS relative_path, "
        "coalesce(c.chunk_index, 0) AS chunk_index, "
        "coalesce(c.chunk_kind, c.section, '') AS chunk_kind, "
        "coalesce(c.line_start, 1) AS line_start, "
        "coalesce(c.line_end, 1) AS line_end, score "
        "ORDER BY score DESC LIMIT $top_k"
    )

    with driver.session() as session:
        rows = list(session.run(cypher, tokens=tokens, top_k=top_k))

    hits: list[VectorHit] = []
    for row in rows:
        hits.append(
            VectorHit(
                chunk_id=str(row["chunk_id"] or ""),
                content=str(row["content"] or ""),
                source_path=str(row["source_path"] or ""),
                score=float(row["score"]),
                metadata={
                    "relative_path": str(row["relative_path"] or ""),
                    "chunk_index": int(row["chunk_index"] or 0),
                    "chunk_kind": str(row["chunk_kind"] or ""),
                    "section": "",
                    "line_start": int(row["line_start"] or 1),
                    "line_end": int(row["line_end"] or 1),
                    "content_hash": "",
                },
            )
        )
    return hits


def _promoted_graph_hop_expansion(driver: Driver, query: str, top_k: int) -> list[VectorHit]:
    """Expand graph hits through ontology class neighborhoods in promoted graph."""
    tokens = _query_tokens_for_lexical_graph(query)
    if not tokens:
        return []

    cypher = (
        "MATCH (oc:OntologyClass)<-[:INSTANCE_OF]-(e:CanonicalEntity) "
        "WHERE any(t IN $tokens WHERE toLower(oc.label) CONTAINS t "
        "OR toLower(coalesce(oc.definition, '')) CONTAINS t) "
        "MATCH (a:Assertion)-[:ASSERTS_SUBJECT|ASSERTS_OBJECT]->(e) "
        "MATCH (a)-[:SUPPORTED_BY]->(c:Chunk) "
        "RETURN coalesce(c.node_id, c.name) AS chunk_id, "
        "coalesce(c.content, '') AS content, "
        "coalesce(c.source_path, '') AS source_path, "
        "coalesce(c.relative_path, c.source_path, '') AS relative_path, "
        "coalesce(c.chunk_index, 0) AS chunk_index, "
        "coalesce(c.chunk_kind, c.section, '') AS chunk_kind, "
        "coalesce(c.line_start, 1) AS line_start, "
        "coalesce(c.line_end, 1) AS line_end, "
        "max(coalesce(a.confidence, 0.0)) AS score "
        "ORDER BY score DESC LIMIT $top_k"
    )

    try:
        with driver.session() as session:
            rows = list(session.run(cypher, tokens=tokens, top_k=top_k))
    except Exception as exc:
        logger.debug("Promoted graph-hop expansion query failed: %s", exc)
        return []

    hits: list[VectorHit] = []
    for row in rows:
        hits.append(
            VectorHit(
                chunk_id=str(row["chunk_id"] or ""),
                content=str(row["content"] or ""),
                source_path=str(row["source_path"] or ""),
                score=float(row["score"]),
                metadata={
                    "relative_path": str(row["relative_path"] or ""),
                    "chunk_index": int(row["chunk_index"] or 0),
                    "chunk_kind": str(row["chunk_kind"] or ""),
                    "section": "",
                    "line_start": int(row["line_start"] or 1),
                    "line_end": int(row["line_end"] or 1),
                    "content_hash": "",
                },
            )
        )
    return hits


def _graph_hop_expansion(driver: Driver, query: str, top_k: int) -> list[VectorHit]:
    """Find additional graph-relevant chunks from one-hop domain relations.

    This expansion is limited to curated ontology-like edges so retrieval
    remains interpretable and bounded in cost.
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
        "OPTIONAL MATCH (e)-[:BELONGS_TO_FAMILY]->(family:RiskMeasureFamily) "
        "OPTIONAL MATCH (family)<-[:BELONGS_TO_FAMILY]-(family_peer:RiskMeasure) "
        "OPTIONAL MATCH (e)<-[:RANGE_VERSION_OF]-(range_variant:RiskMeasure) "
        "OPTIONAL MATCH (e)-[:RANGE_VERSION_OF]->(range_base:RiskMeasure) "
        "OPTIONAL MATCH (e)<-[:DRAWDOWN_ANALOG_OF]-(drawdown_variant:RiskMeasure) "
        "OPTIONAL MATCH (e)-[:DRAWDOWN_ANALOG_OF]->(return_analog:RiskMeasure) "
        "WITH collect(DISTINCT parent) + collect(DISTINCT alt) + collect(DISTINCT req) + "
        "collect(DISTINCT family) + collect(DISTINCT family_peer) + collect(DISTINCT range_variant) + "
        "collect(DISTINCT range_base) + collect(DISTINCT drawdown_variant) + "
        "collect(DISTINCT return_analog) AS related_nodes "
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
    """Merge dense and sparse hit lists into a single ranking.

    Uses reciprocal-rank fusion (RRF) as the primary signal, then blends a
    calibrated score channel and overlap bonus. This avoids brittle behavior
    when dense/sparse raw scores live on different scales.
    """
    if top_k <= 0:
        return []

    merged: dict[str, VectorHit] = {}
    dense_rank = {hit.chunk_id: index + 1 for index, hit in enumerate(dense_hits)}
    sparse_rank = {hit.chunk_id: index + 1 for index, hit in enumerate(sparse_hits)}
    dense_score = _normalize_scores({hit.chunk_id: float(hit.score) for hit in dense_hits})
    sparse_score = _normalize_scores({hit.chunk_id: float(hit.score) for hit in sparse_hits})

    for hit in dense_hits + sparse_hits:
        key = hit.chunk_id
        if key in merged:
            existing = merged[key]
            if not existing.content:
                existing.content = hit.content
            if not existing.source_path:
                existing.source_path = hit.source_path
            if not existing.metadata:
                existing.metadata = dict(hit.metadata)
            continue
        merged[key] = VectorHit(
            chunk_id=hit.chunk_id,
            content=hit.content,
            source_path=hit.source_path,
            score=0.0,
            metadata=dict(hit.metadata),
        )

    rrf_k = 60.0
    for chunk_id, hit in merged.items():
        rrf = 0.0
        if chunk_id in dense_rank:
            rrf += 1.0 / (rrf_k + dense_rank[chunk_id])
        if chunk_id in sparse_rank:
            rrf += 1.0 / (rrf_k + sparse_rank[chunk_id])

        dense_component = dense_score.get(chunk_id, 0.0)
        sparse_component = sparse_score.get(chunk_id, 0.0)
        channel_count = int(chunk_id in dense_rank) + int(chunk_id in sparse_rank)
        calibrated = (dense_component + sparse_component) / max(channel_count, 1)
        overlap_bonus = 0.22 if channel_count == 2 else 0.0
        hit.score = (0.55 * rrf) + (0.23 * calibrated) + overlap_bonus

    ordered = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    return ordered[:top_k]


def _normalize_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize arbitrary backend scores to [0, 1]."""
    if not raw_scores:
        return {}

    values = list(raw_scores.values())
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return {key: 1.0 for key in raw_scores}
    return {key: (score - minimum) / (maximum - minimum) for key, score in raw_scores.items()}
