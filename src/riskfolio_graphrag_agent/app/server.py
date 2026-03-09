"""FastAPI application server for the GraphRAG agent."""

from __future__ import annotations

import logging
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from riskfolio_graphrag_agent.config.settings import Settings
from riskfolio_graphrag_agent.graph.builder import GraphBuilder
from riskfolio_graphrag_agent.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Request payload for question answering."""

    question: str = Field(min_length=1, description="Natural-language question.")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum chunks to return.")


class QueryResponse(BaseModel):
    """Minimal query response payload."""

    answer: str
    citations: list[dict[str, str | int | float | list[str]]]


def _extract_query_tokens(question: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", question.lower())
    seen: set[str] = set()
    deduped: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped[:12]


def _as_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def create_app() -> FastAPI:
    """Create and return a configured FastAPI application instance.

    Returns:
        A configured ``FastAPI`` application object.
    """
    app = FastAPI(title="riskfolio-graphrag-agent", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/graph/stats")
    def graph_stats() -> dict[str, int | dict[str, int]]:
        settings = Settings()
        builder = GraphBuilder(
            neo4j_uri=settings.neo4j_uri,
            neo4j_user=settings.neo4j_user,
            neo4j_password=settings.neo4j_password,
        )
        try:
            return builder.get_stats()
        except Exception as exc:
            logger.exception("Failed to retrieve graph stats")
            raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {exc}") from exc
        finally:
            builder.close()

    @app.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest) -> QueryResponse:
        settings = Settings()
        tokens = _extract_query_tokens(payload.question)
        if not tokens:
            raise HTTPException(status_code=400, detail="Question must contain searchable text.")

        retriever = HybridRetriever(
            neo4j_uri=settings.neo4j_uri,
            neo4j_user=settings.neo4j_user,
            neo4j_password=settings.neo4j_password,
            top_k=payload.top_k,
            vector_store_backend=settings.vector_store_backend,
            chroma_persist_dir=settings.chroma_persist_dir,
            embedding_dim=settings.embedding_dim,
        )

        try:
            results = retriever.retrieve(payload.question)
        except Exception as exc:
            logger.exception("Failed to execute query endpoint")
            raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {exc}") from exc
        finally:
            retriever.close()

        citations: list[dict[str, str | int | float | list[str]]] = []
        for item in results:
            metadata = item.metadata
            citations.append(
                {
                    "chunk_id": str(metadata.get("chunk_id", "")),
                    "source_path": item.source_path,
                    "relative_path": str(metadata.get("relative_path", "")),
                    "chunk_index": _as_int(metadata.get("chunk_index", 0), 0),
                    "section": str(metadata.get("section", "")),
                    "line_start": _as_int(metadata.get("line_start", 1), 1),
                    "line_end": _as_int(metadata.get("line_end", 1), 1),
                    "score": float(item.score),
                    "matched_entities": item.related_entities,
                    "graph_neighbours": item.graph_neighbours,
                }
            )

        if not citations:
            return QueryResponse(
                answer="I could not find matching graph context for that question yet.",
                citations=[],
            )

        top_entities = citations[0]["matched_entities"]
        entity_preview = ", ".join(top_entities[:5]) if isinstance(top_entities, list) else ""
        if not entity_preview:
            entity_preview = "no explicit entities"
        answer = (
            f"I found {len(citations)} ranked hybrid contexts for '{payload.question}'. "
            f"Top matched entities include: {entity_preview}."
        )
        return QueryResponse(answer=answer, citations=citations)

    return app
