"""FastAPI application server for the GraphRAG agent."""

from __future__ import annotations

import logging
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from riskfolio_graphrag_agent.config.settings import Settings
from riskfolio_graphrag_agent.graph.builder import GraphBuilder

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Request payload for question answering."""

    question: str = Field(min_length=1, description="Natural-language question.")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum chunks to return.")


class QueryResponse(BaseModel):
    """Minimal query response payload."""

    answer: str
    citations: list[dict[str, str | int | list[str]]]


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

        builder = GraphBuilder(
            neo4j_uri=settings.neo4j_uri,
            neo4j_user=settings.neo4j_user,
            neo4j_password=settings.neo4j_password,
        )

        cypher = (
            "MATCH (d:Document)-[:MENTIONS]->(e) "
            "WHERE any(token IN $tokens WHERE toLower(e.name) CONTAINS token) "
            "WITH d, collect(DISTINCT e.name)[0..10] AS matched_entities, count(*) AS score "
            "ORDER BY score DESC LIMIT $top_k "
            "RETURN d.name AS document, d.source_path AS source_path, "
            "d.chunk_index AS chunk_index, matched_entities, score"
        )

        try:
            driver = builder._ensure_driver()
            with driver.session() as session:
                rows = list(session.run(cypher, tokens=tokens, top_k=payload.top_k))
        except Exception as exc:
            logger.exception("Failed to execute query endpoint")
            raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {exc}") from exc
        finally:
            builder.close()

        citations: list[dict[str, str | int | list[str]]] = [
            {
                "document": str(row["document"]),
                "source_path": str(row["source_path"]),
                "chunk_index": int(row["chunk_index"]),
                "matched_entities": list(row["matched_entities"]),
                "score": int(row["score"]),
            }
            for row in rows
        ]

        if not citations:
            return QueryResponse(
                answer="I could not find matching graph context for that question yet.",
                citations=[],
            )

        entity_preview = ", ".join(citations[0]["matched_entities"][:5])
        answer = (
            f"I found {len(citations)} matching graph chunks for '{payload.question}'. "
            f"Top matched entities include: {entity_preview}."
        )
        return QueryResponse(answer=answer, citations=citations)

    return app
