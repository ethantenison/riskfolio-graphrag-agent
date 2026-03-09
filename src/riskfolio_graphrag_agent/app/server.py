"""FastAPI application server for the GraphRAG agent."""

from __future__ import annotations

import json
import logging
import re
import ssl
from urllib import request
from urllib.error import HTTPError, URLError

# Observability imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

try:
    import certifi
except Exception:  # pragma: no cover - optional dependency fallback
    certifi = None

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from riskfolio_graphrag_agent.agent.workflow import AgentWorkflow
from riskfolio_graphrag_agent.config.settings import Settings
from riskfolio_graphrag_agent.graph.builder import GraphBuilder
from riskfolio_graphrag_agent.retrieval.retriever import HybridRetriever, RetrievalResult
from riskfolio_graphrag_agent.runtime_ssl import initialize_ssl_truststore_once

logger = logging.getLogger(__name__)


def _build_ssl_context() -> ssl.SSLContext | None:
    if initialize_ssl_truststore_once():
        return None
    if certifi is None:
        return None
    return ssl.create_default_context(cafile=certifi.where())


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


def _build_context_preview(context: list[RetrievalResult], max_items: int = 5) -> str:
    lines: list[str] = []
    for index, item in enumerate(context[:max_items], start=1):
        section = str(item.metadata.get("section", "")).strip()
        chunk_id = str(item.metadata.get("chunk_id", "")).strip()
        label = section or chunk_id or item.source_path
        snippet = " ".join(item.content.split())[:500]
        lines.append(f"[{index}] source={item.source_path} section={label} evidence={snippet}")
    return "\n".join(lines)


def _extract_openai_message_text(payload: dict[str, object]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
    return ""


def _make_openai_llm_generate(settings: Settings):
    def _generate(*, question: str, context: list[RetrievalResult], model_name: str) -> str:
        prompt = (
            "You are a GraphRAG assistant for Riskfolio-Lib. "
            "Answer the question strictly using the provided evidence. "
            "If evidence is insufficient, say that directly and do not invent facts.\n\n"
            f"Question:\n{question}\n\n"
            f"Evidence:\n{_build_context_preview(context)}\n\n"
            "Return a concise answer with factual claims grounded in the evidence."
        )

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You produce grounded technical answers and avoid unsupported claims."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        base_url = settings.openai_base_url.rstrip("/")
        endpoint = f"{base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url=endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with request.urlopen(
                http_request,
                timeout=settings.openai_timeout_seconds,
                context=_build_ssl_context(),
            ) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM HTTP error {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"LLM endpoint unreachable: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM returned non-JSON response") from exc

        answer = _extract_openai_message_text(data)
        if not answer:
            raise RuntimeError("LLM response did not contain assistant content")
        return answer

    return _generate


def create_app() -> FastAPI:
    """Create and return a configured FastAPI application instance.

    Returns:
        A configured ``FastAPI`` application object.
    """
    initialize_ssl_truststore_once()
    app = FastAPI(title="riskfolio-graphrag-agent", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/trace")
    def trace_info() -> dict[str, str]:
        """Return a simple trace status for observability demo."""
        tracer = trace.get_tracer("riskfolio-graphrag-agent")
        return {"tracer": str(tracer), "provider": str(trace.get_tracer_provider())}

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

        llm_generate = None
        if settings.openai_enable_generation and settings.openai_api_key.strip():
            llm_generate = _make_openai_llm_generate(settings)

        workflow = AgentWorkflow(
            retriever=retriever,
            model_name=settings.openai_model,
            llm_generate=llm_generate,
        )

        try:
            state = workflow.run(payload.question)
        except Exception as exc:
            logger.exception("Failed to execute query endpoint")
            raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {exc}") from exc
        finally:
            retriever.close()

        citations = state.citations

        if not citations:
            return QueryResponse(
                answer="I could not find matching graph context for that question yet.",
                citations=[],
            )

        return QueryResponse(answer=state.answer, citations=citations)

    return app
