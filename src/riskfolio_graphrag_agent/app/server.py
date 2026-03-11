"""FastAPI application server for the GraphRAG agent."""

from __future__ import annotations

import json
import logging
import re
import ssl
import uuid
from urllib import request
from urllib.error import HTTPError, URLError

from fastapi import FastAPI, HTTPException
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel, Field

from riskfolio_graphrag_agent.agent.workflow import AgentWorkflow
from riskfolio_graphrag_agent.config.settings import Settings
from riskfolio_graphrag_agent.graph.builder import DOMAIN_ALIASES, GraphBuilder
from riskfolio_graphrag_agent.graph.nl2cypher_guard import append_query_audit, guarded_nl_to_cypher
from riskfolio_graphrag_agent.retrieval.embeddings import resolve_embedding_provider
from riskfolio_graphrag_agent.retrieval.retriever import HybridRetriever, RetrievalResult
from riskfolio_graphrag_agent.retrieval.router import QueryToolRouter
from riskfolio_graphrag_agent.runtime_ssl import initialize_ssl_truststore_once

try:
    import certifi
except Exception:  # pragma: no cover - optional dependency fallback
    certifi = None

logger = logging.getLogger(__name__)


def _configure_tracing(settings: Settings) -> None:
    if not settings.tracing_enabled:
        return

    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        return

    trace.set_tracer_provider(TracerProvider())
    otlp_exporter = OTLPSpanExporter(
        endpoint=settings.tracing_otlp_endpoint,
        insecure=settings.tracing_otlp_insecure,
    )
    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))


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


class NLToCypherRequest(BaseModel):
    question: str = Field(min_length=1)
    tenant_id: str = Field(default="demo-tenant")


class NLToCypherResponse(BaseModel):
    status: str
    reason: str
    requires_human_review: bool
    cypher: str = ""
    params: dict[str, str] = Field(default_factory=dict)
    rows: list[dict[str, str | int | float]] = Field(default_factory=list)


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


def _is_definition_question(question: str) -> bool:
    lowered = question.strip().lower()
    patterns = (
        r"^what\s+is\s+",
        r"^define\s+",
        r"^what\s+does\s+.*\s+mean\??$",
        r"^meaning\s+of\s+",
    )
    return any(re.search(pattern, lowered) is not None for pattern in patterns)


def _build_background_hint(question: str) -> str:
    lowered = question.lower()
    for concepts in DOMAIN_ALIASES.values():
        for canonical_name, aliases in concepts.items():
            for alias in aliases:
                if alias.lower() in lowered:
                    return f"{canonical_name}: {alias}"
            if canonical_name.lower() in lowered:
                return canonical_name
    return ""


def _make_openai_llm_generate(settings: Settings):
    def _generate(*, question: str, context: list[RetrievalResult], model_name: str) -> str:
        definition_mode = _is_definition_question(question)
        background_hint = _build_background_hint(question)

        response_policy = (
            "Return a concise answer with factual claims grounded in the evidence."
            if not definition_mode
            else (
                "If direct definition text is missing from evidence, provide two sections:\n"
                "1) General background (label exactly: 'General background:') with a short standard definition.\n"
                "2) Repo evidence (label exactly: 'Repo evidence:') summarizing what retrieved sources confirm.\n"
                "Do not fabricate repository-specific claims; if repo evidence is thin, state that explicitly in section 2."
            )
        )

        hint_block = f"\n\nDefinition hint:\n{background_hint}" if background_hint else ""
        prompt = (
            "You are a GraphRAG assistant for Riskfolio-Lib. "
            "Prefer evidence-grounded answers. "
            "Do not invent repository-specific facts.\n\n"
            f"Question:\n{question}\n\n"
            f"Evidence:\n{_build_context_preview(context)}\n\n"
            f"{response_policy}"
            f"{hint_block}"
        )

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": ("You produce grounded technical answers and avoid unsupported claims."),
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
    settings = Settings()
    _configure_tracing(settings)
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
            embedding_provider=resolve_embedding_provider(
                provider_name=settings.embedding_provider,
                embedding_dim=settings.embedding_dim,
                openai_api_key=settings.openai_api_key,
                openai_embedding_model=settings.embedding_model,
                openai_base_url=settings.openai_base_url,
                openai_timeout_seconds=settings.openai_timeout_seconds,
                ssl_context=_build_ssl_context(),
            ).provider,
            retrieval_mode=settings.retrieval_mode,
        )

        llm_generate = None
        if settings.openai_enable_generation and settings.openai_api_key.strip():
            llm_generate = _make_openai_llm_generate(settings)

        query_router = None
        if settings.adaptive_tool_routing_enabled:
            query_router = QueryToolRouter(
                min_confidence=settings.adaptive_tool_routing_min_confidence,
            )

        workflow = AgentWorkflow(
            retriever=retriever,
            model_name=settings.openai_model,
            llm_generate=llm_generate,
            query_router=query_router,
        )

        tracer = trace.get_tracer("riskfolio-graphrag-agent")
        with tracer.start_as_current_span("api.query") as span:
            request_id = str(uuid.uuid4())
            span.set_attribute("tenant.id", settings.default_tenant_id)
            span.set_attribute("request.id", request_id)
            span.set_attribute("model.name", settings.openai_model)
            span.set_attribute("retrieval.mode", settings.retrieval_mode)
            span.set_attribute("workflow.stage", "query")

            try:
                state = workflow.run(payload.question)
            except Exception as exc:
                logger.exception("Failed to execute query endpoint")
                raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {exc}") from exc
            finally:
                retriever.close()

            estimated_cost = max(0.0, len(payload.question.split()) * 0.000001)
            span.set_attribute("cost.estimated_usd", estimated_cost)

        citations = state.citations

        if not citations:
            return QueryResponse(
                answer="I could not find matching graph context for that question yet.",
                citations=[],
            )

        return QueryResponse(answer=state.answer, citations=citations)

    @app.post("/graph/nl2cypher", response_model=NLToCypherResponse)
    def nl2cypher(payload: NLToCypherRequest) -> NLToCypherResponse:
        settings = Settings()
        request_id = str(uuid.uuid4())
        decision = guarded_nl_to_cypher(payload.question)

        append_query_audit(
            tenant_id=payload.tenant_id or settings.default_tenant_id,
            request_id=request_id,
            question=payload.question,
            decision=decision,
            audit_path=settings.cypher_audit_log_path,
        )

        if decision.status != "safe":
            return NLToCypherResponse(
                status=decision.status,
                reason=decision.reason,
                requires_human_review=decision.requires_human_review,
                cypher=decision.cypher,
                params=decision.params or {},
                rows=[],
            )

        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
        try:
            with driver.session() as session:
                records = list(session.run(decision.cypher, **(decision.params or {})))
                rows = [dict(row.data()) for row in records]
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {exc}") from exc
        finally:
            driver.close()

        return NLToCypherResponse(
            status=decision.status,
            reason=decision.reason,
            requires_human_review=decision.requires_human_review,
            cypher=decision.cypher,
            params=decision.params or {},
            rows=rows,
        )

    return app
