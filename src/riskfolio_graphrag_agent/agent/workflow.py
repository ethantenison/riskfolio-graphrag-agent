"""Agent workflow implemented with LangGraph state transitions.

This module provides a small but real LangGraph implementation with nodes for
planning, retrieval, reasoning, and verification, plus a retry loop for
self-correction.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol, TypedDict, cast

from langsmith import traceable
from opentelemetry import trace
from opentelemetry.trace import SpanKind

try:
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency fallback
    END = "__end__"
    START = "__start__"
    StateGraph = None
    LANGGRAPH_AVAILABLE = False

from riskfolio_graphrag_agent.retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Mutable state threaded through each workflow node.

    Attributes:
        question: Original user question.
        sub_questions: Decomposed sub-questions (populated by *plan* node).
        context: Retrieved context chunks (populated by *retrieve* node).
        answer: Draft answer (populated by *reason* node).
        citations: Source references included in the answer.
        verified: Whether the *verify* node accepted the answer.
    """

    question: str
    sub_questions: list[str] = field(default_factory=list)
    context: list[RetrievalResult] = field(default_factory=list)
    answer: str = ""
    citations: list[dict[str, str | int | float | list[str]]] = field(default_factory=list)
    verified: bool = False


class RetrieverProtocol(Protocol):
    def retrieve(self, query: str) -> list[RetrievalResult]: ...


class LLMGenerateProtocol(Protocol):
    def __call__(
        self,
        *,
        question: str,
        context: list[RetrievalResult],
        model_name: str,
    ) -> str: ...


class AgentGraphState(TypedDict):
    question: str
    sub_questions: list[str]
    context: list[RetrievalResult]
    answer: str
    citations: list[dict[str, str | int | float | list[str]]]
    verified: bool
    retry_count: int


class AgentWorkflow:
    """Orchestrates the multi-step reasoning workflow.

    Args:
        retriever: An initialised :class:`~riskfolio_graphrag_agent.retrieval.retriever.HybridRetriever`.
        model_name: LLM model identifier to use for generation.
    """

    def __init__(
        self,
        retriever: RetrieverProtocol | None,
        model_name: str = "gpt-4o-mini",
        llm_generate: LLMGenerateProtocol | None = None,
    ) -> None:
        self._retriever = retriever
        self._model_name = model_name
        self._llm_generate = llm_generate
        self._graph = self._build_graph()

    def _build_graph(self):
        if not LANGGRAPH_AVAILABLE or StateGraph is None:
            logger.warning("LangGraph is not available; using sequential fallback workflow.")
            return None

        graph = StateGraph(AgentGraphState)
        graph.add_node("plan", self._plan_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("reason", self._reason_node)
        graph.add_node("verify", self._verify_node)

        graph.add_edge(START, "plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "reason")
        graph.add_edge("reason", "verify")
        graph.add_conditional_edges(
            "verify",
            self._verification_route,
            {
                "retry": "reason",
                "done": END,
            },
        )
        return graph.compile()

    def run(self, question: str) -> AgentState:
        """Execute the full agent workflow for *question*.

        Args:
            question: Natural-language question from the user.

        Returns:
            A final :class:`AgentState` with answer and citations populated.
        """
        logger.info("AgentWorkflow.run called. question=%r", question)
        tracer = trace.get_tracer("riskfolio-graphrag-agent")
        with tracer.start_as_current_span("AgentWorkflow.run", kind=SpanKind.SERVER) as span:
            span.set_attribute("agent.question", question)

            # LangSmith traceable decorator for full workflow
            @traceable(name="AgentWorkflow.run")
            def _run_inner(q: str) -> AgentState:
                if self._graph is None:
                    state = AgentState(question=q)
                    state = _plan(state)
                    state = _retrieve(state, self._retriever)
                    state = _reason(state, llm_generate=self._llm_generate, model_name=self._model_name)
                    state = _verify(state)
                    if not state.verified and state.context:
                        top = state.context[0]
                        state.answer = (
                            f"I found relevant context but could not fully verify all claims. Top source: {top.source_path}."
                        )
                    return state
                initial_state: AgentGraphState = {
                    "question": q,
                    "sub_questions": [],
                    "context": [],
                    "answer": "",
                    "citations": [],
                    "verified": False,
                    "retry_count": 0,
                }
                final_state = cast(AgentGraphState, self._graph.invoke(initial_state))
                return AgentState(
                    question=final_state["question"],
                    sub_questions=final_state["sub_questions"],
                    context=final_state["context"],
                    answer=final_state["answer"],
                    citations=final_state["citations"],
                    verified=final_state["verified"],
                )

            result = _run_inner(question)
            span.set_attribute("agent.answer_length", len(result.answer))
            span.set_attribute("agent.citation_count", len(result.citations))
            return result

        if self._graph is None:
            state = AgentState(question=question)
            state = _plan(state)
            state = _retrieve(state, self._retriever)
            state = _reason(state, llm_generate=self._llm_generate, model_name=self._model_name)
            state = _verify(state)
            if not state.verified and state.context:
                top = state.context[0]
                state.answer = f"I found relevant context but could not fully verify all claims. Top source: {top.source_path}."
            return state

        initial_state: AgentGraphState = {
            "question": question,
            "sub_questions": [],
            "context": [],
            "answer": "",
            "citations": [],
            "verified": False,
            "retry_count": 0,
        }
        final_state = cast(AgentGraphState, self._graph.invoke(initial_state))
        return AgentState(
            question=final_state["question"],
            sub_questions=final_state["sub_questions"],
            context=final_state["context"],
            answer=final_state["answer"],
            citations=final_state["citations"],
            verified=final_state["verified"],
        )

    def _plan_node(self, state: AgentGraphState) -> AgentGraphState:
        agent_state = AgentState(
            question=state["question"],
            sub_questions=state["sub_questions"],
            context=state["context"],
            answer=state["answer"],
            citations=state["citations"],
            verified=state["verified"],
        )
        planned = _plan(agent_state)
        state["sub_questions"] = planned.sub_questions
        return state

    def _retrieve_node(self, state: AgentGraphState) -> AgentGraphState:
        agent_state = AgentState(
            question=state["question"],
            sub_questions=state["sub_questions"],
            context=state["context"],
            answer=state["answer"],
            citations=state["citations"],
            verified=state["verified"],
        )
        retrieved = _retrieve(agent_state, self._retriever)
        state["context"] = retrieved.context
        return state

    def _reason_node(self, state: AgentGraphState) -> AgentGraphState:
        agent_state = AgentState(
            question=state["question"],
            sub_questions=state["sub_questions"],
            context=state["context"],
            answer=state["answer"],
            citations=state["citations"],
            verified=state["verified"],
        )
        reasoned = _reason(
            agent_state,
            llm_generate=self._llm_generate,
            model_name=self._model_name,
        )
        state["answer"] = reasoned.answer
        state["citations"] = reasoned.citations
        return state

    def _verify_node(self, state: AgentGraphState) -> AgentGraphState:
        agent_state = AgentState(
            question=state["question"],
            sub_questions=state["sub_questions"],
            context=state["context"],
            answer=state["answer"],
            citations=state["citations"],
            verified=state["verified"],
        )
        verified = _verify(agent_state)
        state["verified"] = verified.verified
        if not state["verified"]:
            state["retry_count"] += 1
            if state["retry_count"] == 1 and state["context"]:
                top = state["context"][0]
                state["answer"] = (
                    f"I am revising the response to stay grounded in the retrieved sources. Primary source: {top.source_path}."
                )
        return state

    def _verification_route(self, state: AgentGraphState) -> str:
        if not state["verified"] and state["retry_count"] < 2:
            return "retry"
        return "done"


# ── Stub workflow nodes ────────────────────────────────────────────────────────


def _plan(state: AgentState) -> AgentState:
    """Decompose the question into retrievable sub-questions.

    Args:
        state: Current agent state.

    Returns:
        Updated state with ``sub_questions`` populated.
    """
    focus_terms = _extract_focus_terms(state.question)
    state.sub_questions = [state.question]
    if focus_terms:
        state.sub_questions.append(f"Riskfolio details on {' '.join(focus_terms[:4])}")
    if "how" in state.question.lower() or "why" in state.question.lower():
        state.sub_questions.append(f"methodology evidence for: {state.question}")

    deduped: list[str] = []
    seen: set[str] = set()
    for sub_question in state.sub_questions:
        normalized = sub_question.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(sub_question.strip())
    state.sub_questions = deduped[:3]
    return state


def _retrieve(state: AgentState, retriever: RetrieverProtocol | None) -> AgentState:
    """Retrieve relevant context for each sub-question.

    Args:
        state: Current agent state.
        retriever: Initialised :class:`~riskfolio_graphrag_agent.retrieval.retriever.HybridRetriever`.

    Returns:
        Updated state with ``context`` populated.
    """
    if retriever is None:
        state.context = []
        return state

    merged: dict[str, RetrievalResult] = {}
    for sub_question in state.sub_questions:
        try:
            results = retriever.retrieve(sub_question)
        except Exception as exc:
            logger.warning("Retriever failed for sub-question %r: %s", sub_question, exc)
            continue

        for result in results:
            chunk_id = str(result.metadata.get("chunk_id", ""))
            key = chunk_id or f"{result.source_path}::{len(merged)}"
            existing = merged.get(key)
            if existing is None or result.score > existing.score:
                merged[key] = result

    state.context = sorted(merged.values(), key=lambda item: item.score, reverse=True)[:5]
    return state


def _reason(
    state: AgentState,
    llm_generate: LLMGenerateProtocol | None = None,
    model_name: str = "gpt-4o-mini",
) -> AgentState:
    """Generate a draft answer with chain-of-thought over retrieved context.

    Args:
        state: Current agent state with context populated.

    Returns:
        Updated state with ``answer`` and ``citations`` populated.
    """
    if not state.context:
        state.answer = "I could not find supporting context for that question yet."
        state.citations = []
        return state

    generated_answer = ""
    if llm_generate is not None:
        try:
            generated_answer = llm_generate(
                question=state.question,
                context=state.context,
                model_name=model_name,
            ).strip()
        except Exception as exc:
            logger.warning("LLM generation failed; using fallback reasoner: %s", exc)

    state.answer = generated_answer or _fallback_reason_answer(state)

    state.citations = []
    for item in state.context:
        metadata = item.metadata
        state.citations.append(
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
    return state


def _fallback_reason_answer(state: AgentState) -> str:
    top = state.context[0]
    entity_preview = ", ".join(top.related_entities[:5]) if top.related_entities else ""
    if not entity_preview:
        entity_preview = "no explicit entities"

    sections = [str(item.metadata.get("section", "")).strip() for item in state.context]
    section_preview = ", ".join([section for section in sections if section][:3])

    answer = (
        f"For '{state.question}', retrieved evidence indicates key concepts: {entity_preview}. "
        f"Primary source is {top.source_path}."
    )
    if section_preview:
        answer += f" Relevant sections include: {section_preview}."
    return answer


def _verify(state: AgentState) -> AgentState:
    """Self-check whether the answer is grounded in the retrieved context.

    Args:
        state: Current agent state with answer populated.

    Returns:
        Updated state with ``verified`` set.
    """
    if not state.answer or not state.context:
        state.verified = False
        return state

    answer_tokens = set(_tokens(state.answer))
    evidence_parts: list[str] = []
    for item in state.context:
        evidence_parts.append(item.content)
        evidence_parts.extend(item.related_entities)
        section = str(item.metadata.get("section", "")).strip()
        if section:
            evidence_parts.append(section)

    context_tokens = set(_tokens("\n".join(evidence_parts)))
    token_overlap = len(answer_tokens & context_tokens) / max(1, len(answer_tokens))
    has_citations = len(state.citations) > 0
    state.verified = has_citations and token_overlap >= 0.25
    return state


def _extract_focus_terms(question: str) -> list[str]:
    raw = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", question.lower())
    stop = {
        "what",
        "how",
        "why",
        "when",
        "where",
        "which",
        "does",
        "riskfolio",
        "portfolio",
        "about",
        "with",
        "into",
    }
    terms: list[str] = []
    seen: set[str] = set()
    for token in raw:
        if token in stop or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())


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


def is_langgraph_enabled() -> bool:
    return LANGGRAPH_AVAILABLE
