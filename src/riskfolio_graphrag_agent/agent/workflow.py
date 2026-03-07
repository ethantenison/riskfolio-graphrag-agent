"""LangGraph-based agentic workflow (placeholder).

Planned workflow nodes
----------------------
1. **plan**      – decompose user question into sub-questions.
2. **retrieve**  – call :class:`~riskfolio_graphrag_agent.retrieval.retriever.HybridRetriever`.
3. **reason**    – generate a cited answer with chain-of-thought.
4. **verify**    – self-check answer against retrieved sources.
5. **respond**   – format final response with provenance citations.

This module currently contains **stub** implementations.  Wire up
LangGraph edges and nodes once the retrieval layer is complete.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

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
    context: list[str] = field(default_factory=list)
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    verified: bool = False


class AgentWorkflow:
    """Orchestrates the multi-step reasoning workflow.

    Args:
        retriever: An initialised :class:`~riskfolio_graphrag_agent.retrieval.retriever.HybridRetriever`.
        model_name: LLM model identifier to use for generation.
    """

    def __init__(self, retriever: object, model_name: str = "gpt-4o-mini") -> None:
        self._retriever = retriever
        self._model_name = model_name
        # TODO: build LangGraph StateGraph and compile it
        self._graph = None

    def run(self, question: str) -> AgentState:
        """Execute the full agent workflow for *question*.

        Args:
            question: Natural-language question from the user.

        Returns:
            A final :class:`AgentState` with answer and citations populated.
        """
        state = AgentState(question=question)
        logger.info("AgentWorkflow.run called (stub). question=%r", question)

        # TODO: invoke self._graph with state once LangGraph is wired up
        state = _plan(state)
        state = _retrieve(state, self._retriever)
        state = _reason(state)
        state = _verify(state)
        return state


# ── Stub workflow nodes ────────────────────────────────────────────────────────


def _plan(state: AgentState) -> AgentState:
    """Decompose the question into retrievable sub-questions.

    Args:
        state: Current agent state.

    Returns:
        Updated state with ``sub_questions`` populated.
    """
    # TODO: call LLM to decompose question
    state.sub_questions = [state.question]
    return state


def _retrieve(state: AgentState, retriever: object) -> AgentState:
    """Retrieve relevant context for each sub-question.

    Args:
        state: Current agent state.
        retriever: Initialised :class:`~riskfolio_graphrag_agent.retrieval.retriever.HybridRetriever`.

    Returns:
        Updated state with ``context`` populated.
    """
    # TODO: call retriever.retrieve() for each sub_question
    state.context = []
    return state


def _reason(state: AgentState) -> AgentState:
    """Generate a draft answer with chain-of-thought over retrieved context.

    Args:
        state: Current agent state with context populated.

    Returns:
        Updated state with ``answer`` and ``citations`` populated.
    """
    # TODO: call LLM with context and question
    state.answer = "(answer not yet implemented)"
    state.citations = []
    return state


def _verify(state: AgentState) -> AgentState:
    """Self-check whether the answer is grounded in the retrieved context.

    Args:
        state: Current agent state with answer populated.

    Returns:
        Updated state with ``verified`` set.
    """
    # TODO: implement faithfulness check
    state.verified = False
    return state
