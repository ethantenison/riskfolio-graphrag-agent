"""Tests for riskfolio_graphrag_agent.agent.workflow."""

from __future__ import annotations

from riskfolio_graphrag_agent.retrieval.retriever import RetrievalResult

from riskfolio_graphrag_agent.agent.workflow import (
    AgentState,
    AgentWorkflow,
    is_langgraph_enabled,
    _plan,
    _reason,
    _retrieve,
    _verify,
)


class _FakeRetriever:
    def retrieve(self, query: str) -> list[RetrievalResult]:
        _ = query
        return [
            RetrievalResult(
                content="HRP uses clustering and risk parity principles.",
                source_path="/tmp/examples.rst",
                score=0.9,
                related_entities=["Hierarchical Risk Parity", "Risk Parity"],
                graph_neighbours=["examples.rst::chunk:1"],
                metadata={
                    "chunk_id": "examples.rst::chunk:0",
                    "relative_path": "examples.rst",
                    "chunk_index": 0,
                    "section": "Hierarchical Clustering Portfolio Optimization",
                    "line_start": 97,
                    "line_end": 102,
                },
            )
        ]


def test_agent_state_defaults():
    state = AgentState(question="What is CVaR?")
    assert state.question == "What is CVaR?"
    assert state.sub_questions == []
    assert state.verified is False


def test_plan_populates_sub_questions():
    state = AgentState(question="Explain Sharpe ratio.")
    state = _plan(state)
    assert len(state.sub_questions) >= 1


def test_retrieve_stub():
    state = AgentState(question="test", sub_questions=["test"])
    state = _retrieve(state, retriever=_FakeRetriever())
    assert isinstance(state.context, list)
    assert state.context


def test_reason_stub():
    state = AgentState(question="test", sub_questions=["test"])
    state = _retrieve(state, retriever=_FakeRetriever())
    state = _reason(state)
    assert isinstance(state.answer, str)
    assert state.citations


def test_reason_uses_llm_generate_when_available():
    state = AgentState(question="What is HRP?", sub_questions=["What is HRP?"])
    state = _retrieve(state, retriever=_FakeRetriever())

    def _fake_llm_generate(*, question: str, context: list[RetrievalResult], model_name: str) -> str:
        assert question == "What is HRP?"
        assert context
        assert model_name == "gpt-4o-mini"
        return "HRP uses hierarchical clustering and risk-budgeting constraints."

    state = _reason(state, llm_generate=_fake_llm_generate, model_name="gpt-4o-mini")
    assert state.answer.startswith("HRP uses hierarchical clustering")
    assert state.citations


def test_verify_stub():
    state = AgentState(question="What is HRP?")
    state = _plan(state)
    state = _retrieve(state, retriever=_FakeRetriever())
    state = _reason(state)
    state = _verify(state)
    assert state.verified is True


def test_workflow_run_end_to_end():
    """AgentWorkflow.run should complete without error and return an AgentState."""
    workflow = AgentWorkflow(retriever=_FakeRetriever())
    result = workflow.run("What is the minimum variance portfolio?")
    assert isinstance(result, AgentState)
    assert result.question == "What is the minimum variance portfolio?"
    assert result.citations


def test_workflow_compiles_langgraph_when_available():
    workflow = AgentWorkflow(retriever=_FakeRetriever())
    if is_langgraph_enabled():
        assert workflow._graph is not None
