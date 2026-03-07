"""Tests for riskfolio_graphrag_agent.agent.workflow."""

from __future__ import annotations

from riskfolio_graphrag_agent.agent.workflow import (
    AgentState,
    AgentWorkflow,
    _plan,
    _reason,
    _retrieve,
    _verify,
)


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
    state = _retrieve(state, retriever=None)
    assert isinstance(state.context, list)


def test_reason_stub():
    state = AgentState(question="test", sub_questions=["test"])
    state = _reason(state)
    assert isinstance(state.answer, str)


def test_verify_stub():
    state = AgentState(question="test")
    state = _verify(state)
    assert state.verified is False


def test_workflow_run_end_to_end():
    """AgentWorkflow.run should complete without error and return an AgentState."""
    workflow = AgentWorkflow(retriever=None)
    result = workflow.run("What is the minimum variance portfolio?")
    assert isinstance(result, AgentState)
    assert result.question == "What is the minimum variance portfolio?"
