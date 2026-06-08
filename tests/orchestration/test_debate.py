"""Tests for the multi-agent debate system (Phase D14).

All tests use asyncio_mode = "auto" (no @pytest.mark.asyncio needed).
No real LLM calls — FakeModelClient returns deterministic JSON responses.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import pytest

from nexus.core.errors import BudgetExceededError
from nexus.core.models.base import ModelResponse
from nexus.core.types import AgentState, AgentStatus, Message, Role, TokenUsage
from nexus.orchestration.debate.aggregator import (
    JudgeAggregator,
    MajorityVoteAggregator,
    WeightedVoteAggregator,
)
from nexus.orchestration.debate.convergence import ConvergenceDetector
from nexus.orchestration.debate.debater import Debater
from nexus.orchestration.debate.orchestrator import DebateOrchestrator
from nexus.orchestration.debate.router import DebateRouter
from nexus.orchestration.debate.types import (
    AggregationStrategy,
    DebateConfig,
    DebaterConfig,
    DebateResult,
    DebateRound,
    DebaterPosition,
    RoutingDecision,
)
from nexus.orchestration.graph import Graph
from nexus.orchestration.nodes import debate_node, human_node

# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


def _usage(model: str = "fake-model") -> TokenUsage:
    return TokenUsage(
        input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.001, model=model
    )


def _fake_response(content: str, model: str = "fake-model") -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=[],
        token_usage=_usage(model),
        model=model,
        stop_reason="end_turn",
    )


class FakeModelClient:
    """Returns pre-canned responses in order; records all calls for assertion."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, *, messages: list[Any], model: str, **kwargs: Any) -> ModelResponse:
        assert self._responses, f"FakeModelClient ran out of responses (model={model})"
        content = self._responses.pop(0)
        self.calls.append({"messages": messages, "model": model, **kwargs})
        return _fake_response(content, model=model)


class FakeTracer:
    """Records span names emitted via span(name, **attrs)."""

    def __init__(self) -> None:
        self.recorded: list[str] = []

    @contextlib.contextmanager
    def span(self, name: str, **_attrs: Any) -> Generator[None, None, None]:
        self.recorded.append(name)
        yield None


def _round1_json(answer: str, confidence: float = 0.7) -> str:
    return json.dumps(
        {"answer": answer, "reasoning": f"Because {answer}", "confidence": confidence}
    )


def _round2_json(answer: str, confidence: float = 0.7, changed: bool = False) -> str:
    return json.dumps(
        {
            "answer": answer,
            "reasoning": f"Because {answer}",
            "confidence": confidence,
            "changed": changed,
            "change_justification": "New evidence found" if changed else "",
        }
    )


def _debater_cfg(
    client: FakeModelClient, model_id: str = "m1", weight: float = 1.0
) -> DebaterConfig:
    return DebaterConfig(model_client=client, model_id=model_id, temperature=0.7, weight=weight)


def _position(answer: str, index: int = 0, confidence: float = 0.7) -> DebaterPosition:
    return DebaterPosition(
        debater_index=index,
        model_id="m",
        answer=answer,
        reasoning="reason",
        confidence=confidence,
    )


def _debate_result(escalate: bool = False, conv: float = 0.9) -> DebateResult:
    return DebateResult(
        question="q",
        final_answer="42",
        final_reasoning="reason",
        confidence=0.8,
        escalate_to_human=escalate,
        rounds=[],
        aggregation_method=AggregationStrategy.MAJORITY_VOTE,
        routing_decision=RoutingDecision.DEBATE,
        total_rounds_run=2,
        converged=conv >= 0.8,
        final_convergence_score=conv,
        total_token_usage=_usage(),
        total_cost_usd=0.002,
        duration_seconds=0.1,
    )


def _state(user_content: str = "What is 2+2?") -> AgentState:
    return AgentState(
        agent_id="agent-1",
        session_id="sess-1",
        messages=[Message(role=Role.USER, content=user_content)],
    )


# ---------------------------------------------------------------------------
# Convergence detector tests
# ---------------------------------------------------------------------------


def test_convergence_detector_unanimous() -> None:
    detector = ConvergenceDetector(threshold=0.8)
    positions = [_position("Paris", i) for i in range(3)]
    assert detector.score(positions) == 1.0
    assert detector.should_stop(positions) is True


def test_convergence_detector_split() -> None:
    detector = ConvergenceDetector(threshold=0.8)
    positions = [
        _position("Paris", 0),
        _position("Berlin", 1),
        _position("Madrid", 2),
    ]
    score = detector.score(positions)
    assert score < 0.5
    assert detector.should_stop(positions) is False


def test_convergence_detector_majority() -> None:
    detector = ConvergenceDetector(threshold=0.8, jaccard_min=0.3)
    positions = [
        _position("answer Paris France", 0),
        _position("answer Paris France correct", 1),
        _position("completely different unrelated xyz", 2),
    ]
    score = detector.score(positions)
    # Two Paris answers cluster together → 2/3 ≈ 0.67
    assert 0.6 <= score <= 0.7
    assert detector.should_stop(positions) is False


# ---------------------------------------------------------------------------
# Aggregator tests
# ---------------------------------------------------------------------------


async def test_majority_vote_aggregator() -> None:
    positions = [
        _position("Paris", 0, confidence=0.8),
        _position("Paris", 1, confidence=0.7),
        _position("Berlin", 2, confidence=0.95),
    ]
    rnd = DebateRound(round_number=1, positions=positions, convergence_score=0.67)
    cfg = [_debater_cfg(FakeModelClient([]))] * 3

    agg = MajorityVoteAggregator()
    answer, reasoning, confidence = await agg.aggregate("q", [rnd], cfg)
    assert answer == "Paris"
    assert 0.7 <= confidence <= 0.8


async def test_weighted_vote_aggregator() -> None:
    # Two low-confidence Paris + one high-confidence Berlin.
    # With high weights on "Berlin", weighted aggregator can differ from majority.
    clients = [FakeModelClient([]) for _ in range(3)]
    cfg = [
        _debater_cfg(clients[0], weight=0.5),
        _debater_cfg(clients[1], weight=0.5),
        _debater_cfg(clients[2], weight=5.0),
    ]
    positions = [
        _position("Paris", 0, confidence=0.5),
        _position("Paris", 1, confidence=0.5),
        _position("Berlin", 2, confidence=0.9),
    ]
    rnd = DebateRound(round_number=1, positions=positions, convergence_score=0.67)

    agg = WeightedVoteAggregator()
    answer, _, _ = await agg.aggregate("q", [rnd], cfg)
    # "Berlin" has weight 5.0 * 0.9 = 4.5 vs Paris cluster 0.5*0.5 + 0.5*0.5 = 0.5
    assert answer == "Berlin"


async def test_judge_aggregator() -> None:
    judge_response = json.dumps(
        {"answer": "42", "reasoning": "The answer to everything", "confidence": 0.95}
    )
    judge_client = FakeModelClient([judge_response])
    judge_cfg = DebaterConfig(model_client=judge_client, model_id="judge-model")

    positions = [_position("42", 0, 0.8), _position("42", 1, 0.7)]
    rnd = DebateRound(round_number=1, positions=positions, convergence_score=1.0)

    agg = JudgeAggregator(judge_cfg)
    answer, reasoning, confidence = await agg.aggregate("What is 6x7?", [rnd], [])

    assert answer == "42"
    assert confidence == pytest.approx(0.95)
    assert len(judge_client.calls) == 1
    user_msg_content = judge_client.calls[0]["messages"][-1].content
    assert "42" in user_msg_content


async def test_judge_aggregator_fallback() -> None:
    """Non-JSON judge response falls back to majority vote without raising."""
    judge_client = FakeModelClient(["not json at all — just prose"])
    judge_cfg = DebaterConfig(model_client=judge_client, model_id="judge-model")

    positions = [_position("Paris", 0, 0.8), _position("Paris", 1, 0.7)]
    rnd = DebateRound(round_number=1, positions=positions, convergence_score=1.0)

    agg = JudgeAggregator(judge_cfg)
    answer, _, _ = await agg.aggregate("Capital?", [rnd], [])
    assert answer == "Paris"


# ---------------------------------------------------------------------------
# Debater tests
# ---------------------------------------------------------------------------


async def test_debater_round1_independent() -> None:
    client = FakeModelClient([_round1_json("Paris", 0.85)])
    debater = Debater(0, _debater_cfg(client, "m1"))
    pos = await debater.respond("What is the capital of France?", 1, None, None, None)

    assert pos.answer == "Paris"
    assert pos.confidence == pytest.approx(0.85)
    assert pos.changed_from_previous is False
    assert pos.debater_index == 0
    assert len(client.calls) == 1


async def test_debater_round2_sycophancy_prompt() -> None:
    client = FakeModelClient([_round2_json("Paris", 0.9)])
    debater = Debater(0, _debater_cfg(client, "m1"))

    peer_pos = _position("Berlin", index=1, confidence=0.6)
    prev_round = DebateRound(
        round_number=1, positions=[_position("Paris", 0), peer_pos], convergence_score=0.5
    )
    await debater.respond("Capital of France?", 2, prev_round, None, None)

    system_content = client.calls[0]["messages"][0].content
    assert "Restate your previous answer" in system_content
    user_msg_content = client.calls[0]["messages"][-1].content
    assert "Berlin" in user_msg_content


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------


async def test_debate_router_bypasses_on_high_confidence() -> None:
    client = FakeModelClient([_round1_json("42", 0.9)])
    cfg = DebateConfig(
        debaters=[_debater_cfg(client), _debater_cfg(FakeModelClient([]))],
        routing_confidence_threshold=0.85,
        adaptive_routing=True,
    )
    router = DebateRouter(cfg)
    result = await router.route("What is 6x7?", None, None)

    assert result is not None
    assert result.routing_decision == RoutingDecision.SINGLE_AGENT
    assert result.total_rounds_run == 1
    assert result.escalate_to_human is False


async def test_debate_router_proceeds_on_low_confidence() -> None:
    client = FakeModelClient([_round1_json("maybe 42", 0.4)])
    cfg = DebateConfig(
        debaters=[_debater_cfg(client), _debater_cfg(FakeModelClient([]))],
        routing_confidence_threshold=0.85,
        adaptive_routing=True,
    )
    router = DebateRouter(cfg)
    result = await router.route("Hard question?", None, None)
    assert result is None


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


def _three_debater_config(
    answers_per_debater: list[list[str]],
    convergence_threshold: float = 0.8,
    aggregation: AggregationStrategy = AggregationStrategy.MAJORITY_VOTE,
    adaptive_routing: bool = False,
    escalate_threshold: float = 0.5,
) -> tuple[DebateConfig, list[FakeModelClient]]:
    clients = [FakeModelClient(responses) for responses in answers_per_debater]
    cfg = DebateConfig(
        debaters=[_debater_cfg(c, f"m{i}") for i, c in enumerate(clients)],
        max_rounds=3,
        convergence_threshold=convergence_threshold,
        aggregation=aggregation,
        adaptive_routing=adaptive_routing,
        escalate_threshold=escalate_threshold,
    )
    return cfg, clients


async def test_orchestrator_full_debate_3_rounds() -> None:
    """Three debaters with divergent answers — exactly 3 rounds run (no early stop)."""
    cfg, clients = _three_debater_config(
        [
            [_round1_json("Alpha"), _round2_json("Alpha"), _round2_json("Alpha")],
            [_round1_json("Beta"), _round2_json("Beta"), _round2_json("Beta")],
            [_round1_json("Gamma"), _round2_json("Gamma"), _round2_json("Gamma")],
        ],
        convergence_threshold=0.99,  # force no early stop (1/3 < 0.99)
    )
    orch = DebateOrchestrator(cfg)
    result = await orch.run("Question?")

    assert result.total_rounds_run == 3
    assert result.routing_decision == RoutingDecision.DEBATE
    for client in clients:
        assert len(client.calls) == 3


async def test_orchestrator_early_stop() -> None:
    """All debaters agree in round 1 — stops after round 1."""
    answer_json = _round1_json("Paris", 0.9)
    cfg, _ = _three_debater_config(
        [[answer_json], [answer_json], [answer_json]],
        convergence_threshold=0.7,  # 3/3 > 0.7
    )
    orch = DebateOrchestrator(cfg)
    result = await orch.run("Capital of France?")

    assert result.total_rounds_run == 1
    assert result.converged is True


async def test_orchestrator_escalate_on_low_convergence() -> None:
    """Debaters never agree — final convergence is low → escalate_to_human=True."""
    cfg, _ = _three_debater_config(
        [
            [_round1_json("Alpha"), _round2_json("Alpha"), _round2_json("Alpha")],
            [_round1_json("Beta"), _round2_json("Beta"), _round2_json("Beta")],
            [_round1_json("Gamma"), _round2_json("Gamma"), _round2_json("Gamma")],
        ],
        convergence_threshold=0.99,
        escalate_threshold=0.99,  # always escalate when not unanimous
    )
    orch = DebateOrchestrator(cfg)
    result = await orch.run("Hard question?")

    assert result.escalate_to_human is True


async def test_orchestrator_budget_enforced() -> None:
    """BudgetExceededError raised before round 2 when budget is exhausted."""
    mock_tracker = MagicMock()
    mock_tracker.check_budget.side_effect = [
        None,  # before round 1: OK
        BudgetExceededError("Exceeded", code="BUDGET_EXCEEDED"),
    ]

    cfg, _ = _three_debater_config(
        [
            [_round1_json("Alpha"), _round2_json("Alpha")],
            [_round1_json("Beta"), _round2_json("Beta")],
            [_round1_json("Gamma"), _round2_json("Gamma")],
        ],
        convergence_threshold=0.99,  # no early stop in round 1
        adaptive_routing=False,
    )
    orch = DebateOrchestrator(cfg, cost_tracker=mock_tracker)

    with pytest.raises(BudgetExceededError):
        await orch.run("Question?")

    assert mock_tracker.check_budget.call_count == 2


# ---------------------------------------------------------------------------
# debate_node tests
# ---------------------------------------------------------------------------


async def test_debate_node_injects_result() -> None:
    """debate_node appends ASSISTANT message with final_answer and metadata."""

    async def _run(q: str) -> DebateResult:
        return _debate_result()

    mock_orch = MagicMock()
    mock_orch.run = _run

    handler = debate_node(mock_orch)
    result_state = await handler(_state("What is 2+2?"))

    last_msg = result_state.messages[-1]
    assert last_msg.role == Role.ASSISTANT
    assert last_msg.content == "42"
    assert "debate_result" in last_msg.metadata


async def test_debate_node_escalate_metadata() -> None:
    """When escalate_to_human=True and on_escalate is set, metadata flag is True."""

    async def _run(q: str) -> DebateResult:
        return _debate_result(escalate=True, conv=0.3)

    mock_orch = MagicMock()
    mock_orch.run = _run

    handler = debate_node(mock_orch, on_escalate="human_review")
    result_state = await handler(_state())

    assert result_state.metadata.get("debate_escalate") is True


async def test_debate_node_escalate_metadata_not_set_when_no_on_escalate() -> None:
    """Without on_escalate, the metadata flag is NOT written to state.metadata."""

    async def _run(q: str) -> DebateResult:
        return _debate_result(escalate=True, conv=0.3)

    mock_orch = MagicMock()
    mock_orch.run = _run

    handler = debate_node(mock_orch)  # no on_escalate
    result_state = await handler(_state())

    assert "debate_escalate" not in result_state.metadata


async def test_debate_node_default_question_extraction() -> None:
    """Default extractor passes the last USER message content to orchestrator."""
    received: list[str] = []

    async def _run(q: str) -> DebateResult:
        received.append(q)
        return _debate_result()

    mock_orch = MagicMock()
    mock_orch.run = _run

    handler = debate_node(mock_orch)
    await handler(_state("Specific question text"))

    assert received == ["Specific question text"]


async def test_full_pipeline_with_graph() -> None:
    """Graph with debate_node → conditional → human_node routes to human on escalate."""

    async def _run(q: str) -> DebateResult:
        return _debate_result(escalate=True, conv=0.3)

    mock_orch = MagicMock()
    mock_orch.run = _run

    handler = debate_node(mock_orch, on_escalate="human_review")

    async def _route(state: AgentState) -> str:
        return "escalate" if state.metadata.get("debate_escalate") else "end"

    graph = (
        Graph(graph_id="test-debate")
        .add_node("debate", handler, entry=True)
        .add_conditional_edge("debate", _route, {"escalate": "human", "end": None})
        .add_node("human", human_node("Please review this decision."))
    )

    final_state = await graph.execute(_state("Hard question?"))
    assert final_state.status == AgentStatus.WAITING_FOR_HUMAN


# ---------------------------------------------------------------------------
# OTEL span tests
# ---------------------------------------------------------------------------


async def test_debate_otel_spans() -> None:
    """Verify debate.run, debate.round, and debate.debater spans are emitted."""
    tracer = FakeTracer()
    answer_json = _round1_json("Paris", 0.95)
    cfg, _ = _three_debater_config(
        [[answer_json], [answer_json], [answer_json]],
        convergence_threshold=0.7,
        adaptive_routing=False,
    )
    orch = DebateOrchestrator(cfg, tracer=tracer)
    await orch.run("Capital of France?")

    assert "debate.run" in tracer.recorded
    assert "debate.round" in tracer.recorded
    assert "debate.debater" in tracer.recorded
