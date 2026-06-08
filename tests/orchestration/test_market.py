"""Tests for Phase E35 — Market-Based Task Allocation."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.core.errors import MarketAllocationError
from nexus.core.types import (
    AgentDefinition,
    AgentStatus,
    ExecutionResult,
    Message,
    Role,
    TokenUsage,
)
from nexus.orchestration.crew import CrewMember
from nexus.orchestration.market.allocator import MarketAllocator
from nexus.orchestration.market.board import TaskBoard
from nexus.orchestration.market.crew import MarketCrew
from nexus.orchestration.market.registry import CapabilityRegistry
from nexus.orchestration.market.reputation import ReputationTracker
from nexus.orchestration.market.scorer import BidScorer
from nexus.orchestration.market.types import (
    AllocationStatus,
    Bid,
    CapabilityProfile,
    ReputationRecord,
    TaskOutcome,
    TaskSpec,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _profile(
    agent_id: str = "agent-1",
    skills: list[str] | None = None,
    cost: float = 0.01,
) -> CapabilityProfile:
    return CapabilityProfile(
        agent_id=agent_id,
        agent_name=f"Agent {agent_id}",
        skill_tags=skills or ["web_search", "summarize"],
        cost_per_step_usd=cost,
    )


def _task(
    required: list[str] | None = None,
    preferred: list[str] | None = None,
    budget: float | None = 1.0,
    threshold: float = 0.5,
) -> TaskSpec:
    return TaskSpec(
        task_id=str(uuid.uuid4()),
        description="Test task",
        required_skills=required or ["web_search"],
        preferred_skills=preferred or [],
        budget_usd=budget,
        min_success_threshold=threshold,
    )


def _bid(
    agent_id: str = "agent-1",
    task_id: str = "task-1",
    success_prob: float = 0.8,
    cost: float = 0.5,
) -> Bid:
    return Bid(
        bid_id=str(uuid.uuid4()),
        task_id=task_id,
        agent_id=agent_id,
        self_reported_success_prob=success_prob,
        estimated_cost_usd=cost,
        estimated_steps=5,
    )


def _outcome(
    task_id: str = "task-1",
    agent_id: str = "agent-1",
    success: bool = True,
    cost: float = 0.5,
    steps: int = 5,
) -> TaskOutcome:
    return TaskOutcome(
        task_id=task_id,
        agent_id=agent_id,
        actual_success=success,
        actual_cost_usd=cost,
        actual_steps=steps,
    )


def _exec_result(output: str = "done") -> ExecutionResult:
    return ExecutionResult(
        output=output,
        messages=[Message(role=Role.ASSISTANT, content=output)],
        tool_calls_made=0,
        token_usage=TokenUsage(
            input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.01, model="test"
        ),
        duration_seconds=0.1,
        steps_taken=2,
        status=AgentStatus.COMPLETED,
    )


def _mock_runner(output: str = "done") -> MagicMock:
    runner = MagicMock()
    runner.run = AsyncMock(return_value=_exec_result(output))
    runner.cost_summary = MagicMock(return_value=None)
    return runner


def _member(agent_id: str, skills: list[str] | None = None) -> CrewMember:
    return CrewMember(
        agent_def=AgentDefinition(name=agent_id, model="test-model"),
        runner=_mock_runner(),
        role="worker",
    )


def _mock_model_client(json_response: str = "") -> MagicMock:
    from nexus.core.types import TokenUsage

    mock = MagicMock()
    mock.default_model = "test-model"

    async def _complete(**_kwargs: object) -> MagicMock:
        resp = MagicMock()
        resp.content = json_response
        resp.tool_calls = []
        resp.token_usage = TokenUsage(
            input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.001, model="test"
        )
        return resp

    mock.complete = AsyncMock(side_effect=_complete)
    return mock


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_stores_profile() -> None:
    reg = CapabilityRegistry()
    p = _profile("a1", skills=["web_search"])
    await reg.register(p)
    capable = reg.filter_capable(["web_search"], [])
    assert len(capable) == 1
    assert capable[0].agent_id == "a1"


@pytest.mark.asyncio
async def test_filter_capable_requires_all_skills() -> None:
    reg = CapabilityRegistry()
    await reg.register(_profile("a1", skills=["web_search"]))
    await reg.register(_profile("a2", skills=["web_search", "sql"]))
    capable = reg.filter_capable(["web_search", "sql"], [])
    assert len(capable) == 1
    assert capable[0].agent_id == "a2"


@pytest.mark.asyncio
async def test_filter_capable_ranks_by_preferred() -> None:
    reg = CapabilityRegistry()
    await reg.register(_profile("a1", skills=["web_search"]))
    await reg.register(_profile("a2", skills=["web_search", "summarize", "sql"]))
    capable = reg.filter_capable(["web_search"], ["summarize", "sql"])
    assert capable[0].agent_id == "a2"


@pytest.mark.asyncio
async def test_filter_capable_max_candidates() -> None:
    reg = CapabilityRegistry(max_candidates=5)
    for i in range(10):
        await reg.register(_profile(f"agent-{i}", skills=["web_search"]))
    capable = reg.filter_capable(["web_search"], [])
    assert len(capable) == 5


@pytest.mark.asyncio
async def test_deregister_removes_profile() -> None:
    reg = CapabilityRegistry()
    p = _profile("a1", skills=["web_search"])
    await reg.register(p)
    await reg.deregister("a1")
    assert reg.filter_capable(["web_search"], []) == []


# ---------------------------------------------------------------------------
# Reputation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_agent_returns_defaults() -> None:
    tracker = ReputationTracker()
    record = await tracker.get("new-agent")
    assert record.total_tasks == 0
    assert record.success_rate == 0.0
    assert record.calibration_factor == 1.0


@pytest.mark.asyncio
async def test_update_increments_success_rate() -> None:
    tracker = ReputationTracker()
    for i in range(4):
        await tracker.update(_outcome(f"t{i}", success=(i < 3)))
    record = await tracker.get("agent-1")
    assert abs(record.success_rate - 0.75) < 1e-9


@pytest.mark.asyncio
async def test_ucb_bonus_decreases_with_more_runs() -> None:
    tracker = ReputationTracker()
    for i in range(20):
        await tracker.update(_outcome(f"t{i}", agent_id="veteran"))
    bonus_veteran = await tracker.ucb_bonus("veteran")
    bonus_new = await tracker.ucb_bonus("new-agent")
    assert bonus_new > bonus_veteran


@pytest.mark.asyncio
async def test_calibration_factor_adjusts_down() -> None:
    tracker = ReputationTracker()
    tracker.record_self_report("agent-1", 0.9)
    tracker.record_self_report("agent-1", 0.9)
    for i in range(4):
        await tracker.update(_outcome(f"t{i}", success=(i < 2)))
    record = await tracker.get("agent-1")
    assert record.calibration_factor < 1.0


@pytest.mark.asyncio
async def test_cost_accuracy_ema() -> None:
    tracker = ReputationTracker()
    outcome = TaskOutcome(
        task_id="t1",
        agent_id="agent-1",
        actual_success=True,
        actual_cost_usd=2.0,
        actual_steps=5,
    )
    await tracker.update(outcome)
    record = await tracker.get("agent-1")
    assert record.cost_accuracy >= 1.0


@pytest.mark.asyncio
async def test_reputation_persists_in_dapr() -> None:
    mock_store = MagicMock()
    mock_store.save = AsyncMock()
    mock_store.get = AsyncMock(return_value=(None, ""))
    tracker = ReputationTracker(state_store=mock_store)
    await tracker.update(_outcome("t1", success=True))
    mock_store.save.assert_called_once()


# ---------------------------------------------------------------------------
# Scorer tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_applies_calibration_discount() -> None:
    tracker = ReputationTracker()
    record = ReputationRecord(
        agent_id="a1", total_tasks=10, successful_tasks=5, calibration_factor=0.5
    )
    tracker._records["a1"] = record
    scorer = BidScorer(tracker)
    b = _bid("a1", success_prob=0.9)
    score = await scorer.score(b, _task())
    assert abs(score.calibrated_success_prob - 0.45) < 1e-6


@pytest.mark.asyncio
async def test_score_below_threshold_gets_negative() -> None:
    tracker = ReputationTracker()
    record = ReputationRecord(
        agent_id="a1", total_tasks=5, successful_tasks=1, calibration_factor=0.3
    )
    tracker._records["a1"] = record
    scorer = BidScorer(tracker)
    b = _bid("a1", success_prob=0.6)
    task = _task(threshold=0.5)
    score = await scorer.score(b, task)
    assert score.final_score == -1.0


@pytest.mark.asyncio
async def test_score_all_sorted_descending() -> None:
    tracker = ReputationTracker()
    scorer = BidScorer(tracker)
    t = _task(budget=1.0)
    bids = [_bid("a1", success_prob=0.9, cost=0.1), _bid("a2", success_prob=0.4, cost=0.9)]
    scores = await scorer.score_all(bids, t)
    assert scores[0].final_score >= scores[1].final_score


@pytest.mark.asyncio
async def test_ucb_bonus_included_in_final() -> None:
    tracker = ReputationTracker()
    scorer = BidScorer(tracker)
    b = _bid("a1", success_prob=0.9)
    score = await scorer.score(b, _task())
    assert score.ucb_bonus > 0.0
    assert abs(score.final_score - (score.composite + score.ucb_bonus)) < 1e-9


@pytest.mark.asyncio
async def test_cost_score_normalized_to_budget() -> None:
    tracker = ReputationTracker()
    scorer = BidScorer(tracker)
    t = _task(budget=1.0)
    cheap = _bid("a1", success_prob=0.8, cost=0.1)
    expensive = _bid("a2", success_prob=0.8, cost=0.9)
    s_cheap = await scorer.score(cheap, t)
    s_expensive = await scorer.score(expensive, t)
    assert s_cheap.cost_score > s_expensive.cost_score


def test_alpha_beta_gamma_must_sum_to_one() -> None:
    tracker = ReputationTracker()
    with pytest.raises(ValueError, match="1.0"):
        BidScorer(tracker, alpha=0.5, beta=0.5, gamma=0.5)


# ---------------------------------------------------------------------------
# Board tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_task_returns_task_id() -> None:
    board = TaskBoard()
    t = _task()
    task_id = await board.post_task(t)
    assert task_id == t.task_id


@pytest.mark.asyncio
async def test_submit_bid_stored() -> None:
    board = TaskBoard()
    t = _task()
    await board.post_task(t)
    b = _bid(task_id=t.task_id)
    await board.submit_bid(b)
    bids = await board.get_bids_for_task(t.task_id)
    assert len(bids) == 1
    assert bids[0].bid_id == b.bid_id


@pytest.mark.asyncio
async def test_get_bids_returns_all_for_task() -> None:
    board = TaskBoard()
    t = _task()
    await board.post_task(t)
    for agent in ["a1", "a2", "a3"]:
        await board.submit_bid(_bid(agent_id=agent, task_id=t.task_id))
    bids = await board.get_bids_for_task(t.task_id)
    assert len(bids) == 3


@pytest.mark.asyncio
async def test_status_transitions() -> None:
    board = TaskBoard()
    t = _task()
    await board.post_task(t)
    assert board._task_statuses[t.task_id] == AllocationStatus.PENDING
    await board.update_task_status(t.task_id, AllocationStatus.ALLOCATED)
    assert board._task_statuses[t.task_id] == AllocationStatus.ALLOCATED
    await board.mark_outcome(_outcome(task_id=t.task_id, success=True))
    assert board._task_statuses[t.task_id] == AllocationStatus.COMPLETED


# ---------------------------------------------------------------------------
# Allocator tests
# ---------------------------------------------------------------------------


def _build_allocator(
    profiles: list[CapabilityProfile] | None = None,
    json_response: str = '{"self_reported_success_prob": 0.8, "estimated_cost_usd": 0.5, "estimated_steps": 5, "rationale": "test"}',
) -> tuple[MarketAllocator, CapabilityRegistry, TaskBoard, ReputationTracker]:
    reg = CapabilityRegistry()
    board = TaskBoard()
    tracker = ReputationTracker()
    scorer = BidScorer(tracker)
    model_client = _mock_model_client(json_response)
    allocator = MarketAllocator(reg, board, scorer, tracker, model_client)

    async def _register_all() -> None:
        for p in profiles or []:
            await reg.register(p)

    import asyncio

    asyncio.get_event_loop().run_until_complete(_register_all())
    return allocator, reg, board, tracker


@pytest.mark.asyncio
async def test_allocate_selects_highest_scoring_bid() -> None:
    reg = CapabilityRegistry()
    board = TaskBoard()
    tracker = ReputationTracker()
    scorer = BidScorer(tracker)

    call_count = 0

    async def _complete(messages: object, model: str, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        if call_count == 1:
            resp.content = '{"self_reported_success_prob": 0.9, "estimated_cost_usd": 0.1, "estimated_steps": 3, "rationale": "good"}'
        else:
            resp.content = '{"self_reported_success_prob": 0.4, "estimated_cost_usd": 0.8, "estimated_steps": 10, "rationale": "bad"}'
        resp.tool_calls = []
        resp.token_usage = TokenUsage(
            input_tokens=5, output_tokens=5, total_tokens=10, cost_usd=0.001, model="test"
        )
        return resp

    model_client = MagicMock()
    model_client.default_model = "test"
    model_client.complete = AsyncMock(side_effect=_complete)

    await reg.register(_profile("a1", skills=["web_search"]))
    await reg.register(_profile("a2", skills=["web_search"]))
    allocator = MarketAllocator(reg, board, scorer, tracker, model_client)

    result = await allocator.allocate(_task(threshold=0.3))
    assert result.status == AllocationStatus.ALLOCATED
    assert result.winning_agent_id == "a1"


@pytest.mark.asyncio
async def test_allocate_rejects_when_no_capable_agents() -> None:
    reg = CapabilityRegistry()
    board = TaskBoard()
    tracker = ReputationTracker()
    scorer = BidScorer(tracker)
    allocator = MarketAllocator(reg, board, scorer, tracker, _mock_model_client())

    result = await allocator.allocate(_task(required=["rare_skill"]))
    assert result.status == AllocationStatus.REJECTED
    assert "No capable" in (result.reject_reason or "")


@pytest.mark.asyncio
async def test_allocate_rejects_when_all_bids_below_threshold() -> None:
    reg = CapabilityRegistry()
    board = TaskBoard()
    tracker = ReputationTracker()
    scorer = BidScorer(tracker)
    await reg.register(_profile("a1", skills=["web_search"]))

    record = ReputationRecord(agent_id="a1", total_tasks=10, calibration_factor=0.1)
    tracker._records["a1"] = record

    model_client = _mock_model_client(
        '{"self_reported_success_prob": 0.4, "estimated_cost_usd": 0.5, "estimated_steps": 5, "rationale": "low"}'
    )
    allocator = MarketAllocator(reg, board, scorer, tracker, model_client)
    result = await allocator.allocate(_task(threshold=0.9))
    assert result.status == AllocationStatus.REJECTED


@pytest.mark.asyncio
async def test_allocate_filters_incapable_before_soliciting() -> None:
    reg = CapabilityRegistry()
    board = TaskBoard()
    tracker = ReputationTracker()
    scorer = BidScorer(tracker)

    await reg.register(_profile("capable", skills=["web_search", "sql"]))
    await reg.register(_profile("incapable", skills=["other"]))

    model_client = _mock_model_client(
        '{"self_reported_success_prob": 0.8, "estimated_cost_usd": 0.5, "estimated_steps": 5, "rationale": "ok"}'
    )
    allocator = MarketAllocator(reg, board, scorer, tracker, model_client)
    await allocator.allocate(_task(required=["web_search", "sql"]))

    assert model_client.complete.call_count == 1


@pytest.mark.asyncio
async def test_report_outcome_updates_reputation() -> None:
    reg = CapabilityRegistry()
    board = TaskBoard()
    tracker = ReputationTracker()
    scorer = BidScorer(tracker)
    allocator = MarketAllocator(reg, board, scorer, tracker, _mock_model_client())

    await allocator.report_outcome(_outcome("t1", "a1", success=True))
    record = await tracker.get("a1")
    assert record.total_tasks == 1
    assert record.success_rate == 1.0


@pytest.mark.asyncio
async def test_solicit_bid_fallback_on_parse_failure() -> None:
    reg = CapabilityRegistry()
    board = TaskBoard()
    tracker = ReputationTracker()
    scorer = BidScorer(tracker)

    await reg.register(_profile("a1", skills=["web_search"], cost=0.02))

    bad_client = _mock_model_client("NOT VALID JSON !!!")
    allocator = MarketAllocator(reg, board, scorer, tracker, bad_client)

    result = await allocator.allocate(_task(threshold=0.0))
    assert result.status in (AllocationStatus.ALLOCATED, AllocationStatus.REJECTED)

    bids = await board.get_bids_for_task(result.task_id)
    if bids:
        assert bids[0].self_reported_success_prob == 0.3


@pytest.mark.asyncio
async def test_otel_spans_emitted() -> None:
    reg = CapabilityRegistry()
    board = TaskBoard()
    tracker = ReputationTracker()
    scorer = BidScorer(tracker)
    await reg.register(_profile("a1", skills=["web_search"]))

    model_client = _mock_model_client(
        '{"self_reported_success_prob": 0.8, "estimated_cost_usd": 0.5, "estimated_steps": 5, "rationale": "ok"}'
    )

    mock_tracer = MagicMock()
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)
    span.set_attribute = MagicMock()
    mock_tracer._tracer = MagicMock()
    mock_tracer._tracer.start_as_current_span = MagicMock(return_value=span)

    allocator = MarketAllocator(reg, board, scorer, tracker, model_client, tracer=mock_tracer)
    await allocator.allocate(_task())

    assert mock_tracer._tracer.start_as_current_span.call_count >= 1
    call_names = [c.args[0] for c in mock_tracer._tracer.start_as_current_span.call_args_list]
    assert "market.allocate" in call_names


# ---------------------------------------------------------------------------
# MarketCrew tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_market_crew_use_market_false_bypasses_allocation() -> None:
    m = _member("agent-1")
    crew = MarketCrew(
        [m],
        session_id="sess-1",
        use_market=False,
    )
    await crew.run_task_with_market("do something", ["web_search"])
    m.runner.run.assert_called_once()


@pytest.mark.asyncio
async def test_market_crew_raises_on_rejected() -> None:
    m = _member("agent-1")
    reg = CapabilityRegistry()
    board = TaskBoard()
    tracker = ReputationTracker()
    scorer = BidScorer(tracker)
    allocator = MarketAllocator(reg, board, scorer, tracker, _mock_model_client())

    crew = MarketCrew(
        [m],
        session_id="sess-1",
        allocator=allocator,
        use_market=True,
    )
    with pytest.raises(MarketAllocationError):
        await crew.run_task_with_market("do something", ["missing_skill"])


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_allocation_pipeline() -> None:
    reg = CapabilityRegistry()
    board = TaskBoard()
    tracker = ReputationTracker()
    scorer = BidScorer(tracker)

    await reg.register(_profile("a1", skills=["web_search", "summarize"], cost=0.01))
    await reg.register(_profile("a2", skills=["web_search"], cost=0.02))
    await reg.register(_profile("a3", skills=["sql", "analysis"], cost=0.015))

    model_client = _mock_model_client(
        '{"self_reported_success_prob": 0.85, "estimated_cost_usd": 0.1, "estimated_steps": 5, "rationale": "good fit"}'
    )
    allocator = MarketAllocator(reg, board, scorer, tracker, model_client)

    spec = _task(required=["web_search"], preferred=["summarize"])
    result = await allocator.allocate(spec)

    assert result.status == AllocationStatus.ALLOCATED
    assert result.winning_agent_id is not None

    assert "a3" in result.capability_filtered_out

    outcome = TaskOutcome(
        task_id=result.task_id,
        agent_id=result.winning_agent_id,
        actual_success=True,
        actual_cost_usd=0.09,
        actual_steps=4,
    )
    await allocator.report_outcome(outcome)

    record = await tracker.get(result.winning_agent_id)
    assert record.total_tasks == 1
    assert record.success_rate == 1.0
