"""Tests for E33 Uncertainty Quantification.

All tests use asyncio_mode = "auto" (set in pyproject.toml).
No real LLM calls — FakeModelClient returns deterministic strings.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from grampus.core.errors import UncertaintyError
from grampus.core.models.base import ModelResponse
from grampus.core.types import (
    AgentDefinition,
    AgentState,
    AgentStatus,
    Message,
    Role,
    TokenUsage,
    ToolCall,
    ToolResult,
)
from grampus.orchestration.runner import AgentRunner
from grampus.orchestration.uncertainty import (
    StepUncertainty,
    UncertaintyAction,
    UncertaintyEstimator,
    UncertaintyLevel,
    UncertaintyMonitor,
    UncertaintyPolicy,
    UncertaintyPropagator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_client(content: str = "response") -> Any:
    """Return an async model client that always responds with the given content."""
    client = MagicMock()
    usage = TokenUsage(input_tokens=5, output_tokens=5, total_tokens=10, cost_usd=0.0, model="test")
    client.complete = AsyncMock(
        return_value=ModelResponse(
            content=content,
            tool_calls=[],
            token_usage=usage,
            model="test",
            stop_reason="end_turn",
        )
    )
    return client


def _agent_def(name: str = "agent") -> AgentDefinition:
    return AgentDefinition(name=name, model="test-model")


def _tool_result(call_id: str = "tc-1") -> ToolResult:
    return ToolResult(tool_call_id=call_id, output="ok", duration_ms=1)


# ---------------------------------------------------------------------------
# 1. Verbalized extraction
# ---------------------------------------------------------------------------


async def test_verbalized_json_confidence() -> None:
    est = UncertaintyEstimator(enable_p_true=False)
    raw = est.extract_verbalized('{"answer": "Paris", "confidence": 0.90}')
    assert raw == pytest.approx(0.90)


async def test_verbalized_json_nested_search() -> None:
    est = UncertaintyEstimator(enable_p_true=False)
    text = 'The capital is Paris. {"confidence": 0.75} Some more text.'
    raw = est.extract_verbalized(text)
    assert raw == pytest.approx(0.75)


async def test_verbalized_heuristic_markers() -> None:
    est = UncertaintyEstimator(enable_p_true=False)
    text = "I think this might be correct, I'm not sure, it could be wrong."
    raw = est.extract_verbalized(text)
    assert raw < 0.7


async def test_verbalized_fallback_default() -> None:
    est = UncertaintyEstimator(enable_p_true=False)
    raw = est.extract_verbalized("The answer is 42.")
    assert raw == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 2. Calibration
# ---------------------------------------------------------------------------


async def test_calibration_verbalized() -> None:
    est = UncertaintyEstimator(verbalized_calibration_bias=0.25, enable_p_true=False)
    result = est._calibrate(0.90, 0.25)
    assert result == pytest.approx(0.90 * 0.75)


async def test_calibration_p_true() -> None:
    est = UncertaintyEstimator(p_true_calibration_bias=0.10, enable_p_true=False)
    result = est._calibrate(0.85, 0.10)
    assert result == pytest.approx(0.85 * 0.90)


async def test_calibration_cap() -> None:
    est = UncertaintyEstimator(enable_p_true=False)
    assert est._calibrate(1.0, 0.25) == pytest.approx(0.75)
    assert est._calibrate(0.0, 0.25) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. P(True)
# ---------------------------------------------------------------------------


async def test_p_true_parsed() -> None:
    client = _fake_client('{"p_true": 0.82, "reason": "correct"}')
    est = UncertaintyEstimator(enable_p_true=True)
    val = await est.p_true([], "some answer", client, "test-model")
    assert val == pytest.approx(0.82)


async def test_p_true_fallback_on_bad_json() -> None:
    client = _fake_client("not json at all ###")
    est = UncertaintyEstimator(enable_p_true=True)
    val = await est.p_true([], "some answer", client, "test-model")
    assert val == pytest.approx(0.5)


async def test_p_true_skipped_when_disabled() -> None:
    client = _fake_client('{"answer": "x", "confidence": 0.80}')
    est = UncertaintyEstimator(enable_p_true=False)
    policy = UncertaintyPolicy(enable_p_true=False)
    monitor = UncertaintyMonitor(estimator=est, policy=policy)
    monitor.initialize("s1", "a1")
    await monitor.observe_llm_response(
        response_text='{"answer": "x", "confidence": 0.80}',
        step_id="s0",
        model_client=client,
        model_id="test-model",
    )
    assert client.complete.call_count == 0


# ---------------------------------------------------------------------------
# 4. Fusion
# ---------------------------------------------------------------------------


async def test_fusion_with_p_true() -> None:
    est = UncertaintyEstimator(
        verbalized_weight=0.4,
        p_true_weight=0.6,
        verbalized_calibration_bias=0.25,
        p_true_calibration_bias=0.10,
    )
    fused = est.fuse(0.9, 0.6, p_true_ran=True)
    expected = 0.4 * (0.9 * 0.75) + 0.6 * (0.6 * 0.90)
    assert fused == pytest.approx(expected, abs=1e-6)


async def test_fusion_without_p_true() -> None:
    est = UncertaintyEstimator(verbalized_calibration_bias=0.25)
    fused = est.fuse(0.9, -1.0, p_true_ran=False)
    expected = 0.9 * 0.75
    assert fused == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 5. Semantic entropy
# ---------------------------------------------------------------------------


async def test_semantic_entropy_early_stop() -> None:
    identical = "The answer is Paris capital of France"
    client = _fake_client(identical)
    est = UncertaintyEstimator(
        min_samples=2,
        max_samples=5,
        early_stop_jaccard=0.6,
        enable_semantic_sampling=True,
    )
    conf, samples = await est.semantic_entropy([], client, "test")
    assert samples == 2
    assert conf == pytest.approx(1.0)


async def test_semantic_entropy_extends_on_disagreement() -> None:
    responses = [
        "Paris is the capital of France",
        "Berlin zoo is in Germany",
        "London bridge is falling down",
        "Tokyo subway system Japan",
        "Sydney opera house Australia",
    ]
    call_count = 0

    async def fake_complete(**kwargs: Any) -> ModelResponse:
        nonlocal call_count
        content = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return ModelResponse(
            content=content,
            tool_calls=[],
            token_usage=TokenUsage(
                input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.0, model="t"
            ),
            model="t",
            stop_reason="end_turn",
        )

    client = MagicMock()
    client.complete = fake_complete
    est = UncertaintyEstimator(
        min_samples=2,
        max_samples=5,
        early_stop_jaccard=0.6,
        enable_semantic_sampling=True,
    )
    conf, samples = await est.semantic_entropy([], client, "test")
    assert samples == 5
    assert 0.0 <= conf <= 1.0


async def test_semantic_entropy_unanimous() -> None:
    client = _fake_client("same exact answer every time")
    est = UncertaintyEstimator(min_samples=2, max_samples=3, early_stop_jaccard=0.6)
    conf, samples = await est.semantic_entropy([], client, "test")
    assert conf == pytest.approx(1.0)
    assert samples == 2


async def test_semantic_entropy_split() -> None:
    responses = ["aaa bbb ccc", "ddd eee fff", "ggg hhh iii", "jjj kkk lll", "mmm nnn ooo"]
    call_count = 0

    async def fake_complete(**kwargs: Any) -> ModelResponse:
        nonlocal call_count
        content = responses[call_count % len(responses)]
        call_count += 1
        return ModelResponse(
            content=content,
            tool_calls=[],
            token_usage=TokenUsage(
                input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.0, model="t"
            ),
            model="t",
            stop_reason="end_turn",
        )

    client = MagicMock()
    client.complete = fake_complete
    est = UncertaintyEstimator(min_samples=2, max_samples=5, early_stop_jaccard=0.6)
    conf, samples = await est.semantic_entropy([], client, "test")
    assert conf <= 0.3
    assert samples == 5


async def test_semantic_entropy_pessimistic_fusion() -> None:
    client_entropy = _fake_client("Paris France capital city Europe")
    est = UncertaintyEstimator(
        min_samples=2,
        max_samples=2,
        early_stop_jaccard=0.6,
        enable_semantic_sampling=True,
        semantic_trigger_low=0.50,
        semantic_trigger_high=0.80,
    )
    fused_before = 0.70
    assert est.should_sample(fused_before)
    entropy_conf, _ = await est.semantic_entropy([], client_entropy, "test")
    final = min(fused_before, entropy_conf)
    assert final <= fused_before


async def test_semantic_sampling_not_triggered_outside_zone() -> None:
    est = UncertaintyEstimator(
        semantic_trigger_low=0.50,
        semantic_trigger_high=0.72,
        enable_semantic_sampling=True,
    )
    assert not est.should_sample(0.85)
    assert not est.should_sample(0.30)


# ---------------------------------------------------------------------------
# 6. Propagation (SAUP)
# ---------------------------------------------------------------------------


async def test_propagator_decision_step() -> None:
    prop = UncertaintyPropagator()
    result = prop.propagate(
        fused_confidence=0.90,
        step_type="decision",
        cumulative_confidence=0.60,
    )
    expected = 0.70 * 0.90 + 0.30 * 0.60
    assert result == pytest.approx(expected)


async def test_propagator_accumulation_decay() -> None:
    prop = UncertaintyPropagator()
    c = 1.0
    p1 = prop.propagate(0.9, "llm_call", c)
    c = prop.update_cumulative(p1, c, "llm_call")
    after_confident = c

    p2 = prop.propagate(0.3, "llm_call", c)
    c = prop.update_cumulative(p2, c, "llm_call")

    p3 = prop.propagate(0.9, "llm_call", c)
    c = prop.update_cumulative(p3, c, "llm_call")
    after_all = c

    assert after_all < after_confident


async def test_propagator_memory_read_discounted() -> None:
    prop = UncertaintyPropagator()
    result = prop.propagate(
        fused_confidence=0.40,
        step_type="memory_read",
        cumulative_confidence=0.80,
    )
    expected = 0.35 * 0.40 + 0.65 * 0.80
    assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 7. Policy
# ---------------------------------------------------------------------------


async def test_policy_all_thresholds() -> None:
    policy = UncertaintyPolicy(
        low_threshold=0.80,
        medium_threshold=0.60,
        high_threshold=0.40,
        irreversible_tool_names=[],
    )
    assert policy.classify(0.85) == UncertaintyLevel.LOW
    assert policy.decide(UncertaintyLevel.LOW, "llm_call") == UncertaintyAction.PROCEED

    assert policy.classify(0.70) == UncertaintyLevel.MEDIUM
    assert policy.decide(UncertaintyLevel.MEDIUM, "llm_call") == UncertaintyAction.PROCEED_WITH_LOG

    assert policy.classify(0.50) == UncertaintyLevel.HIGH
    assert policy.decide(UncertaintyLevel.HIGH, "llm_call") == UncertaintyAction.PAUSE_FOR_HUMAN

    assert policy.classify(0.30) == UncertaintyLevel.CRITICAL
    assert policy.decide(UncertaintyLevel.CRITICAL, "llm_call") == UncertaintyAction.ABORT


async def test_policy_irreversible_escalates_at_medium() -> None:
    policy = UncertaintyPolicy(irreversible_tool_names=["send_email"])
    action = policy.decide(UncertaintyLevel.MEDIUM, "tool_call", tool_name="send_email")
    assert action == UncertaintyAction.PAUSE_FOR_HUMAN


async def test_policy_irreversible_safe_at_low() -> None:
    policy = UncertaintyPolicy(irreversible_tool_names=["send_email"])
    action = policy.decide(UncertaintyLevel.LOW, "tool_call", tool_name="send_email")
    assert action == UncertaintyAction.PROCEED


async def test_policy_non_irreversible_medium_proceeds() -> None:
    policy = UncertaintyPolicy(irreversible_tool_names=["send_email"])
    action = policy.decide(UncertaintyLevel.MEDIUM, "tool_call", tool_name="search")
    assert action == UncertaintyAction.PROCEED_WITH_LOG


# ---------------------------------------------------------------------------
# 8. Monitor + AgentRunner integration
# ---------------------------------------------------------------------------


async def test_monitor_full_pipeline_low() -> None:
    client = _fake_client('{"answer": "Paris", "confidence": 0.95}')
    client.complete = AsyncMock(
        side_effect=[
            ModelResponse(
                content='{"p_true": 0.90, "reason": "confident"}',
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=5, output_tokens=5, total_tokens=10, cost_usd=0.0, model="t"
                ),
                model="t",
                stop_reason="end_turn",
            )
        ]
    )
    policy = UncertaintyPolicy(enable_p_true=True)
    monitor = UncertaintyMonitor(policy=policy)
    monitor.initialize("s1", "a1")
    step, action = await monitor.observe_llm_response(
        response_text='{"answer": "Paris", "confidence": 0.95}',
        step_id="step_0",
        model_client=client,
        model_id="t",
    )
    assert step.level == UncertaintyLevel.LOW
    assert action == UncertaintyAction.PROCEED
    assert len(monitor.get_belief_state().step_uncertainties) == 1


async def test_monitor_full_pipeline_pause() -> None:
    low_conf_resp = '{"answer": "maybe", "confidence": 0.10}'
    p_true_resp = '{"p_true": 0.05, "reason": "unsure"}'
    client = MagicMock()
    client.complete = AsyncMock(
        return_value=ModelResponse(
            content=p_true_resp,
            tool_calls=[],
            token_usage=TokenUsage(
                input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.0, model="t"
            ),
            model="t",
            stop_reason="end_turn",
        )
    )
    policy = UncertaintyPolicy(enable_p_true=True, high_threshold=0.40)
    monitor = UncertaintyMonitor(policy=policy)
    monitor.initialize("s1", "a1")
    step, action = await monitor.observe_llm_response(
        response_text=low_conf_resp,
        step_id="step_0",
        model_client=client,
        model_id="t",
    )
    assert step.level in (UncertaintyLevel.HIGH, UncertaintyLevel.CRITICAL)
    assert action in (UncertaintyAction.PAUSE_FOR_HUMAN, UncertaintyAction.ABORT)
    assert monitor.get_belief_state().high_uncertainty_steps == 1


async def test_monitor_tool_call_irreversible_blocked() -> None:
    policy = UncertaintyPolicy(
        low_threshold=0.80,
        medium_threshold=0.60,
        high_threshold=0.40,
        irreversible_tool_names=["delete_records"],
    )
    monitor = UncertaintyMonitor(policy=policy)
    monitor.initialize("s1", "a1")
    monitor._belief.cumulative_confidence = 0.65
    _, action = await monitor.observe_tool_call(
        tool_name="delete_records",
        step_id="tool_0",
    )
    assert action == UncertaintyAction.PAUSE_FOR_HUMAN


async def test_monitor_initialize_clears_state() -> None:
    client = _fake_client('{"confidence": 0.5}')
    policy = UncertaintyPolicy(enable_p_true=False)
    monitor = UncertaintyMonitor(policy=policy)
    monitor.initialize("s1", "a1")
    await monitor.observe_llm_response("some text", "s0", model_client=client)
    assert len(monitor.get_belief_state().step_uncertainties) == 1

    monitor.initialize("s2", "a2")
    state = monitor.get_belief_state()
    assert len(state.step_uncertainties) == 0
    assert state.cumulative_confidence == pytest.approx(1.0)


async def test_runner_uncertainty_pause_on_low_confidence() -> None:
    llm_response = '{"answer": "maybe", "confidence": 0.10}'
    p_true_response = '{"p_true": 0.05, "reason": "very unsure"}'

    call_results = [
        ModelResponse(
            content=llm_response,
            tool_calls=[],
            token_usage=TokenUsage(
                input_tokens=5, output_tokens=5, total_tokens=10, cost_usd=0.0, model="t"
            ),
            model="t",
            stop_reason="end_turn",
        ),
        ModelResponse(
            content=p_true_response,
            tool_calls=[],
            token_usage=TokenUsage(
                input_tokens=5, output_tokens=5, total_tokens=10, cost_usd=0.0, model="t"
            ),
            model="t",
            stop_reason="end_turn",
        ),
    ]
    client = MagicMock()
    client.complete = AsyncMock(side_effect=call_results)

    tool_exec = MagicMock()
    tool_exec.execute = AsyncMock(return_value=_tool_result())

    policy = UncertaintyPolicy(
        enable_p_true=True,
        low_threshold=0.80,
        medium_threshold=0.60,
        high_threshold=0.40,
        inject_reflection_on_high=False,
    )
    monitor = UncertaintyMonitor(policy=policy)
    runner = AgentRunner(client, tool_exec, uncertainty_monitor=monitor)
    result = await runner.run(_agent_def(), "question", session_id="s1")
    assert result.status == AgentStatus.WAITING_FOR_HUMAN


async def test_runner_uncertainty_abort_on_critical() -> None:
    llm_response = '{"answer": "idk", "confidence": 0.01}'
    p_true_response = '{"p_true": 0.01, "reason": "no idea"}'

    client = MagicMock()
    client.complete = AsyncMock(
        side_effect=[
            ModelResponse(
                content=llm_response,
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.0, model="t"
                ),
                model="t",
                stop_reason="end_turn",
            ),
            ModelResponse(
                content=p_true_response,
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.0, model="t"
                ),
                model="t",
                stop_reason="end_turn",
            ),
        ]
    )
    tool_exec = MagicMock()
    tool_exec.execute = AsyncMock(return_value=_tool_result())

    policy = UncertaintyPolicy(
        enable_p_true=True,
        low_threshold=0.99,
        medium_threshold=0.80,
        high_threshold=0.60,
    )
    monitor = UncertaintyMonitor(policy=policy)
    runner = AgentRunner(client, tool_exec, uncertainty_monitor=monitor)
    with pytest.raises(UncertaintyError):
        await runner.run(_agent_def(), "question", session_id="s1")


async def test_runner_irreversible_tool_blocks_execution() -> None:
    # p_true=0.30 → fused≈0.312 → propagated≈0.622 (MEDIUM, LLM proceeds with log)
    # cumulative_after_llm≈0.792 (MEDIUM) → tool check on irreversible "delete_file" → PAUSE
    llm_content = "I will delete the file."
    p_true_resp = '{"p_true": 0.30, "reason": "somewhat unsure"}'
    tool_call = ToolCall(id="tc-1", name="delete_file", arguments={})

    client = MagicMock()
    client.complete = AsyncMock(
        side_effect=[
            ModelResponse(
                content=llm_content,
                tool_calls=[tool_call],
                token_usage=TokenUsage(
                    input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.0, model="t"
                ),
                model="t",
                stop_reason="tool_use",
            ),
            ModelResponse(
                content=p_true_resp,
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.0, model="t"
                ),
                model="t",
                stop_reason="end_turn",
            ),
        ]
    )
    tool_exec = MagicMock()
    tool_exec.execute = AsyncMock(return_value=_tool_result("tc-1"))

    policy = UncertaintyPolicy(
        enable_p_true=True,
        low_threshold=0.80,
        medium_threshold=0.60,
        high_threshold=0.40,
        irreversible_tool_names=["delete_file"],
    )
    monitor = UncertaintyMonitor(policy=policy)
    runner = AgentRunner(client, tool_exec, uncertainty_monitor=monitor)
    result = await runner.run(_agent_def(), "delete the file", session_id="s1")
    assert result.status == AgentStatus.WAITING_FOR_HUMAN
    contents = [m.content for m in result.messages if m.role == Role.SYSTEM]
    assert any("delete_file" in (c or "") for c in contents)


async def test_runner_reflection_injected_before_pause() -> None:
    llm_response = '{"answer": "dunno", "confidence": 0.05}'
    p_true_response = '{"p_true": 0.03, "reason": "nope"}'

    client = MagicMock()
    client.complete = AsyncMock(
        side_effect=[
            ModelResponse(
                content=llm_response,
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.0, model="t"
                ),
                model="t",
                stop_reason="end_turn",
            ),
            ModelResponse(
                content=p_true_response,
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.0, model="t"
                ),
                model="t",
                stop_reason="end_turn",
            ),
        ]
    )
    tool_exec = MagicMock()
    tool_exec.execute = AsyncMock(return_value=_tool_result())

    policy = UncertaintyPolicy(
        enable_p_true=True,
        low_threshold=0.80,
        medium_threshold=0.60,
        high_threshold=0.40,
        inject_reflection_on_high=True,
    )
    monitor = UncertaintyMonitor(policy=policy)
    runner = AgentRunner(client, tool_exec, uncertainty_monitor=monitor)
    result = await runner.run(_agent_def(), "question", session_id="s1")
    assert result.status == AgentStatus.WAITING_FOR_HUMAN
    system_contents = [m.content or "" for m in result.messages if m.role == Role.SYSTEM]
    assert any("what you don't know" in c for c in system_contents)


async def test_runner_proceed_with_log_does_not_pause() -> None:
    resp_text = '{"answer": "ok", "confidence": 0.72}'
    p_true_resp = '{"p_true": 0.68, "reason": "mostly sure"}'
    final_resp = "Final answer."

    client = MagicMock()
    client.complete = AsyncMock(
        side_effect=[
            ModelResponse(
                content=resp_text,
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.0, model="t"
                ),
                model="t",
                stop_reason="end_turn",
            ),
            ModelResponse(
                content=p_true_resp,
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.0, model="t"
                ),
                model="t",
                stop_reason="end_turn",
            ),
            ModelResponse(
                content=final_resp,
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.0, model="t"
                ),
                model="t",
                stop_reason="end_turn",
            ),
            ModelResponse(
                content=p_true_resp,
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.0, model="t"
                ),
                model="t",
                stop_reason="end_turn",
            ),
        ]
    )
    tool_exec = MagicMock()
    tool_exec.execute = AsyncMock(return_value=_tool_result())

    policy = UncertaintyPolicy(
        enable_p_true=True,
        low_threshold=0.99,
        medium_threshold=0.60,
        high_threshold=0.10,
    )
    monitor = UncertaintyMonitor(policy=policy)
    runner = AgentRunner(client, tool_exec, uncertainty_monitor=monitor)
    result = await runner.run(_agent_def(), "question", session_id="s1")
    assert result.status == AgentStatus.COMPLETED


# ---------------------------------------------------------------------------
# 9. uncertainty_guard_node
# ---------------------------------------------------------------------------


async def test_uncertainty_guard_node_escalates() -> None:
    from grampus.orchestration.nodes import uncertainty_guard_node

    mock_monitor = MagicMock()
    mock_monitor.observe_llm_response = AsyncMock(
        return_value=(
            MagicMock(spec=StepUncertainty),
            UncertaintyAction.PAUSE_FOR_HUMAN,
        )
    )
    mock_monitor.summary_metadata = MagicMock(return_value={"overall_level": "high"})

    state = AgentState(agent_id="a", session_id="s")
    state.messages.append(Message(role=Role.ASSISTANT, content="unsure answer"))
    state.status = AgentStatus.RUNNING

    handler = uncertainty_guard_node(mock_monitor, escalate_node="review")
    new_state = await handler(state)

    assert new_state.status == AgentStatus.WAITING_FOR_HUMAN
    assert new_state.metadata.get("uncertainty_escalate") is True


async def test_uncertainty_guard_node_proceeds() -> None:
    from grampus.orchestration.nodes import uncertainty_guard_node

    mock_monitor = MagicMock()
    mock_monitor.observe_llm_response = AsyncMock(
        return_value=(
            MagicMock(spec=StepUncertainty),
            UncertaintyAction.PROCEED,
        )
    )
    mock_monitor.summary_metadata = MagicMock(return_value={"overall_level": "low"})

    state = AgentState(agent_id="a", session_id="s")
    state.messages.append(Message(role=Role.ASSISTANT, content="confident answer"))
    state.status = AgentStatus.RUNNING

    handler = uncertainty_guard_node(mock_monitor)
    new_state = await handler(state)

    assert new_state.status == AgentStatus.RUNNING
    assert "uncertainty_escalate" not in new_state.metadata


# ---------------------------------------------------------------------------
# 10. OTEL spans
# ---------------------------------------------------------------------------


async def test_otel_spans_emitted() -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    fake_tracer = MagicMock()
    fake_tracer._tracer = provider.get_tracer("test")

    policy = UncertaintyPolicy(enable_p_true=False, high_threshold=0.99)
    monitor = UncertaintyMonitor(policy=policy, tracer=fake_tracer)
    monitor.initialize("s1", "a1")

    await monitor.observe_llm_response(
        response_text='{"confidence": 0.95}',
        step_id="span_test_0",
        model_client=None,
    )

    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]
    assert "uncertainty.estimate" in span_names
    estimate_span = next(s for s in spans if s.name == "uncertainty.estimate")
    attrs = dict(estimate_span.attributes or {})
    assert attrs.get("step_id") == "span_test_0"


async def test_otel_semantic_span_only_when_sampled() -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    fake_tracer = MagicMock()
    fake_tracer._tracer = provider.get_tracer("test2")

    policy = UncertaintyPolicy(enable_p_true=False, enable_semantic_sampling=False)
    monitor = UncertaintyMonitor(policy=policy, tracer=fake_tracer)
    monitor.initialize("s1", "a1")

    await monitor.observe_llm_response(
        response_text='{"confidence": 0.80}',
        step_id="no_semantic_0",
        model_client=None,
    )

    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]
    assert "uncertainty.semantic" not in span_names
