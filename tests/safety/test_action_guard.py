"""Tests for SafetyActionGuard."""

from __future__ import annotations

from nexus.core.types import ToolCall
from nexus.safety.action_guard import ActionCheckResult, ActionPolicy, SafetyActionGuard


def _make_tool_call(name: str) -> ToolCall:
    return ToolCall(id="tc-1", name=name, arguments={})


class TestSafetyActionGuard:
    def test_allows_tool_not_in_denylist(self) -> None:
        policy = ActionPolicy(agent_id="a1")
        guard = SafetyActionGuard(policy)
        result = guard.check_tool_call(_make_tool_call("search"))
        assert result.allowed

    def test_blocks_denied_tool(self) -> None:
        policy = ActionPolicy(agent_id="a1", denied_tools=["shell"])
        guard = SafetyActionGuard(policy)
        result = guard.check_tool_call(_make_tool_call("shell"))
        assert not result.allowed
        assert result.reason is not None

    def test_blocks_tool_not_in_allowlist(self) -> None:
        policy = ActionPolicy(agent_id="a1", allowed_tools=["search", "read"])
        guard = SafetyActionGuard(policy)
        result = guard.check_tool_call(_make_tool_call("shell"))
        assert not result.allowed

    def test_allows_tool_in_allowlist(self) -> None:
        policy = ActionPolicy(agent_id="a1", allowed_tools=["search"])
        guard = SafetyActionGuard(policy)
        result = guard.check_tool_call(_make_tool_call("search"))
        assert result.allowed

    def test_blocks_when_max_calls_per_turn_exceeded(self) -> None:
        policy = ActionPolicy(agent_id="a1", max_tool_calls_per_turn=3)
        guard = SafetyActionGuard(policy)
        result = guard.check_tool_call(_make_tool_call("search"), calls_this_turn=3)
        assert not result.allowed

    def test_allows_when_at_limit_not_exceeded(self) -> None:
        policy = ActionPolicy(agent_id="a1", max_tool_calls_per_turn=3)
        guard = SafetyActionGuard(policy)
        result = guard.check_tool_call(_make_tool_call("search"), calls_this_turn=2)
        assert result.allowed

    def test_blocks_when_consecutive_calls_exceeded(self) -> None:
        policy = ActionPolicy(agent_id="a1", max_consecutive_tool_calls=5)
        guard = SafetyActionGuard(policy)
        result = guard.check_tool_call(_make_tool_call("search"), consecutive_calls=5)
        assert not result.allowed

    def test_requires_approval_for_configured_tool(self) -> None:
        policy = ActionPolicy(
            agent_id="a1",
            require_human_approval_for=["send_email"],
        )
        guard = SafetyActionGuard(policy)
        result = guard.check_tool_call(_make_tool_call("send_email"))
        assert result.allowed
        assert result.requires_human_approval

    def test_does_not_require_approval_for_other_tool(self) -> None:
        policy = ActionPolicy(
            agent_id="a1",
            require_human_approval_for=["send_email"],
        )
        guard = SafetyActionGuard(policy)
        result = guard.check_tool_call(_make_tool_call("search"))
        assert not result.requires_human_approval

    def test_check_url_blocks_denied_domain(self) -> None:
        policy = ActionPolicy(agent_id="a1", denied_domains=["evil.com"])
        guard = SafetyActionGuard(policy)
        result = guard.check_url("https://evil.com/path")
        assert not result.allowed

    def test_check_url_allows_permitted_domain(self) -> None:
        policy = ActionPolicy(agent_id="a1", denied_domains=["evil.com"])
        guard = SafetyActionGuard(policy)
        result = guard.check_url("https://good.com/path")
        assert result.allowed

    def test_reset_turn_resets_counters(self) -> None:
        policy = ActionPolicy(agent_id="a1", max_tool_calls_per_turn=2)
        guard = SafetyActionGuard(policy)
        # Exceeds limit
        result = guard.check_tool_call(_make_tool_call("search"), calls_this_turn=2)
        assert not result.allowed
        # After reset, internal state allows again (calls_this_turn is caller-tracked)
        guard.reset_turn()
        result2 = guard.check_tool_call(_make_tool_call("search"), calls_this_turn=0)
        assert result2.allowed

    def test_action_check_result_is_pydantic(self) -> None:
        policy = ActionPolicy(agent_id="a1")
        guard = SafetyActionGuard(policy)
        result = guard.check_tool_call(_make_tool_call("x"))
        assert isinstance(result, ActionCheckResult)
