"""Tests for ActionGuard — allowlist/denylist, rate limiting, cost guard."""

from __future__ import annotations

from unittest.mock import patch

from nexus.tools.boundaries import ActionGuard, BoundaryConfig, GuardResult


def _make_guard(
    *,
    allowed_tools: list[str] | None = None,
    denied_tools: list[str] | None = None,
    max_calls_per_minute: int = 60,
    max_cost_usd: float | None = None,
    agent_id: str = "test-agent",
) -> ActionGuard:
    return ActionGuard(
        BoundaryConfig(
            allowed_tools=allowed_tools,
            denied_tools=denied_tools or [],
            max_calls_per_minute=max_calls_per_minute,
            max_cost_usd=max_cost_usd,
            agent_id=agent_id,
        )
    )


# ---------------------------------------------------------------------------
# Allowlist / denylist
# ---------------------------------------------------------------------------


def test_check_tool_allows_when_no_restrictions() -> None:
    guard = _make_guard()
    result = guard.check_tool("any_tool")
    assert result.allowed is True
    assert result.reason is None


def test_check_tool_blocks_denied_tool() -> None:
    guard = _make_guard(denied_tools=["dangerous_tool"])
    result = guard.check_tool("dangerous_tool")
    assert result.allowed is False
    assert result.reason is not None


def test_check_tool_allows_tool_not_in_denylist() -> None:
    guard = _make_guard(denied_tools=["dangerous_tool"])
    result = guard.check_tool("safe_tool")
    assert result.allowed is True


def test_check_tool_allows_tool_in_allowlist() -> None:
    guard = _make_guard(allowed_tools=["good_tool", "other_tool"])
    result = guard.check_tool("good_tool")
    assert result.allowed is True


def test_check_tool_blocks_tool_not_in_allowlist() -> None:
    guard = _make_guard(allowed_tools=["good_tool"])
    result = guard.check_tool("unlisted_tool")
    assert result.allowed is False
    assert result.reason is not None


def test_denylist_overrides_allowlist() -> None:
    guard = _make_guard(allowed_tools=["my_tool"], denied_tools=["my_tool"])
    result = guard.check_tool("my_tool")
    assert result.allowed is False


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limit_allows_calls_under_threshold() -> None:
    guard = _make_guard(max_calls_per_minute=5)
    for _ in range(4):
        guard.record_call("tool_a")
    result = guard.check_tool("tool_a")
    assert result.allowed is True


def test_rate_limit_blocks_at_threshold() -> None:
    guard = _make_guard(max_calls_per_minute=3)
    for _ in range(3):
        guard.record_call("any_tool")
    result = guard.check_tool("any_tool")
    assert result.allowed is False
    assert result.reason is not None


def test_rate_limit_resets_after_window() -> None:
    """Timestamps older than 60 s are evicted from the sliding window."""
    guard = _make_guard(max_calls_per_minute=2)

    # Simulate two calls that happened 61 seconds ago.
    old_ts = 0.0
    with patch("nexus.tools.boundaries.time.monotonic", return_value=old_ts):
        guard.record_call("tool_x")
        guard.record_call("tool_x")

    # Now at t=61 s: the old calls are outside the 60-second window.
    with patch("nexus.tools.boundaries.time.monotonic", return_value=61.0):
        result = guard.check_tool("tool_x")

    assert result.allowed is True


def test_record_call_accumulates_cost() -> None:
    guard = _make_guard(max_cost_usd=0.80)
    guard.record_call("tool_a", cost_usd=0.30)
    guard.record_call("tool_b", cost_usd=0.25)
    # cumulative = 0.55; 0.55 + 0.40 = 0.95 > 0.80 → blocked
    assert guard.check_budget(0.40).allowed is False


# ---------------------------------------------------------------------------
# Budget guard
# ---------------------------------------------------------------------------


def test_budget_allows_when_no_limit_set() -> None:
    guard = _make_guard(max_cost_usd=None)
    assert guard.check_budget(9999.0).allowed is True


def test_budget_allows_when_under_limit() -> None:
    guard = _make_guard(max_cost_usd=1.0)
    guard.record_call("tool_a", cost_usd=0.40)
    result = guard.check_budget(0.50)
    assert result.allowed is True


def test_budget_blocks_when_over_limit() -> None:
    guard = _make_guard(max_cost_usd=1.0)
    guard.record_call("tool_a", cost_usd=0.80)
    result = guard.check_budget(0.30)
    assert result.allowed is False
    assert result.reason is not None


def test_budget_blocks_at_exact_limit() -> None:
    guard = _make_guard(max_cost_usd=1.0)
    guard.record_call("tool_a", cost_usd=0.70)
    result = guard.check_budget(0.30)
    # 0.70 + 0.30 == 1.0 → not strictly greater, should still be allowed
    assert result.allowed is True


def test_budget_blocks_beyond_exact_limit() -> None:
    guard = _make_guard(max_cost_usd=1.0)
    guard.record_call("tool_a", cost_usd=0.70)
    result = guard.check_budget(0.31)
    assert result.allowed is False


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset_clears_call_history_and_cost() -> None:
    guard = _make_guard(max_calls_per_minute=2, max_cost_usd=1.0)
    guard.record_call("tool_a", cost_usd=0.50)
    guard.record_call("tool_a", cost_usd=0.50)

    guard.reset()

    assert guard.check_tool("tool_a").allowed is True
    assert guard.check_budget(0.99).allowed is True


# ---------------------------------------------------------------------------
# GuardResult model
# ---------------------------------------------------------------------------


def test_guard_result_round_trips_json() -> None:
    result = GuardResult(allowed=False, reason="denied")
    assert GuardResult.model_validate_json(result.model_dump_json()) == result
