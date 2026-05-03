"""Action guard — per-agent allowlist/denylist, rate limiting, and cost budgeting."""

from __future__ import annotations

import time
from collections import deque

from pydantic import BaseModel, Field

from nexus.core.logging import get_logger

logger = get_logger(__name__)


class BoundaryConfig(BaseModel):
    """Configuration for an ActionGuard instance.

    Attributes:
        allowed_tools: Explicit allowlist; None means all tools are permitted.
        denied_tools: Tools always blocked regardless of allowlist.
        max_calls_per_minute: Sliding-window rate limit across all tool calls.
        max_cost_usd: Cumulative cost ceiling; None means no limit.
        agent_id: Identifies the agent this guard belongs to.
    """

    allowed_tools: list[str] | None = None
    denied_tools: list[str] = Field(default_factory=list)
    max_calls_per_minute: int = 60
    max_cost_usd: float | None = None
    agent_id: str


class GuardResult(BaseModel):
    """Result of an ActionGuard check.

    Attributes:
        allowed: Whether the action is permitted.
        reason: Human-readable explanation when *allowed* is False.
    """

    allowed: bool
    reason: str | None = None


class ActionGuard:
    """Enforces per-agent tool access boundaries, rate limits, and cost budgets.

    Args:
        config: Boundary configuration for this agent.
    """

    def __init__(self, config: BoundaryConfig) -> None:
        self._config = config
        self._call_timestamps: deque[float] = deque()
        self._cumulative_cost_usd: float = 0.0

    def check_tool(self, tool_name: str) -> GuardResult:
        """Check whether *tool_name* may be called right now.

        Denylist is evaluated first; allowlist (when set) is evaluated second;
        rate limit is evaluated last.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            GuardResult indicating whether the call is permitted.
        """
        if tool_name in self._config.denied_tools:
            logger.debug("guard.denied", agent=self._config.agent_id, tool=tool_name)
            return GuardResult(allowed=False, reason=f"Tool '{tool_name}' is in the denylist")

        if self._config.allowed_tools is not None and tool_name not in self._config.allowed_tools:
            logger.debug("guard.not_allowed", agent=self._config.agent_id, tool=tool_name)
            return GuardResult(allowed=False, reason=f"Tool '{tool_name}' is not in the allowlist")

        rate_result = self._check_rate_limit()
        if not rate_result.allowed:
            return rate_result

        return GuardResult(allowed=True)

    def record_call(self, tool_name: str, cost_usd: float = 0.0) -> None:
        """Record that a tool call happened at this moment and accumulate its cost.

        Args:
            tool_name: Name of the tool that was called.
            cost_usd: Cost of this call in US dollars.
        """
        now = time.monotonic()
        self._call_timestamps.append(now)
        self._cumulative_cost_usd += cost_usd
        logger.debug(
            "guard.recorded",
            agent=self._config.agent_id,
            tool=tool_name,
            cost_usd=cost_usd,
            cumulative_cost=self._cumulative_cost_usd,
        )

    def check_budget(self, estimated_cost_usd: float) -> GuardResult:
        """Check whether an additional *estimated_cost_usd* would exceed the budget.

        Args:
            estimated_cost_usd: Projected cost of the next action.

        Returns:
            GuardResult indicating whether the budget permits this action.
        """
        if self._config.max_cost_usd is None:
            return GuardResult(allowed=True)

        projected = self._cumulative_cost_usd + estimated_cost_usd
        if projected > self._config.max_cost_usd:
            logger.debug(
                "guard.budget_exceeded",
                agent=self._config.agent_id,
                projected=projected,
                limit=self._config.max_cost_usd,
            )
            return GuardResult(
                allowed=False,
                reason=(
                    f"Projected cost ${projected:.4f} exceeds budget "
                    f"${self._config.max_cost_usd:.4f}"
                ),
            )
        return GuardResult(allowed=True)

    def reset(self) -> None:
        """Reset call history and accumulated cost (useful for testing)."""
        self._call_timestamps.clear()
        self._cumulative_cost_usd = 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_rate_limit(self) -> GuardResult:
        """Evict stale timestamps and check against max_calls_per_minute."""
        now = time.monotonic()
        window_start = now - 60.0
        while self._call_timestamps and self._call_timestamps[0] <= window_start:
            self._call_timestamps.popleft()

        if len(self._call_timestamps) >= self._config.max_calls_per_minute:
            logger.debug(
                "guard.rate_limited",
                agent=self._config.agent_id,
                calls_in_window=len(self._call_timestamps),
                limit=self._config.max_calls_per_minute,
            )
            return GuardResult(
                allowed=False,
                reason=(f"Rate limit of {self._config.max_calls_per_minute} calls/minute exceeded"),
            )
        return GuardResult(allowed=True)

    def _evict_stale(self, now: float) -> None:
        """Remove timestamps outside the 60-second window."""
        window_start = now - 60.0
        while self._call_timestamps and self._call_timestamps[0] <= window_start:
            self._call_timestamps.popleft()
