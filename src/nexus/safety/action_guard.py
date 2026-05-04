"""Orchestration-level action guard — policy-driven tool call enforcement."""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import BaseModel, Field

from nexus.core.logging import get_logger
from nexus.core.types import ToolCall

_log = get_logger(__name__)


class ActionPolicy(BaseModel):
    """Per-agent action policy loaded from YAML."""

    agent_id: str
    allowed_tools: list[str] | None = None
    denied_tools: list[str] = Field(default_factory=list)
    max_tool_calls_per_turn: int = 20
    max_consecutive_tool_calls: int = 5
    denied_domains: list[str] = Field(default_factory=list)
    require_human_approval_for: list[str] = Field(default_factory=list)


class ActionCheckResult(BaseModel):
    """Result of an action policy check."""

    allowed: bool
    requires_human_approval: bool = False
    reason: str | None = None


class SafetyActionGuard:
    """Orchestration-level action guard enforcing policy rules.

    Can be constructed either with an ``ActionPolicy`` object or with inline
    keyword arguments (``allowed_tools``, ``denied_tools``,
    ``max_tool_calls_per_turn``) for quick one-off usage.

    Args:
        policy: Full ActionPolicy for this agent.  When provided, all keyword
            arguments are ignored.
        allowed_tools: Optional allowlist — when set only listed tools are
            permitted.
        denied_tools: Explicit denylist of tool names.
        max_tool_calls_per_turn: Hard cap on tool calls per agent turn.
    """

    def __init__(
        self,
        policy: ActionPolicy | None = None,
        *,
        allowed_tools: list[str] | None = None,
        denied_tools: list[str] | None = None,
        max_tool_calls_per_turn: int = 20,
        max_consecutive_calls: int = 5,
    ) -> None:
        if policy is None:
            policy = ActionPolicy(
                agent_id="default",
                allowed_tools=allowed_tools,
                denied_tools=denied_tools or [],
                max_tool_calls_per_turn=max_tool_calls_per_turn,
                max_consecutive_tool_calls=max_consecutive_calls,
            )
        self._policy = policy

    def check_tool_call(
        self,
        tool_call: ToolCall,
        *,
        calls_this_turn: int = 0,
        consecutive_calls: int = 0,
    ) -> ActionCheckResult:
        """Check whether a tool call is permitted under policy.

        Args:
            tool_call: The tool call to evaluate.
            calls_this_turn: Total tool calls already made in the current turn.
            consecutive_calls: Number of consecutive tool calls without a non-tool step.

        Returns:
            ActionCheckResult indicating whether the call is allowed.
        """
        name = tool_call.name

        denied = self._check_denied(name)
        if denied is not None:
            return denied

        not_allowed = self._check_allowlist(name)
        if not_allowed is not None:
            return not_allowed

        over_turn = self._check_turn_limit(calls_this_turn)
        if over_turn is not None:
            return over_turn

        over_consecutive = self._check_consecutive_limit(consecutive_calls)
        if over_consecutive is not None:
            return over_consecutive

        needs_approval = name in self._policy.require_human_approval_for
        return ActionCheckResult(allowed=True, requires_human_approval=needs_approval)

    def check_url(self, url: str) -> ActionCheckResult:
        """Check whether a URL's domain is permitted.

        Args:
            url: The URL to evaluate.

        Returns:
            ActionCheckResult indicating whether the domain is allowed.
        """
        try:
            host = urlparse(url).hostname or ""
        except Exception:
            return ActionCheckResult(allowed=False, reason="Invalid URL")

        for denied in self._policy.denied_domains:
            if host == denied or host.endswith(f".{denied}"):
                _log.debug("guard.url_blocked", domain=host, agent=self._policy.agent_id)
                return ActionCheckResult(allowed=False, reason=f"Domain '{host}' is denied")

        return ActionCheckResult(allowed=True)

    def reset_turn(self) -> None:
        """Reset per-turn state. Call at the start of each agent turn."""
        _log.debug("guard.turn_reset", agent=self._policy.agent_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_denied(self, name: str) -> ActionCheckResult | None:
        if name in self._policy.denied_tools:
            _log.debug("guard.denied", tool=name, agent=self._policy.agent_id)
            return ActionCheckResult(allowed=False, reason=f"Tool '{name}' is in the denylist")
        return None

    def _check_allowlist(self, name: str) -> ActionCheckResult | None:
        if self._policy.allowed_tools is None:
            return None
        if name not in self._policy.allowed_tools:
            _log.debug("guard.not_allowed", tool=name, agent=self._policy.agent_id)
            return ActionCheckResult(allowed=False, reason=f"Tool '{name}' is not in the allowlist")
        return None

    def _check_turn_limit(self, calls_this_turn: int) -> ActionCheckResult | None:
        if calls_this_turn >= self._policy.max_tool_calls_per_turn:
            return ActionCheckResult(
                allowed=False,
                reason=f"Per-turn limit of {self._policy.max_tool_calls_per_turn} calls exceeded",
            )
        return None

    def _check_consecutive_limit(self, consecutive_calls: int) -> ActionCheckResult | None:
        if consecutive_calls >= self._policy.max_consecutive_tool_calls:
            return ActionCheckResult(
                allowed=False,
                reason=f"Consecutive call limit of {self._policy.max_consecutive_tool_calls} exceeded",
            )
        return None
