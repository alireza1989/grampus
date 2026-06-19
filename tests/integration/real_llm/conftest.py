"""Shared fixtures for real-LLM integration tests.

Skipped entirely unless RUN_REAL_LLM_TESTS=true is set.
Requires: ANTHROPIC_API_KEY, OPENAI_API_KEY environment variables.
Cost budget: $0.50 per session — subsequent tests skipped on overage.
"""

from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Callable

import pytest

from grampus.core.models.anthropic import AnthropicClient
from grampus.core.models.openai import OpenAIClient
from grampus.core.types import AgentDefinition, ToolDefinition, ToolParameter

# Module-level skip guard — runs before any test is collected
if not os.environ.get("RUN_REAL_LLM_TESTS"):
    pytest.skip("Set RUN_REAL_LLM_TESTS=true to run", allow_module_level=True)

pytestmark = [pytest.mark.real_llm, pytest.mark.integration]

# Cheapest models — minimise cost per nightly run
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
OPENAI_MODEL = "gpt-4o-mini"

_COSTS_FILE = pathlib.Path("test_costs.json")
_SESSION_BUDGET_USD = 0.50


class _SessionCost:
    total: float = 0.0


_session_cost = _SessionCost()


@pytest.fixture(autouse=True)
def check_session_budget() -> None:  # type: ignore[return]
    """Skip remaining tests when cumulative session cost exceeds $0.50."""
    if _session_cost.total >= _SESSION_BUDGET_USD:
        pytest.skip(f"Session cost budget exceeded (${_session_cost.total:.4f})")
    yield  # type: ignore[misc]


def _record_cost(cost_usd: float, test_name: str) -> None:
    """Accumulate cost and append to test_costs.json."""
    _session_cost.total += cost_usd
    try:
        existing = json.loads(_COSTS_FILE.read_text()) if _COSTS_FILE.exists() else []
    except Exception:
        existing = []
    existing.append(
        {
            "test": test_name,
            "cost_usd": cost_usd,
            "cumulative_usd": _session_cost.total,
        }
    )
    _COSTS_FILE.write_text(json.dumps(existing, indent=2))


@pytest.fixture
def record_cost(request: pytest.FixtureRequest) -> Callable[[float], None]:
    """Return a callable that records cost and accumulates session total."""

    def _do_record(cost_usd: float) -> None:
        _record_cost(cost_usd, request.node.name)

    return _do_record


# ── Client fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def anthropic_client() -> AnthropicClient:
    """Real AnthropicClient — skipped when ANTHROPIC_API_KEY is absent."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return AnthropicClient(api_key=key)


@pytest.fixture(scope="session")
def openai_client() -> OpenAIClient:
    """Real OpenAIClient — skipped when OPENAI_API_KEY is absent."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return OpenAIClient(api_key=key)


# ── Shared tool definition ───────────────────────────────────────────────────


@pytest.fixture(scope="session")
def calculator_tool() -> ToolDefinition:
    """A calculator tool definition for testing tool calling."""
    return ToolDefinition(
        name="calculator",
        description="Evaluate a mathematical expression. Returns the numeric result.",
        parameters=[
            ToolParameter(
                name="expression",
                type="string",
                description="A Python-evaluable math expression, e.g. '7 * 6'",
                required=True,
            )
        ],
        version="1.0.0",
    )


# ── Shared agent definition (minimal, cheap) ────────────────────────────────


@pytest.fixture(scope="session")
def minimal_agent_def() -> AgentDefinition:
    """Minimal cheap AgentDefinition for runner tests."""
    return AgentDefinition(
        name="test-agent",
        model=ANTHROPIC_MODEL,
        system_prompt="You are a helpful assistant. Answer concisely.",
    )
