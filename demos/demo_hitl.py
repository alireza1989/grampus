"""
Manual HITL demo — no API key required.

Run:
    uv run python demos/demo_hitl.py

Then in a second terminal, trigger a paused session:
    curl -X POST http://localhost:8000/run \
      -H "Content-Type: application/json" \
      -d '{"input": "do something", "session_id": "test-1"}'

Open http://localhost:8000/ui in a browser to see the session and respond.
"""

from __future__ import annotations

import json
from typing import Any

import uvicorn
from pydantic import BaseModel

from nexus.core.models.base import ModelResponse
from nexus.core.types import (
    AgentDefinition,
    Message,
    Role,
    TokenUsage,
    ToolCall,
)
from nexus.orchestration.runner import AgentRunner
from nexus.server.app import create_app
from nexus.tools.executor import ToolExecutor
from nexus.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# In-memory state store (no Dapr required)
# ---------------------------------------------------------------------------


class _MemoryStateStore:
    """Minimal in-memory state store for local demos."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def save(self, namespace: str, key: str, value: Any) -> None:
        if isinstance(value, BaseModel):
            self._data[key] = value.model_dump_json()
        else:
            self._data[key] = json.dumps(value)

    async def get(self, namespace: str, key: str, model_class: type) -> tuple[Any, str]:
        raw = self._data.get(key)
        if raw is None:
            return None, ""
        return model_class.model_validate_json(raw), "etag-1"

    async def delete(self, namespace: str, key: str) -> None:
        self._data.pop(key, None)


# ---------------------------------------------------------------------------
# Fake LLM: first turn always pauses for human input; resumed turns complete
# ---------------------------------------------------------------------------


def _make_usage(model: str) -> TokenUsage:
    return TokenUsage(input_tokens=10, output_tokens=15, total_tokens=25, cost_usd=0.0, model=model)


class _HumanInputModel:
    """Fake model client — no API key needed."""

    async def complete(
        self, messages: list[Message], model: str, temperature: float
    ) -> ModelResponse:
        last_user = next((m for m in reversed(messages) if m.role == Role.USER), None)
        last_content = (last_user.content or "").lower() if last_user else ""

        # If the user has already been asked once (there's a tool result in messages)
        # and now replies, give a final answer.
        has_tool_result = any(m.role == Role.TOOL for m in messages)
        if has_tool_result:
            return ModelResponse(
                content=f"Thanks for confirming. I'll proceed with: '{last_content}'.",
                tool_calls=[],
                token_usage=_make_usage(model),
                model=model,
                stop_reason="end_turn",
            )

        # First turn: pause and ask the human
        return ModelResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="hitl-1",
                    name="human_input",
                    arguments={"prompt": "Please confirm you want to proceed."},
                )
            ],
            token_usage=_make_usage(model),
            model=model,
            stop_reason="tool_use",
        )

    async def stream(self, messages: list[Message], model: str, temperature: float) -> Any:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


def build_app() -> Any:
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    state_store = _MemoryStateStore()

    runner = AgentRunner(
        _HumanInputModel(),
        executor,
        state_store=state_store,
    )

    agent_def = AgentDefinition(name="demo-agent", model="demo")
    return create_app(runner, agent_def)


if __name__ == "__main__":
    app = build_app()
    print()
    print("  Nexus HITL demo running at   http://localhost:8000")
    print("  Human-in-the-loop UI:        http://localhost:8000/ui")
    print("  API docs:                    http://localhost:8000/docs")
    print()
    print("  Trigger a paused session:")
    print("  curl -X POST http://localhost:8000/run \\")
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"input": "do something", "session_id": "test-1"}\'')
    print()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
