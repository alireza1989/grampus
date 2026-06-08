"""
Execution trace viewer demo — no API key required.

Run the server:
    uv run python demos/demo_trace.py

Then in a browser open:
    http://localhost:8000/trace?session=demo-1

Then in a second terminal trigger the agent:
    curl -s -X POST http://localhost:8000/run \\
      -H "Content-Type: application/json" \\
      -d '{"input": "research the weather and compute travel time", "session_id": "demo-1"}'

Watch the LLM → tool → tool → answer sequence appear live in the viewer.
Run it a second time with session=demo-2 etc. to see history replay.
"""

from __future__ import annotations

import asyncio
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
    ToolResult,
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
# Fake LLM: two tool calls then a final answer, with delays so the stream
# is visible in the trace viewer
# ---------------------------------------------------------------------------


def _usage(model: str, step: int) -> TokenUsage:
    return TokenUsage(
        input_tokens=40 + step * 10,
        output_tokens=20 + step * 5,
        total_tokens=60 + step * 15,
        cost_usd=round((60 + step * 15) * 0.000003, 6),
        model=model,
    )


class _MultiToolModel:
    """Fake model: step 1 → call get_weather, step 2 → call calc_travel, step 3 → answer."""

    async def complete(
        self, messages: list[Message], model: str, temperature: float
    ) -> ModelResponse:
        await asyncio.sleep(0.4)  # small delay makes the live stream visible

        tool_results = [m for m in messages if m.role == Role.TOOL]
        step = len(tool_results)

        if step == 0:
            return ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="tc-weather",
                        name="get_weather",
                        arguments={"city": "San Francisco"},
                    )
                ],
                token_usage=_usage(model, 1),
                model=model,
                stop_reason="tool_use",
            )

        if step == 1:
            return ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="tc-travel",
                        name="calc_travel",
                        arguments={"distance_km": 120, "speed_kmh": 90},
                    )
                ],
                token_usage=_usage(model, 2),
                model=model,
                stop_reason="tool_use",
            )

        weather_result = ""
        travel_result = ""
        for msg in messages:
            if msg.role == Role.TOOL and msg.tool_results:
                for tr in msg.tool_results:
                    if tr.tool_call_id == "tc-weather":
                        weather_result = str(tr.output)
                    elif tr.tool_call_id == "tc-travel":
                        travel_result = str(tr.output)

        return ModelResponse(
            content=(
                f"Research complete. Weather: {weather_result}. "
                f"Estimated travel: {travel_result}. "
                "Conditions look suitable for the trip."
            ),
            tool_calls=[],
            token_usage=_usage(model, 3),
            model=model,
            stop_reason="end_turn",
        )

    async def stream(self, messages: list[Message], model: str, temperature: float) -> Any:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Fake tools
# ---------------------------------------------------------------------------


async def _get_weather(city: str) -> str:
    await asyncio.sleep(0.3)
    forecasts = {
        "San Francisco": "16°C, partly cloudy, wind 18 km/h",
        "New York": "22°C, sunny, wind 8 km/h",
    }
    return forecasts.get(city, f"18°C, clear skies (data for {city})")


async def _calc_travel(distance_km: float, speed_kmh: float) -> str:
    await asyncio.sleep(0.2)
    hours = distance_km / speed_kmh
    minutes = int(hours * 60)
    return f"{hours:.1f} h ({minutes} min) at {speed_kmh} km/h over {distance_km} km"


class _FakeToolExecutor(ToolExecutor):
    """Executor that handles our two fake tools without a real sandbox."""

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        import time

        start = time.monotonic()
        try:
            if tool_call.name == "get_weather":
                output = await _get_weather(str(tool_call.arguments.get("city", "unknown")))
            elif tool_call.name == "calc_travel":
                output = await _calc_travel(
                    float(tool_call.arguments.get("distance_km", 100)),
                    float(tool_call.arguments.get("speed_kmh", 80)),
                )
            else:
                output = f"unknown tool: {tool_call.name}"
            duration = int((time.monotonic() - start) * 1000)
            return ToolResult(
                tool_call_id=tool_call.id, output=output, error=None, duration_ms=duration
            )
        except Exception as exc:
            duration = int((time.monotonic() - start) * 1000)
            return ToolResult(
                tool_call_id=tool_call.id, output=None, error=str(exc), duration_ms=duration
            )


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


def build_app() -> Any:
    registry = ToolRegistry()
    executor = _FakeToolExecutor(registry)
    state_store = _MemoryStateStore()

    runner = AgentRunner(
        _MultiToolModel(),
        executor,
        state_store=state_store,
    )

    agent_def = AgentDefinition(name="trace-demo", model="demo")
    return create_app(runner, agent_def)


if __name__ == "__main__":
    app = build_app()
    print()
    print("  Nexus trace demo running at   http://localhost:8000")
    print("  ── Open the trace viewer ──────────────────────────────────────────")
    print("  http://localhost:8000/trace?session=demo-1")
    print()
    print("  ── Then in a second terminal, trigger the agent ───────────────────")
    print("  curl -s -X POST http://localhost:8000/run \\")
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"input": "research weather and travel time", "session_id": "demo-1"}\'')
    print()
    print("  Watch the events stream live: LLM call → tool → tool → answer.")
    print("  Re-run with session-id demo-2, demo-3 … to see history replay.")
    print()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
