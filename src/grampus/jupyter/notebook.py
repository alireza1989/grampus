"""GrampusNotebook — Jupyter-friendly façade over AgentRunner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from grampus.core.types import (
    AgentDefinition,
    ExecutionResult,
    StreamEventType,
    TokenUsage,
)
from grampus.jupyter._compat import run_async
from grampus.jupyter.display import (
    _ipython_available,
    render_result,
    render_stream_token,
    render_tool_call,
    render_tool_result,
)


def _display(obj: Any) -> None:
    """Call IPython.display.display when available, otherwise print."""
    if _ipython_available():
        from IPython.display import display

        display(obj)
    else:
        print(obj)


@dataclass
class StreamSummary:
    """Summary returned at the end of a streamed agent execution."""

    output: str
    tool_calls_made: int
    token_usage: TokenUsage | None


class GrampusNotebook:
    """Jupyter-friendly façade over AgentRunner.

    Provides both async (await-able) and sync (nest_asyncio-backed) APIs,
    plus auto-display of rich HTML output.

    Args:
        runner: An AgentRunner instance.
        agent_def: The AgentDefinition to use.
        session_id: Fixed session ID. Auto-generated if None.
        auto_display: When True, automatically call IPython.display() on results.
    """

    def __init__(
        self,
        runner: Any,
        agent_def: AgentDefinition,
        *,
        session_id: str | None = None,
        auto_display: bool = True,
    ) -> None:
        self._runner = runner
        self._agent_def = agent_def
        self._session_id = session_id or f"nb-{uuid4().hex[:8]}"
        self._auto_display = auto_display

    async def run(self, input_text: str, *, session_id: str | None = None) -> ExecutionResult:
        """Run the agent and return ExecutionResult. Awaitable in Jupyter."""
        sid = session_id or self._session_id
        result: ExecutionResult = await self._runner.run(
            self._agent_def, input_text, session_id=sid
        )
        if self._auto_display:
            self.display(result)
        return result

    def run_sync(self, input_text: str, *, session_id: str | None = None) -> ExecutionResult:
        """Sync wrapper using nest_asyncio. For callers that cannot use await."""
        return run_async(self.run(input_text, session_id=session_id))

    async def stream(self, input_text: str, *, session_id: str | None = None) -> StreamSummary:
        """Stream agent execution with live display of tokens and tool events.

        TOKEN events are always rendered via render_stream_token.
        Tool call/result badges are rendered only when auto_display is True.
        """
        sid = session_id or self._session_id
        full_text = ""
        tool_calls_made = 0
        token_usage: TokenUsage | None = None

        async for event in self._runner.stream(self._agent_def, input_text, session_id=sid):
            if event.event_type == StreamEventType.TOKEN and event.chunk:
                render_stream_token(event.chunk.delta)
                full_text += event.chunk.delta
            elif event.event_type == StreamEventType.TOOL_CALL_START and event.tool_call:
                if self._auto_display:
                    _display(render_tool_call(event.tool_call.name, event.tool_call.arguments))
            elif event.event_type == StreamEventType.TOOL_CALL_END and event.tool_call:
                tool_calls_made += 1
                if self._auto_display and event.tool_result:
                    _display(render_tool_result(event.tool_call.name, event.tool_result.output))
            elif event.event_type == StreamEventType.AGENT_END and event.chunk:
                token_usage = event.chunk.token_usage

        return StreamSummary(
            output=full_text,
            tool_calls_made=tool_calls_made,
            token_usage=token_usage,
        )

    def display(self, result: ExecutionResult) -> None:
        """Display an ExecutionResult as rich HTML, or plain text if IPython is absent."""
        if _ipython_available():
            from IPython.display import display

            display(render_result(result, agent_name=self._agent_def.name))
        else:
            print(result.output)

    def display_messages(self) -> None:
        """Display conversation history. Reserved for future implementation."""
        raise NotImplementedError("display_messages is not yet implemented")
