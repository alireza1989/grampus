"""Integration tests for plugin hooks wired into AgentRunner and MemoryManager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from grampus.core.errors import MemorySecurityError, SafetyError
from grampus.core.models.base import ModelResponse
from grampus.core.types import AgentDefinition, Message, Role, TokenUsage, ToolCall, ToolResult
from grampus.memory.manager import MemoryManager
from grampus.orchestration.runner import AgentRunner
from grampus.plugins import GrampusPlugin, HookBlockedError, PluginManager
from grampus.plugins.types import AgentStartContext

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _agent_def(name: str = "test-agent") -> AgentDefinition:
    return AgentDefinition(name=name, model="test-model")


def _token_usage() -> TokenUsage:
    return TokenUsage(
        input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.001, model="test-model"
    )


def _model_response(
    content: str | None = "done",
    tool_calls: list[ToolCall] | None = None,
) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tool_calls or [],
        token_usage=_token_usage(),
        model="test-model",
        stop_reason="end_turn",
    )


def _tool_result(call_id: str = "tc-1") -> ToolResult:
    return ToolResult(tool_call_id=call_id, output="result", error=None, duration_ms=5)


def _make_runner(
    plugin_manager: PluginManager | None = None,
    model_response: ModelResponse | None = None,
) -> tuple[AgentRunner, AsyncMock, AsyncMock]:
    mc = AsyncMock()
    mc.complete = AsyncMock(return_value=model_response or _model_response())
    te = AsyncMock()
    te.execute = AsyncMock(return_value=_tool_result())
    runner = AgentRunner(mc, te, plugin_manager=plugin_manager)
    return runner, mc, te


def _make_mm(plugin_manager: PluginManager | None = None) -> tuple[MemoryManager, AsyncMock]:
    episodic = AsyncMock()
    episodic.store = AsyncMock()
    mm = MemoryManager(
        working_memory=MagicMock(),
        episodic_memory=episodic,
        semantic_memory=AsyncMock(),
        procedural_memory=AsyncMock(),
        episodic_retriever=AsyncMock(),
        semantic_retriever=AsyncMock(),
        consolidation_pipeline=AsyncMock(),
        agent_id="test-agent",
        plugin_manager=plugin_manager,
    )
    return mm, episodic


# ---------------------------------------------------------------------------
# SpyPlugin — records every hook call
# ---------------------------------------------------------------------------


class SpyPlugin(GrampusPlugin):
    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self.calls: list[tuple[str, object]] = []

    async def on_agent_start(self, ctx: AgentStartContext) -> None:
        self.calls.append(("on_agent_start", ctx))

    async def on_agent_end(self, ctx):  # type: ignore[override]
        self.calls.append(("on_agent_end", ctx))

    async def pre_llm_call(self, ctx, messages, tools):  # type: ignore[override]
        self.calls.append(("pre_llm_call", list(messages)))
        return None

    async def post_llm_call(self, ctx, content, usage):  # type: ignore[override]
        self.calls.append(("post_llm_call", content))

    async def pre_tool_call(self, ctx, arguments):  # type: ignore[override]
        self.calls.append(("pre_tool_call", dict(arguments)))
        return None

    async def post_tool_call(self, ctx, result):  # type: ignore[override]
        self.calls.append(("post_tool_call", result))

    async def pre_memory_write(self, ctx, content):  # type: ignore[override]
        self.calls.append(("pre_memory_write", content))
        return None

    async def post_memory_write(self, ctx, record_id):  # type: ignore[override]
        self.calls.append(("post_memory_write", record_id))

    async def on_error(self, ctx):  # type: ignore[override]
        self.calls.append(("on_error", ctx))


# ===========================================================================
# AgentRunner hook tests
# ===========================================================================


async def test_on_agent_start_called_at_run_start() -> None:
    spy = SpyPlugin(name="spy")
    pm = PluginManager(plugins=[spy])
    runner, _, _ = _make_runner(plugin_manager=pm)
    await runner.run(_agent_def(), "hello", session_id="s1")
    hook_names = [c[0] for c in spy.calls]
    assert "on_agent_start" in hook_names
    assert hook_names.index("on_agent_start") == 0


async def test_pre_llm_call_can_modify_messages() -> None:
    class PrependPlugin(GrampusPlugin):
        async def pre_llm_call(self, ctx, messages, tools):  # type: ignore[override]
            system = Message(role=Role.SYSTEM, content="prepended_by_plugin")
            return [system] + list(messages)

    pm = PluginManager(plugins=[PrependPlugin(name="prepend")])
    runner, mc, _ = _make_runner(plugin_manager=pm)
    await runner.run(_agent_def(), "hello", session_id="s1")

    call_messages = mc.complete.call_args.kwargs["messages"]
    assert call_messages[0].content == "prepended_by_plugin"
    assert call_messages[0].role == Role.SYSTEM


async def test_pre_llm_call_blocked_raises_safety_error() -> None:
    class BlockPlugin(GrampusPlugin):
        async def pre_llm_call(self, ctx, messages, tools):  # type: ignore[override]
            raise HookBlockedError("compliance policy violation")

    pm = PluginManager(plugins=[BlockPlugin(name="blocker")])
    runner, _, _ = _make_runner(plugin_manager=pm)
    with pytest.raises(SafetyError) as exc_info:
        await runner.run(_agent_def(), "hello", session_id="s1")
    assert exc_info.value.code == "PLUGIN_BLOCKED"


async def test_post_llm_call_called_after_response() -> None:
    spy = SpyPlugin(name="spy")
    pm = PluginManager(plugins=[spy])
    runner, _, _ = _make_runner(plugin_manager=pm, model_response=_model_response(content="answer"))
    await runner.run(_agent_def(), "hello", session_id="s1")
    post_llm_calls = [c for c in spy.calls if c[0] == "post_llm_call"]
    assert len(post_llm_calls) >= 1
    assert post_llm_calls[0][1] == "answer"


async def test_pre_tool_call_can_modify_arguments() -> None:
    class ArgPlugin(GrampusPlugin):
        async def pre_tool_call(self, ctx, arguments):  # type: ignore[override]
            return {**arguments, "extra": "added"}

    tc = ToolCall(id="tc-1", name="my_tool", arguments={"x": 1})
    mc = AsyncMock()
    mc.complete = AsyncMock(
        side_effect=[
            _model_response(tool_calls=[tc]),
            _model_response(content="done"),
        ]
    )
    te = AsyncMock()
    te.execute = AsyncMock(return_value=_tool_result())

    pm = PluginManager(plugins=[ArgPlugin(name="arg")])
    runner = AgentRunner(mc, te, plugin_manager=pm)
    await runner.run(_agent_def(), "use tool", session_id="s1")

    te.execute.assert_called_once()
    exec_tc = te.execute.call_args.args[0]
    assert exec_tc.arguments == {"x": 1, "extra": "added"}


async def test_pre_tool_call_blocked_raises_safety_error() -> None:
    class BlockPlugin(GrampusPlugin):
        async def pre_tool_call(self, ctx, arguments):  # type: ignore[override]
            raise HookBlockedError("tool blocked by policy")

    tc = ToolCall(id="tc-1", name="my_tool", arguments={})
    mc = AsyncMock()
    mc.complete = AsyncMock(return_value=_model_response(tool_calls=[tc]))
    te = AsyncMock()
    te.execute = AsyncMock(return_value=_tool_result())

    pm = PluginManager(plugins=[BlockPlugin(name="blocker")])
    runner = AgentRunner(mc, te, plugin_manager=pm)
    with pytest.raises(SafetyError) as exc_info:
        await runner.run(_agent_def(), "use tool", session_id="s1")
    assert exc_info.value.code == "PLUGIN_BLOCKED"


async def test_post_tool_call_called_with_result() -> None:
    spy = SpyPlugin(name="spy")
    tc = ToolCall(id="tc-1", name="my_tool", arguments={})
    mc = AsyncMock()
    mc.complete = AsyncMock(
        side_effect=[
            _model_response(tool_calls=[tc]),
            _model_response(content="done"),
        ]
    )
    te = AsyncMock()
    te.execute = AsyncMock(return_value=_tool_result())

    pm = PluginManager(plugins=[spy])
    runner = AgentRunner(mc, te, plugin_manager=pm)
    await runner.run(_agent_def(), "use tool", session_id="s1")

    post_tool_calls = [c for c in spy.calls if c[0] == "post_tool_call"]
    assert len(post_tool_calls) >= 1
    assert post_tool_calls[0][1] == "result"


async def test_on_agent_end_called_with_stats() -> None:
    spy = SpyPlugin(name="spy")
    pm = PluginManager(plugins=[spy])
    runner, _, _ = _make_runner(plugin_manager=pm)
    await runner.run(_agent_def(), "hello", session_id="s1")

    end_calls = [c for c in spy.calls if c[0] == "on_agent_end"]
    assert len(end_calls) == 1
    ctx = end_calls[0][1]
    assert ctx.steps_taken >= 1
    assert ctx.duration_seconds >= 0.0


async def test_on_error_called_when_runner_raises() -> None:
    spy = SpyPlugin(name="spy")
    mc = AsyncMock()
    mc.complete = AsyncMock(side_effect=RuntimeError("model blew up"))
    te = AsyncMock()

    pm = PluginManager(plugins=[spy])
    runner = AgentRunner(mc, te, plugin_manager=pm)
    with pytest.raises(RuntimeError):
        await runner.run(_agent_def(), "hello", session_id="s1")

    error_calls = [c for c in spy.calls if c[0] == "on_error"]
    assert len(error_calls) == 1
    assert isinstance(error_calls[0][1].error, RuntimeError)


async def test_plugin_manager_none_runner_unchanged() -> None:
    runner, mc, _ = _make_runner(plugin_manager=None)
    result = await runner.run(_agent_def(), "hello", session_id="s1")
    assert result.output == "done"
    mc.complete.assert_called_once()


# ===========================================================================
# MemoryManager hook tests
# ===========================================================================


async def test_pre_memory_write_can_modify_content() -> None:
    class RedactPlugin(GrampusPlugin):
        async def pre_memory_write(self, ctx, content):  # type: ignore[override]
            return content.replace("secret", "[REDACTED]")

    pm = PluginManager(plugins=[RedactPlugin(name="redact")])
    mm, episodic = _make_mm(plugin_manager=pm)

    await mm.remember("contains secret information", session_id="s1")

    episodic.store.assert_called_once()
    stored = episodic.store.call_args.args[0]
    assert "secret" not in stored
    assert "[REDACTED]" in stored


async def test_pre_memory_write_blocked_raises_memory_security_error() -> None:
    class BlockPlugin(GrampusPlugin):
        async def pre_memory_write(self, ctx, content):  # type: ignore[override]
            raise HookBlockedError("memory write blocked by HIPAA plugin")

    pm = PluginManager(plugins=[BlockPlugin(name="blocker")])
    mm, episodic = _make_mm(plugin_manager=pm)

    with pytest.raises(MemorySecurityError) as exc_info:
        await mm.remember("some content", session_id="s1")

    assert exc_info.value.code == "PLUGIN_BLOCKED"
    episodic.store.assert_not_called()


async def test_post_memory_write_called_after_store() -> None:
    spy = SpyPlugin(name="spy")
    pm = PluginManager(plugins=[spy])
    mm, _ = _make_mm(plugin_manager=pm)

    await mm.remember("remember this", session_id="s1")

    post_calls = [c for c in spy.calls if c[0] == "post_memory_write"]
    assert len(post_calls) == 1
