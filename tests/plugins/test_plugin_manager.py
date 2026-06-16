"""Tests for the H49 plugin system — Part 1: infrastructure."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from grampus.plugins import (
    AgentEndContext,
    AgentStartContext,
    ErrorContext,
    GrampusPlugin,
    HookBlockedError,
    LLMCallContext,
    MemoryWriteContext,
    PluginManager,
    ToolCallContext,
    ToolResultContext,
    load_entry_point_plugins,
)

# ---------------------------------------------------------------------------
# Shared test contexts (frozen dataclasses, constructed once)
# ---------------------------------------------------------------------------

_LLM_CTX = LLMCallContext(agent_id="a1", session_id="s1", model="claude-3-5-sonnet", step=1)
_TOOL_CTX = ToolCallContext(agent_id="a1", session_id="s1", tool_name="search", step=1)
_TOOL_RESULT_CTX = ToolResultContext(
    agent_id="a1", session_id="s1", tool_name="search", duration_ms=10.0, ok=True
)
_MEM_CTX = MemoryWriteContext(
    agent_id="a1", session_id="s1", memory_type="episodic", source_id="user"
)
_START_CTX = AgentStartContext(
    agent_id="a1", session_id="s1", user_input="hello", model="claude-3-5-sonnet"
)
_END_CTX = AgentEndContext(
    agent_id="a1",
    session_id="s1",
    output="done",
    steps_taken=3,
    total_cost_usd=0.001,
    duration_seconds=1.5,
)
_ERR_CTX = ErrorContext(agent_id="a1", session_id="s1", error=ValueError("oops"), step=1)


# ---------------------------------------------------------------------------
# Reusable tracking plugin
# ---------------------------------------------------------------------------


class TrackingPlugin(GrampusPlugin):
    """Records every hook invocation for assertion."""

    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self.calls: list[tuple[str, object]] = []

    async def on_agent_start(self, ctx: AgentStartContext) -> None:
        self.calls.append(("on_agent_start", ctx))

    async def on_agent_end(self, ctx: AgentEndContext) -> None:
        self.calls.append(("on_agent_end", ctx))

    async def pre_llm_call(self, ctx: LLMCallContext, messages: list, tools: list | None):  # type: ignore[override]
        self.calls.append(("pre_llm_call", messages))
        return None

    async def post_llm_call(self, ctx: LLMCallContext, content: str | None, usage: object) -> None:
        self.calls.append(("post_llm_call", content))

    async def pre_tool_call(self, ctx: ToolCallContext, arguments: dict):  # type: ignore[override]
        self.calls.append(("pre_tool_call", arguments))
        return None

    async def post_tool_call(self, ctx: ToolResultContext, result: object) -> None:
        self.calls.append(("post_tool_call", result))

    async def pre_memory_write(self, ctx: MemoryWriteContext, content: str):  # type: ignore[override]
        self.calls.append(("pre_memory_write", content))
        return None

    async def post_memory_write(self, ctx: MemoryWriteContext, record_id: str | None) -> None:
        self.calls.append(("post_memory_write", record_id))

    async def on_error(self, ctx: ErrorContext) -> None:
        self.calls.append(("on_error", ctx))


# ===========================================================================
# Registration tests
# ===========================================================================


async def test_register_adds_plugin() -> None:
    pm = PluginManager()
    pm.register(TrackingPlugin(name="tracker"))
    assert "tracker" in pm.list_plugins()


async def test_register_duplicate_name_raises_value_error() -> None:
    pm = PluginManager()
    pm.register(TrackingPlugin(name="dup"))
    with pytest.raises(ValueError, match="dup"):
        pm.register(TrackingPlugin(name="dup"))


async def test_unregister_removes_plugin() -> None:
    pm = PluginManager()
    pm.register(TrackingPlugin(name="removeme"))
    pm.unregister("removeme")
    assert "removeme" not in pm.list_plugins()


async def test_unregister_missing_name_is_silent() -> None:
    pm = PluginManager()
    pm.unregister("nonexistent")  # must not raise


async def test_list_plugins_returns_names_in_priority_order() -> None:
    pm = PluginManager()
    pm.register(TrackingPlugin(name="low", priority=90))
    pm.register(TrackingPlugin(name="high", priority=10))
    assert pm.list_plugins() == ["high", "low"]


async def test_disabled_plugin_not_called() -> None:
    pm = PluginManager()
    p = TrackingPlugin(name="disabled", enabled=False)
    pm.register(p)
    await pm.call_on_agent_start(_START_CTX)
    assert len(p.calls) == 0


# ===========================================================================
# Pre-hook sequential threading tests
# ===========================================================================


async def test_call_pre_llm_returns_messages_unchanged_when_plugin_returns_none() -> None:
    pm = PluginManager(plugins=[TrackingPlugin(name="t")])
    msgs: list = [{"role": "user", "content": "hi"}]
    result = await pm.call_pre_llm(_LLM_CTX, msgs, None)
    assert result is msgs


async def test_call_pre_llm_returns_modified_messages_when_plugin_returns_list() -> None:
    class ModPlugin(GrampusPlugin):
        async def pre_llm_call(self, ctx, messages, tools):  # type: ignore[override]
            return [{"role": "system", "content": "injected"}] + messages

    pm = PluginManager(plugins=[ModPlugin(name="mod")])
    msgs: list = [{"role": "user", "content": "hi"}]
    result = await pm.call_pre_llm(_LLM_CTX, msgs, None)
    assert result[0]["content"] == "injected"
    assert result[1] == msgs[0]


async def test_call_pre_llm_chains_through_multiple_plugins() -> None:
    class PluginA(GrampusPlugin):
        async def pre_llm_call(self, ctx, messages, tools):  # type: ignore[override]
            return list(messages) + ["from_A"]

    class PluginB(GrampusPlugin):
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            super().__init__(**kwargs)
            self.received: list = []

        async def pre_llm_call(self, ctx, messages, tools):  # type: ignore[override]
            self.received = list(messages)
            return list(messages) + ["from_B"]

    a = PluginA(name="a", priority=10)
    b = PluginB(name="b", priority=20)
    pm = PluginManager(plugins=[a, b])
    result = await pm.call_pre_llm(_LLM_CTX, [], None)
    assert "from_A" in b.received  # B saw A's output
    assert result == ["from_A", "from_B"]


async def test_call_pre_llm_hook_blocked_error_propagates() -> None:
    class BlockPlugin(GrampusPlugin):
        async def pre_llm_call(self, ctx, messages, tools):  # type: ignore[override]
            raise HookBlockedError("blocked by compliance policy")

    pm = PluginManager(plugins=[BlockPlugin(name="blocker")])
    with pytest.raises(HookBlockedError):
        await pm.call_pre_llm(_LLM_CTX, [], None)


async def test_call_pre_llm_non_blocked_exception_suppressed_continues_to_next_plugin() -> None:
    class ErrorPlugin(GrampusPlugin):
        async def pre_llm_call(self, ctx, messages, tools):  # type: ignore[override]
            raise ValueError("unexpected failure")

    class ContinuePlugin(GrampusPlugin):
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            super().__init__(**kwargs)
            self.called = False

        async def pre_llm_call(self, ctx, messages, tools):  # type: ignore[override]
            self.called = True
            return None

    err_p = ErrorPlugin(name="err", priority=10)
    cont_p = ContinuePlugin(name="cont", priority=20)
    pm = PluginManager(plugins=[err_p, cont_p])
    await pm.call_pre_llm(_LLM_CTX, [], None)  # must not raise
    assert cont_p.called is True


async def test_call_pre_tool_returns_arguments_unchanged_when_none() -> None:
    pm = PluginManager(plugins=[TrackingPlugin(name="t")])
    args = {"query": "foo"}
    result = await pm.call_pre_tool(_TOOL_CTX, args)
    assert result is args


async def test_call_pre_tool_returns_modified_arguments() -> None:
    class ArgPlugin(GrampusPlugin):
        async def pre_tool_call(self, ctx, arguments):  # type: ignore[override]
            return {**arguments, "extra": "added"}

    pm = PluginManager(plugins=[ArgPlugin(name="arg")])
    result = await pm.call_pre_tool(_TOOL_CTX, {"query": "foo"})
    assert result == {"query": "foo", "extra": "added"}


async def test_call_pre_memory_write_returns_content_unchanged() -> None:
    pm = PluginManager(plugins=[TrackingPlugin(name="t")])
    result = await pm.call_pre_memory_write(_MEM_CTX, "original content")
    assert result == "original content"


async def test_call_pre_memory_write_returns_modified_content() -> None:
    class RedactPlugin(GrampusPlugin):
        async def pre_memory_write(self, ctx, content):  # type: ignore[override]
            return content.replace("secret", "[REDACTED]")

    pm = PluginManager(plugins=[RedactPlugin(name="redact")])
    result = await pm.call_pre_memory_write(_MEM_CTX, "contains secret data")
    assert result == "contains [REDACTED] data"


# ===========================================================================
# Priority ordering tests
# ===========================================================================


async def test_pre_hooks_called_in_priority_order() -> None:
    """Lower priority number runs first; the second plugin sees the first's output."""

    class FirstPlugin(GrampusPlugin):
        async def pre_llm_call(self, ctx, messages, tools):  # type: ignore[override]
            return list(messages) + ["first"]

    class SecondPlugin(GrampusPlugin):
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            super().__init__(**kwargs)
            self.saw_first = False

        async def pre_llm_call(self, ctx, messages, tools):  # type: ignore[override]
            self.saw_first = "first" in messages
            return list(messages) + ["second"]

    first = FirstPlugin(name="first", priority=10)
    second = SecondPlugin(name="second", priority=20)
    # Register out of insertion order to confirm sort is by priority, not insertion
    pm = PluginManager(plugins=[second, first])
    result = await pm.call_pre_llm(_LLM_CTX, [], None)
    assert second.saw_first is True
    assert result == ["first", "second"]


# ===========================================================================
# Observational hook concurrency tests
# ===========================================================================


async def test_observational_hooks_called_for_all_plugins() -> None:
    p1 = TrackingPlugin(name="p1")
    p2 = TrackingPlugin(name="p2")
    pm = PluginManager(plugins=[p1, p2])
    await pm.call_on_agent_start(_START_CTX)
    assert any(c[0] == "on_agent_start" for c in p1.calls)
    assert any(c[0] == "on_agent_start" for c in p2.calls)


async def test_observational_hook_failure_does_not_propagate() -> None:
    class BrokenPlugin(GrampusPlugin):
        async def on_agent_end(self, ctx: AgentEndContext) -> None:
            raise RuntimeError("plugin crashed hard")

    pm = PluginManager(plugins=[BrokenPlugin(name="broken")])
    await pm.call_on_agent_end(_END_CTX)  # must not raise


async def test_empty_plugin_list_all_hooks_are_no_ops() -> None:
    pm = PluginManager()
    await pm.call_on_agent_start(_START_CTX)
    await pm.call_on_agent_end(_END_CTX)
    await pm.call_post_llm(_LLM_CTX, "response", None)
    await pm.call_post_tool(_TOOL_RESULT_CTX, {"ok": True})
    await pm.call_post_memory_write(_MEM_CTX, "record-123")
    await pm.call_on_error(_ERR_CTX)
    msgs = await pm.call_pre_llm(_LLM_CTX, [], None)
    assert msgs == []
    args = await pm.call_pre_tool(_TOOL_CTX, {"x": 1})
    assert args == {"x": 1}
    content = await pm.call_pre_memory_write(_MEM_CTX, "hello")
    assert content == "hello"


# ===========================================================================
# Loader tests
# ===========================================================================


async def test_load_entry_point_plugins_returns_empty_when_no_plugins_registered() -> None:
    # Real call — "grampus.plugins" entry-point group is empty in the test environment
    plugins = load_entry_point_plugins()
    assert isinstance(plugins, list)
    assert all(isinstance(p, GrampusPlugin) for p in plugins)


async def test_load_entry_point_plugins_skips_bad_entry_point() -> None:
    bad_ep = MagicMock()
    bad_ep.name = "bad_plugin"
    bad_ep.load.side_effect = ImportError("module not found")

    with patch("importlib.metadata.entry_points", return_value=[bad_ep]):
        plugins = load_entry_point_plugins()

    assert plugins == []


async def test_load_entry_point_plugins_skips_non_grampus_plugin_class() -> None:
    class NotAPlugin:
        pass

    good_ep = MagicMock()
    good_ep.name = "not_a_plugin"
    good_ep.load.return_value = NotAPlugin

    with patch("importlib.metadata.entry_points", return_value=[good_ep]):
        plugins = load_entry_point_plugins()

    assert plugins == []
