# Plugin System

Nexus exposes a lifecycle hook system that lets you intercept every significant event in the agent
execution loop — without modifying core framework code. Plugins can observe, mutate, or block
operations at 9 named hook points across `AgentRunner` and `MemoryManager`.

---

## Overview

Two hook tiers with different semantics:

| Tier | Hooks | Dispatch | Can mutate? | Can block? |
|---|---|---|---|---|
| **Pre-hooks** | `pre_llm_call`, `pre_tool_call`, `pre_memory_write` | Sequential, priority order | Yes | Yes (`HookBlockedError`) |
| **Observational** | `on_agent_start`, `on_agent_end`, `post_llm_call`, `post_tool_call`, `post_memory_write`, `on_error` | Concurrent | No | No (failures suppressed) |

Pre-hooks form a **transformation pipeline**: each plugin receives the value returned by the
previous plugin. Raising `HookBlockedError` in a pre-hook cancels the operation — the LLM call,
tool call, or memory write does not execute.

Observational hooks fire concurrently via `asyncio.gather`. A plugin that raises an exception in
an observational hook is logged and ignored — it never crashes the agent.

---

## Quick start

```python
from nexus.plugins import NexusPlugin, PluginManager
from nexus.plugins.types import LLMCallContext

class LoggingPlugin(NexusPlugin):
    async def post_llm_call(self, ctx: LLMCallContext, content, usage) -> None:
        print(f"[{ctx.step}] model={ctx.model} tokens={usage.total_tokens}")

pm = PluginManager(plugins=[LoggingPlugin(name="logger")])

runner = AgentRunner(model_client, tool_executor, plugin_manager=pm)
await runner.run(agent_def, "hello", session_id="s1")
```

Pass `plugin_manager=None` (the default) to disable the plugin system entirely — zero overhead.

---

## Hook reference

### `on_agent_start(ctx: AgentStartContext) -> None`

Fires at the very start of `AgentRunner.run()`, before any LLM call.

```python
@dataclass(frozen=True)
class AgentStartContext:
    agent_id: str
    session_id: str
    user_input: str
    model: str
```

### `pre_llm_call(ctx: LLMCallContext, messages, tools) -> list | None`

Fires before each LLM call. Return a modified messages list to replace the input, or `None`
to pass through unchanged.

```python
@dataclass(frozen=True)
class LLMCallContext:
    agent_id: str
    session_id: str
    model: str
    step: int
```

### `post_llm_call(ctx: LLMCallContext, content: str | None, usage: TokenUsage) -> None`

Fires after each LLM response is received.

### `pre_tool_call(ctx: ToolCallContext, arguments: dict) -> dict | None`

Fires before each tool execution. Return a modified arguments dict to replace the input,
or `None` to pass through unchanged.

```python
@dataclass(frozen=True)
class ToolCallContext:
    agent_id: str
    session_id: str
    tool_name: str
    step: int
```

### `post_tool_call(ctx: ToolResultContext, result: str) -> None`

Fires after each tool returns a result.

```python
@dataclass(frozen=True)
class ToolResultContext:
    agent_id: str
    session_id: str
    tool_name: str
    duration_ms: float
    ok: bool
```

### `on_agent_end(ctx: AgentEndContext) -> None`

Fires after the agent loop completes successfully.

```python
@dataclass(frozen=True)
class AgentEndContext:
    agent_id: str
    session_id: str
    output: str
    steps_taken: int
    total_cost_usd: float
    duration_seconds: float
```

### `on_error(ctx: ErrorContext) -> None`

Fires when the agent loop raises an unhandled exception.

```python
@dataclass(frozen=True)
class ErrorContext:
    agent_id: str
    session_id: str
    error: Exception
    step: int
```

### `pre_memory_write(ctx: MemoryWriteContext, content: str) -> str | None`

Fires before any `MemoryManager.remember()` call. Return a modified content string to replace
the input, or `None` to pass through unchanged.

```python
@dataclass(frozen=True)
class MemoryWriteContext:
    agent_id: str
    session_id: str
    memory_type: str   # "episodic", "semantic", etc.
    source_id: str
```

### `post_memory_write(ctx: MemoryWriteContext, record_id: str | None) -> None`

Fires after the memory write completes.

---

## Pre-hooks: modifying inputs

Pre-hooks thread their return values through a pipeline. Each plugin receives the value
returned by the previous plugin:

```python
class SystemPromptPlugin(NexusPlugin):
    async def pre_llm_call(self, ctx, messages, tools):
        from nexus.core.types import Message, Role
        system = Message(role=Role.SYSTEM, content="Always respond in JSON.")
        return [system] + list(messages)

class PIIRedactPlugin(NexusPlugin):
    async def pre_memory_write(self, ctx, content):
        import re
        return re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", content)
```

Return `None` (or omit the return) to pass the current value unchanged to the next plugin.

---

## Blocking operations

Raise `HookBlockedError` in any pre-hook to cancel the operation:

```python
from nexus.plugins import HookBlockedError, NexusPlugin

class CompliancePlugin(NexusPlugin):
    BLOCKED_TOOLS = {"delete_file", "send_email"}

    async def pre_tool_call(self, ctx, arguments):
        if ctx.tool_name in self.BLOCKED_TOOLS:
            raise HookBlockedError(f"Tool '{ctx.tool_name}' blocked by compliance policy")
        return None
```

- In `AgentRunner`: surfaces as `SafetyError(code="PLUGIN_BLOCKED")`
- In `MemoryManager.remember()`: surfaces as `MemorySecurityError(code="PLUGIN_BLOCKED")`

---

## Priority and ordering

Set `priority` (default `50`) to control pre-hook order. Lower numbers run first:

```python
class EarlyPlugin(NexusPlugin):
    priority = 10   # runs before StandardPlugin

class StandardPlugin(NexusPlugin):
    priority = 50   # default

class LatePlugin(NexusPlugin):
    priority = 90   # runs last
```

Observational hooks (`on_*`, `post_*`) run concurrently regardless of priority.

Temporarily disable a plugin without unregistering it:

```python
plugin.enabled = False
```

---

## Distributing as a package

Third-party plugins are auto-discovered via Python entry points. In your package's
`pyproject.toml`:

```toml
[project.entry-points."nexus.plugins"]
my_plugin = "my_package.plugin:MyPlugin"
```

Then load all registered plugins at runtime:

```python
from nexus.plugins import create_manager_from_entry_points

pm = create_manager_from_entry_points()
runner = AgentRunner(model_client, tool_executor, plugin_manager=pm)
```

`create_manager_from_entry_points()` skips any entry points that fail to load or are not
`NexusPlugin` subclasses — a broken third-party package never crashes your agent.

---

## Example: SOC2 audit plugin

```python
import json
import time
from nexus.plugins import NexusPlugin
from nexus.plugins.types import (
    AgentStartContext, AgentEndContext, LLMCallContext,
    ToolCallContext, ToolResultContext, ErrorContext,
)

class AuditPlugin(NexusPlugin):
    """Append-only audit log for SOC2 compliance."""

    name = "soc2-audit"
    priority = 1   # run first so no event is missed

    def __init__(self, audit_log_path: str, **kwargs):
        super().__init__(**kwargs)
        self._path = audit_log_path

    def _write(self, event: str, data: dict) -> None:
        record = {"ts": time.time(), "event": event, **data}
        with open(self._path, "a") as f:
            f.write(json.dumps(record) + "\n")

    async def on_agent_start(self, ctx: AgentStartContext) -> None:
        self._write("agent_start", {
            "agent": ctx.agent_id, "session": ctx.session_id,
            "model": ctx.model,
        })

    async def on_agent_end(self, ctx: AgentEndContext) -> None:
        self._write("agent_end", {
            "agent": ctx.agent_id, "session": ctx.session_id,
            "steps": ctx.steps_taken, "cost_usd": ctx.total_cost_usd,
        })

    async def post_llm_call(self, ctx: LLMCallContext, content, usage) -> None:
        self._write("llm_call", {
            "agent": ctx.agent_id, "session": ctx.session_id,
            "model": ctx.model, "step": ctx.step,
            "tokens": usage.total_tokens if usage else None,
        })

    async def post_tool_call(self, ctx: ToolResultContext, result) -> None:
        self._write("tool_call", {
            "agent": ctx.agent_id, "session": ctx.session_id,
            "tool": ctx.tool_name, "ok": ctx.ok, "duration_ms": ctx.duration_ms,
        })

    async def on_error(self, ctx: ErrorContext) -> None:
        self._write("agent_error", {
            "agent": ctx.agent_id, "session": ctx.session_id,
            "step": ctx.step, "error": str(ctx.error),
        })
```

Register it:

```python
pm = PluginManager(plugins=[AuditPlugin(name="soc2-audit", audit_log_path="/var/log/nexus/audit.jsonl")])
runner = AgentRunner(model_client, tool_executor, plugin_manager=pm)
```
