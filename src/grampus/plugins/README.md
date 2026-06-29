# `grampus/plugins/` — Lifecycle Hook Plugin System (H49)

This package implements the two-tier hook plugin system described in ADR-024. It enables third-party observability integrations, compliance controls, and cross-cutting concerns to be added to Grampus agents without modifying framework code.

Plugins are optional — when `plugin_manager=None` (the default in `AgentRunner` and `MemoryManager`), there is zero overhead and zero behavioral change.

---

## Key abstractions

| Class | File | Role |
|---|---|---|
| `GrampusPlugin` | `base.py` | Base class — subclass this; override only the hooks you need |
| `PluginManager` | `manager.py` | Runs pre-hooks sequentially (priority order) and post-hooks concurrently |
| `HookBlockedError` | `types.py` | Raised by a plugin to block the operation; surfaces as `SafetyError`/`MemorySecurityError` |
| `LLMCallContext` | `types.py` | Frozen dataclass passed to pre/post LLM hooks |
| `ToolCallContext` | `types.py` | Frozen dataclass passed to pre/post tool hooks |
| `MemoryWriteContext` | `types.py` | Frozen dataclass passed to pre/post memory hooks |
| `AgentStartContext` | `types.py` | Frozen dataclass passed to `on_agent_start` |
| `AgentEndContext` | `types.py` | Frozen dataclass passed to `on_agent_end` |
| `ErrorContext` | `types.py` | Frozen dataclass passed to `on_error` |
| `create_manager_from_entry_points` | `loader.py` | Discovers and loads third-party plugins via Python entry points |

---

## Two-tier hook system

### Tier 1 — Pre-hooks (sequential, can block, can transform)

Pre-hooks run **sequentially in priority order** (lowest `priority` value runs first). Each hook can:
- Return `None` — pass through unchanged
- Return a modified value — the next hook sees the modified value (pipeline pattern)
- Raise `HookBlockedError` — immediately stops the chain and blocks the operation

| Hook method | Called before | Can modify |
|---|---|---|
| `pre_llm_call(ctx)` | Every LLM call | `list[Message]` (the messages list) |
| `pre_tool_call(ctx)` | Every tool execution | `dict[str, Any]` (the tool arguments) |
| `pre_memory_write(ctx)` | Every memory write | `str` (the content) |

### Tier 2 — Observational hooks (concurrent, failures suppressed)

Observational hooks run **concurrently via `asyncio.gather`** after the operation completes. Individual failures are logged and suppressed — a broken plugin never crashes agent execution.

| Hook method | Called after |
|---|---|
| `on_agent_start(ctx)` | `AgentRunner.run()` begins |
| `on_agent_end(ctx)` | `AgentRunner.run()` completes (success or failure) |
| `post_llm_call(ctx)` | LLM call completes |
| `post_tool_call(ctx)` | Tool execution completes |
| `post_memory_write(ctx)` | Memory write completes |
| `on_error(ctx)` | Any exception in the runner |

---

## Writing a plugin

```python
from grampus.plugins.base import GrampusPlugin
from grampus.plugins.types import (
    LLMCallContext, MemoryWriteContext, HookBlockedError,
    LLMCallModification, MemoryWriteModification,
)

class PIIRedactionPlugin(GrampusPlugin):
    name = "pii-redaction"
    priority = 10  # lower = runs earlier in pre-hook chain

    async def pre_memory_write(
        self, ctx: MemoryWriteContext
    ) -> MemoryWriteModification:
        if contains_pii(ctx.content):
            return redact_pii(ctx.content)  # modified content passed to next hook
        return None  # pass through unchanged

    async def post_llm_call(self, ctx: LLMCallContext) -> None:
        await self._audit_log.record(
            agent_id=ctx.agent_id,
            model=ctx.model_id,
            tokens=ctx.token_usage.total_tokens if ctx.token_usage else 0,
        )
```

---

## Using plugins

```python
from grampus.plugins.manager import PluginManager
from mypackage.plugins import PIIRedactionPlugin, DatadogPlugin

plugin_manager = PluginManager(plugins=[
    PIIRedactionPlugin(),
    DatadogPlugin(api_key="..."),
])

runner = AgentRunner(
    ...
    plugin_manager=plugin_manager,
)

memory_manager = MemoryManager(
    ...
    plugin_manager=plugin_manager,  # same instance, shared hooks
)
```

---

## Third-party plugin discovery (entry points)

Third-party packages can publish plugins via Python entry points. No configuration needed beyond installing the package:

```toml
# In the third-party package's pyproject.toml:
[project.entry-points."grampus.plugins"]
my-plugin = "mypkg.plugin:MyPlugin"
```

Load all installed plugins automatically:
```python
from grampus.plugins.loader import create_manager_from_entry_points
plugin_manager = create_manager_from_entry_points()
```

---

## Priority ordering

Lower `priority` integer runs earlier in pre-hooks:

```
priority=10 PIIRedactionPlugin.pre_memory_write → modified content
priority=20 AuditPlugin.pre_memory_write        → sees redacted content
priority=50 (default)                           → other plugins
```

Observational hooks use insertion order (priority has no effect on concurrent dispatch).

---

## Hard invariants

- **`HookBlockedError` is the only exception that propagates from plugins.** All other exceptions in pre-hooks are caught, logged, and the chain continues (the hook is skipped). All exceptions in observational hooks are suppressed entirely.
- **Context objects are `@dataclass(frozen=True)`.** Plugins receive read-only contexts. They cannot modify agent state through the context — only through their return values (for pre-hooks) or side effects (for post-hooks).
- **`plugin_manager=None` means zero overhead.** The `if self._plugins:` guards in `runner.py` and `memory/manager.py` ensure no method calls occur when plugins are not configured.
- **`HookBlockedError` maps to specific error types**: `pre_memory_write` block → `MemorySecurityError(code="PLUGIN_BLOCKED")`, `pre_llm_call` / `pre_tool_call` block → `SafetyError(code="PLUGIN_BLOCKED")`.
- **Plugin `name` must be unique within a `PluginManager`.** `PluginManager` raises `ValueError` on duplicate plugin names at construction.

---

## Dependency map

```
plugins/ depends on:      core/ (errors, logging, types)
plugins/ is imported by:  orchestration/runner.py, memory/manager.py
plugins/ must NOT import from: dapr/, memory/, tools/, safety/, evaluation/
                               (plugins are a cross-cutting concern — keep them dependency-free)
```

---

## ADR references

- **ADR-024** — Lifecycle hook plugin system: full design rationale
