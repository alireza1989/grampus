# Tools API Reference

## ToolRegistry

Central registry for all tools available to an agent.

::: grampus.tools.registry.ToolRegistry
    options:
      show_source: false
      members: [register, tool, get, get_or_raise, list_all, to_definitions]

---

## ToolExecutor

Executes tool calls with validation, timeout, retry, and idempotency.

::: grampus.tools.executor.ToolExecutor
    options:
      show_source: false
      members: [execute, get_record, all_records]

---

## MCP client

::: grampus.tools.mcp_client.MCPClient
    options:
      show_source: false
      members: [list_tools, invoke_tool, close]

---

## Sandbox

::: grampus.tools.sandbox.manager.SandboxManager
    options:
      show_source: false
      members: [create_sandbox, run, destroy]

::: grampus.tools.sandbox.code_executor.CodeExecutor
    options:
      show_source: false
      members: [execute]

---

## Action guard

::: grampus.tools.boundaries.ActionGuard
    options:
      show_source: false
      members: [check]

---

## Types

### RegisteredTool

```python
@dataclass
class RegisteredTool:
    name: str
    description: str
    definition: ToolDefinition
    fn: Callable[..., Awaitable[Any]]
```

### ToolDefinition

::: grampus.core.types.ToolDefinition
    options:
      show_source: false
      members: [to_function_schema]

### ToolParameter

::: grampus.core.types.ToolParameter
    options:
      show_source: false
      members: []

### ToolCall

::: grampus.core.types.ToolCall
    options:
      show_source: false
      members: []

### ToolResult

::: grampus.core.types.ToolResult
    options:
      show_source: false
      members: []

### ToolExecutionRecord

```python
@dataclass
class ToolExecutionRecord:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: ToolResult
    started_at: datetime
    duration_ms: int
```

---

## to_function_schema() output

`ToolDefinition.to_function_schema()` returns an OpenAI/Anthropic-compatible JSON schema:

```python
from grampus.core.types import ToolDefinition, ToolParameter

defn = ToolDefinition(
    name="get_weather",
    description="Get current weather for a city.",
    parameters=[
        ToolParameter(name="city", type="string", description="City name", required=True),
        ToolParameter(
            name="units",
            type="string",
            description="Temperature units",
            required=False,
            default="celsius",
            enum=["celsius", "fahrenheit"],
        ),
    ],
)

import json
print(json.dumps(defn.to_function_schema(), indent=2))
```

Output:

```json
{
  "name": "get_weather",
  "description": "Get current weather for a city.",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "City name"
      },
      "units": {
        "type": "string",
        "description": "Temperature units",
        "default": "celsius",
        "enum": ["celsius", "fahrenheit"]
      }
    },
    "required": ["city"]
  }
}
```
