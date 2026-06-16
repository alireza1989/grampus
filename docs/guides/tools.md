# Tools Guide

## What you'll learn

- Register tools with the decorator API
- Configure timeouts, retries, and idempotency
- Connect to MCP-compatible tool servers
- Execute LLM-generated code safely in a Docker sandbox
- Enforce per-agent tool boundaries with `ActionGuard`

---

## Tool registry

### Register with the decorator

```python
from grampus.core.types import ToolParameter
from grampus.tools.registry import ToolRegistry

registry = ToolRegistry()


@registry.tool(
    name="calculate",
    description="Perform a mathematical calculation.",
    parameters=[
        ToolParameter(
            name="expression",
            type="string",
            description="A Python math expression, e.g. '2 + 2' or 'sqrt(16)'",
            required=True,
        ),
    ],
    version="1.0.0",
)
async def calculate(expression: str) -> dict[str, float]:
    import math  # noqa: PLC0415
    result = eval(expression, {"__builtins__": {}}, vars(math))  # noqa: S307
    return {"result": float(result)}
```

### Register programmatically

```python
from grampus.core.types import ToolParameter

async def fetch_weather(city: str, units: str = "celsius") -> dict[str, str]:
    return {"city": city, "temp": "22°C", "condition": "sunny"}

registry.register(
    fetch_weather,
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
```

### Inspect the registry

```python
print(f"Tools registered: {len(registry)}")

for tool in registry.list_all():
    print(f"  {tool.name} v{tool.definition.version}: {tool.description}")

# Get JSON schema for LLM
schemas = registry.to_definitions()
for defn in schemas:
    import json
    print(json.dumps(defn.to_function_schema(), indent=2))
```

### Check for a tool

```python
if "calculate" in registry:
    tool = registry.get("calculate")

# Raises ToolNotFoundError if missing
tool = registry.get_or_raise("calculate")
```

---

## Tool executor

`ToolExecutor` adds timeout, retry, idempotency, and execution records on top of the registry.

```python
from grampus.tools.executor import ToolExecutor

executor = ToolExecutor(
    registry,
    timeout_seconds=30.0,    # abort tool after 30s → ToolTimeoutError
    max_retries=2,           # retry transient errors up to 2 times
    retry_delay_seconds=0.5, # wait 500ms between retries
)
```

### Execute a tool call

```python
from grampus.core.types import ToolCall

tool_call = ToolCall(
    id="call_abc123",
    name="calculate",
    arguments={"expression": "sqrt(144)"},
)

result = await executor.execute(tool_call)
print(f"Output:   {result.output}")      # {"result": 12.0}
print(f"Error:    {result.error}")       # None
print(f"Duration: {result.duration_ms}ms")
```

### Idempotency

The executor caches results by `tool_call_id`. In workflow replay scenarios (e.g., after a crash and restart), calling `execute()` with the same `tool_call_id` returns the cached result without re-running the tool:

```python
result1 = await executor.execute(tool_call)   # runs the tool
result2 = await executor.execute(tool_call)   # returns cached result
assert result1.output == result2.output
```

### Inspect execution records

```python
record = executor.get_record("call_abc123")
print(f"Tool:      {record.tool_name}")
print(f"Args:      {record.arguments}")
print(f"Started:   {record.started_at}")
print(f"Duration:  {record.duration_ms}ms")

all_records = executor.all_records()
print(f"Total tool calls this session: {len(all_records)}")
```

---

## MCP client

The MCP (Model Context Protocol) client lets you connect to any MCP-compatible tool server — filesystem tools, browser automation, databases, and more.

```python
from grampus.tools.mcp_client import MCPClient

# Connect to an MCP server
mcp_client = MCPClient(server_url="http://localhost:3100")

# Discover available tools
tools = await mcp_client.list_tools()
for tool in tools:
    print(f"  MCP tool: {tool.name} — {tool.description}")

# Invoke an MCP tool directly
result = await mcp_client.invoke_tool(
    name="read_file",
    arguments={"path": "/data/report.txt"},
)
print(f"File contents: {result.output}")
```

!!! note "Provenance tagging"
    Results from MCP tools are automatically tagged with `SourceType.EXTERNAL_DATA` (trust=0.3) when stored in memory. This ensures tool results from external servers are treated as lower-trust content.

---

## Sandbox execution

The sandbox runs tool code in an isolated Docker container. Grampus uses sandbox isolation by default for LLM-generated code.

### Code executor (LLM-generated Python)

```python
from grampus.tools.sandbox.code_executor import CodeExecutor
from grampus.tools.sandbox.manager import SandboxManager

sandbox_manager = SandboxManager(
    network_access=False,        # no outbound network
    memory_mb=256,               # 256 MB RAM limit
    cpu_count=1,                 # 1 CPU core
    timeout_seconds=30.0,        # 30s execution timeout
)

executor = CodeExecutor(
    sandbox_manager=sandbox_manager,
    registry=registry,           # inject registered tools into namespace
)

# Execute LLM-generated code safely
code = """
result = calculate(expression="2 ** 10")
print(f"2^10 = {result['result']}")
"""

execution = await executor.execute(code)
print(f"Stdout:   {execution.stdout}")    # "2^10 = 1024.0\n"
print(f"Stderr:   {execution.stderr}")    # ""
print(f"Return:   {execution.return_value}")
print(f"Duration: {execution.duration_ms}ms")
```

### Sandbox configuration

```python
sandbox_manager = SandboxManager(
    network_access=True,          # allow outbound HTTP
    allowed_hosts=["api.example.com"],   # allowlist specific hosts
    filesystem_mounts={
        "/data": "/host/data",    # mount host path read-only
    },
    memory_mb=512,
    cpu_count=2,
    timeout_seconds=60.0,
)
```

!!! warning "Always use the sandbox"
    LLM-generated code execution **must** use the sandbox. Executing arbitrary LLM output in the host process creates severe security vulnerabilities. See [ADR-007](../architecture/decisions.md) for rationale.

---

## Action guard

`ActionGuard` enforces per-agent boundaries: which tools are allowed, how often they can be called, and cost limits.

```python
from grampus.safety.action_guard import ActionGuard, AgentPolicy

policy = AgentPolicy(
    allowed_tools=["web_search", "calculate"],   # explicit allowlist
    denied_tools=[],                             # or use a denylist
    max_tool_calls_per_turn=15,                  # across all tools
    max_consecutive_tool_calls=5,                # before requiring LLM response
    max_cost_per_action_usd=0.01,               # per-tool-call cost cap
)

guard = ActionGuard(policy=policy)

# Check before executing
from grampus.core.types import ToolCall
tool_call = ToolCall(id="x", name="web_search", arguments={"query": "test"})
checked_call, violations = await guard.check(
    tool_call,
    calls_this_turn=3,
    consecutive_calls=2,
)
# Raises SafetyError(code="ACTION_BLOCKED") if denied
```

---

## Full example: registering and executing a tool

```python
import asyncio
from grampus.core.types import ToolCall, ToolParameter
from grampus.tools.executor import ToolExecutor
from grampus.tools.registry import ToolRegistry

registry = ToolRegistry()


@registry.tool(
    name="reverse_string",
    description="Reverse a string.",
    parameters=[
        ToolParameter(name="text", type="string", description="String to reverse", required=True),
    ],
)
async def reverse_string(text: str) -> dict[str, str]:
    return {"reversed": text[::-1]}


async def main() -> None:
    executor = ToolExecutor(registry, timeout_seconds=5.0)

    call = ToolCall(id="call-1", name="reverse_string", arguments={"text": "hello"})
    result = await executor.execute(call)

    print(f"Result:   {result.output}")       # {"reversed": "olleh"}
    print(f"Duration: {result.duration_ms}ms")

    record = executor.get_record("call-1")
    print(f"Recorded: {record.tool_name} at {record.started_at}")


asyncio.run(main())
```

---

## Next steps

- **[Safety guide →](safety.md)** — Validate tool results for injection and PII
- **[Tools API reference →](../reference/tools-api.md)** — Full `ToolRegistry` and `ToolExecutor` reference
- **[Single-agent guide →](single-agent.md)** — See tools wired into a full agent