# `grampus/tools/` — Tool System

This package owns tool registration, execution, sandboxing, MCP protocol integration, and action boundaries. It is the **only** code that may execute tools or invoke external processes on behalf of agents.

All tool execution goes through `ToolExecutor`. No other code calls tool functions directly.

---

## Key abstractions

| Class / Module | File | Role |
|---|---|---|
| `ToolRegistry` | `registry.py` | Registers tools by decorator; validates definitions; emits JSON schemas for model clients |
| `ToolExecutor` | `executor.py` | Validates args, enforces timeouts + retries, idempotency, traces calls |
| `ActionGuard` | `boundaries.py` | Per-agent allowlists/denylists, rate limiting, cost guards |
| `MCPClient` | `mcp_client.py` | MCP protocol client: discover tool servers, list tools, invoke remotely |
| `SandboxManager` | `sandbox/manager.py` | Orchestrates Docker container lifecycle for sandboxed code execution |
| `DockerSandbox` | `sandbox/docker.py` | Container creation, execution, output capture, cleanup; warm container pool |
| `CodeExecutor` | `sandbox/code_executor.py` | Runs LLM-generated Python in sandbox; injects registered tools as callables |
| `VectorStore` | `vector/base.py` | ABC for pluggable vector store backends (pgvector, Pinecone, Qdrant, Weaviate) |

---

## Tool registration

```python
from grampus.tools.registry import ToolRegistry

registry = ToolRegistry()

@registry.tool(name="web_search", description="Search the web for current information")
async def web_search(query: str, max_results: int = 5) -> dict:
    ...

# The decorator registers the function and infers ToolDefinition from the signature.
# Access the schema for model clients:
definitions = registry.to_definitions()   # list[ToolDefinition]
```

### From YAML agent definition

```yaml
tools:
  - name: web_search
    description: Search the web
    parameters:
      - name: query
        type: string
        required: true
```

---

## Tool execution flow

```
AgentRunner receives tool_calls from LLM response
    │
    ▼
ToolExecutor.execute(tool_call, agent_id=...)
    │
    ├─ Check idempotency cache (self._records[tool_call.id])
    │   → Return cached ToolResult if already executed (workflow replay)
    │
    ├─ ActionGuard.check(agent_id, tool_name)
    │   → ToolError(code="tool.action_blocked") if not allowed
    │
    ├─ Validate required parameters vs ToolDefinition
    │   → ToolValidationError(code="tool.missing_args") if invalid
    │
    ├─ Execute with asyncio.wait_for(timeout=30s)
    │   → Sync functions wrapped via asyncio.to_thread
    │   → ToolTimeoutError(code="tool.timeout") on timeout
    │
    ├─ Retry up to max_retries=2 on unexpected failures
    │   (ToolNotFoundError, ToolValidationError, ToolTimeoutError are NOT retried)
    │
    └─ Return ToolResult(output, duration_ms, error=None)
```

---

## Sandboxed code execution

LLM-generated Python always runs in a Docker container — never on the host process.

```python
from grampus.tools.sandbox.code_executor import CodeExecutor

executor = CodeExecutor(sandbox_manager=SandboxManager())

result = await executor.run(
    code="result = web_search(query='latest news')",
    injected_tools={"web_search": web_search_fn},  # callables from registry
    timeout_seconds=30,
)
# result.stdout, result.stderr, result.return_value, result.ok
```

Container warm pool: the first tool call in a session has ~200ms overhead (container spin-up). Subsequent calls use pre-warmed containers (~10ms overhead).

**Sandbox restrictions (configurable per tool):**
- Filesystem access limited to designated temp mount
- Network disabled by default (opt-in per tool)
- CPU and memory limits enforced
- All output captured; containers destroyed after execution

---

## MCP protocol integration

```python
from grampus.tools.mcp_client import MCPClient

client = MCPClient(server_url="http://my-mcp-server:8080")
await client.connect()

# Discover remote tools — returns list[ToolDefinition]
tools = await client.list_tools()

# Invoke a remote tool
result = await client.invoke("filesystem.read", {"path": "/tmp/data.txt"})
# result is tagged with SourceType.EXTERNAL_DATA (trust_level=0.3) for memory provenance
```

---

## Action boundaries

```python
from grampus.tools.boundaries import ActionGuard

guard = ActionGuard(
    allowed_tools={"web_search", "calculator"},  # allowlist (empty = all allowed)
    denied_tools={"shell_execute"},              # explicit denylist
    max_calls_per_minute=60,
    max_cost_usd=1.0,
)
# Per-agent — each AgentDefinition can have its own ActionGuard.
```

---

## Vector store adapters

`tools/vector/` contains `VectorStore` ABC implementations for pluggable backends:

| Adapter | Backend | Install extra |
|---|---|---|
| `PgVectorStore` | PostgreSQL + pgvector | included in base |
| `PineconeStore` | Pinecone cloud | `[pinecone]` |
| `QdrantStore` | Qdrant | `[qdrant]` |
| `WeaviateStore` | Weaviate | `[weaviate]` |

Swap backends by injecting a different `VectorStore` into `EpisodicMemory` or `SemanticMemory`. The memory stores call `.upsert()`, `.search()`, and `.delete()` — no memory code changes required.

---

## Hard invariants

- **LLM-generated code must ALWAYS run in the Docker sandbox** — never via `exec()` or `eval()` on the host process. The `CodeExecutor` enforces this. `NEVER bypass sandbox for tool execution` is a hard rule in `CLAUDE.md`.
- **`ToolExecutor` is the sole execution path.** No code outside `ToolExecutor` may call tool functions from the registry. This ensures idempotency, tracing, and timeout enforcement are never bypassed.
- **`ToolNotFoundError`, `ToolValidationError`, and `ToolTimeoutError` are never retried.** Only unexpected `Exception` subclasses trigger the retry loop. Retrying validation errors would be pointless; retrying timeouts would double the wait.
- **MCP tool results are tagged `SourceType.EXTERNAL_DATA` (trust=0.3)** for memory provenance. Never override this — external data has the lowest trust level.
- **`ToolRegistry.register()` raises `ToolError(code="tool.duplicate_registration")` on duplicate names.** Tool names are globally unique within a registry instance. Never silently overwrite.

---

## Extension guide

### Adding a new built-in tool

1. Create a function in `src/grampus/tools/library/` (or anywhere — the decorator handles registration).
2. Decorate with `@registry.tool(name=..., description=...)`.
3. Add parameter type hints — `ToolExecutor` validates these against `ToolDefinition`.
4. If the tool is async, use `async def`. If sync, use `def` — `ToolExecutor` wraps sync tools in `asyncio.to_thread`.
5. Add tests in `tests/tools/`.

### Adding a new vector store backend

1. Subclass `VectorStore` in `tools/vector/mybackend.py`.
2. Implement `.upsert(records)`, `.search(embedding, top_k, namespace)`, `.delete(ids)`.
3. Add the optional SDK to `pyproject.toml` under a new extras group.
4. Document the extras group in the README and CONTRIBUTING.md.

---

## Dependency map

```
tools/ depends on:     core/, dapr/ (for sandbox state, MCP result storage)
tools/ is imported by: orchestration/ (AgentRunner calls ToolExecutor)
tools/ must NOT import from: memory/ (circular — memory imports embedding_service,
                             not ToolExecutor), safety/, evaluation/
```

---

## ADR references

- **ADR-007** — Sandbox by default for tool execution (Docker; warm container pool)
- **ADR-009** — Code agents as primary, JSON tool calling as fallback
- **ADR-010** — MCP + A2A protocol support (MCP client in this package)
- **ADR-021** — Document processing tools (optional `[documents]` extras)
- **ADR-022** — Code analysis tools (stdlib AST, subprocess ruff/mypy)
