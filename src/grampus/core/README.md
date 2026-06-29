# `grampus/core/` — Foundation Layer

This package owns the shared vocabulary of the entire framework: configuration, the error hierarchy, structured logging, all Pydantic data models, and the five LLM provider clients. Every other package imports from here; `core/` imports from nothing else in `grampus/`.

It does **not** own infrastructure (that is `dapr/`), execution logic (that is `orchestration/`), or any domain feature (memory, tools, safety, etc.).

---

## Key abstractions

| Class / Function | File | Role |
|---|---|---|
| `GrampusConfig` | `config.py` | Top-level settings, loaded from env vars (`GRAMPUS_` prefix) or `grampus.yaml` |
| `ModelConfig` | `config.py` | LLM provider keys and defaults; env prefix `GRAMPUS_MODEL__` |
| `MemoryConfig` | `config.py` | Token limits, decay rates, summarization strategy; prefix `GRAMPUS_MEMORY__` |
| `DaprConfig` | `config.py` | Sidecar host/port and store names; prefix `GRAMPUS_DAPR__` |
| `GrampusError` | `errors.py` | Root exception — every error carries `message`, `code`, `details`, `hint` |
| `get_logger(name)` | `logging.py` | Returns a structlog bound logger; **the only way to log in this codebase** |
| `bind_correlation_id()` | `logging.py` | Injects a UUID into all log events for the current async context |
| `Message` | `types.py` | A single conversation message (role, content, tool_calls, tool_results, timestamp) |
| `AgentDefinition` | `types.py` | Immutable blueprint for an agent (model, system_prompt, tools, budget) |
| `AgentState` | `types.py` | Mutable runtime state of an executing agent |
| `ToolDefinition` | `types.py` | Full tool spec; `.to_function_schema()` generates OpenAI/Anthropic-compatible JSON |
| `ExecutionResult` | `types.py` | Final output of one `AgentRunner.run()` call |
| `StreamChunk` / `StreamEvent` | `types.py` | Streaming token and lifecycle events |
| `TokenUsage` | `types.py` | Input/output tokens and `cost_usd` for one model call |
| `ModelClient` | `models/base.py` | ABC: `async complete(…) -> ModelResponse` and `stream(…) -> AsyncIterator[StreamChunk]` |
| `ModelResponse` | `models/base.py` | Unified response from any provider (content, tool_calls, token_usage, stop_reason) |
| `AnthropicClient` | `models/anthropic.py` | Anthropic SDK adapter |
| `OpenAIClient` | `models/openai.py` | OpenAI SDK adapter |
| `GeminiClient` | `models/gemini.py` | Google GenAI adapter |
| `OllamaClient` | `models/ollama.py` | Local Ollama adapter |
| `CohereClient` | `models/cohere.py` | Cohere adapter |

---

## How to use this package

```python
# Config — reads GRAMPUS_ env vars automatically
from grampus.core.config import GrampusConfig
cfg = GrampusConfig()                        # from env / grampus.yaml
cfg = GrampusConfig(_config_file="my.yaml") # explicit YAML

# Logging — always via get_logger, never print() or logging.info()
from grampus.core.logging import get_logger
log = get_logger(__name__)
log.info("thing_happened", agent_id="abc", cost_usd=0.02)

# Types — the shared vocabulary
from grampus.core.types import Message, Role, AgentDefinition, ToolDefinition, ToolParameter

# Errors — subclass GrampusError, always provide code=
from grampus.core.errors import ToolError
raise ToolError("Something broke", code="TOOL_EXECUTE_FAILED", details={"tool": "web_search"})

# Model client — pick a provider, inject via constructor
from grampus.core.models.anthropic import AnthropicClient
client = AnthropicClient(api_key="sk-ant-...")
response = await client.complete(messages=[...], model="claude-haiku-4-5-20251001")
```

---

## Hard invariants

- **Every public type is a Pydantic v2 `BaseModel`.** No raw dicts in the public API. Downstream code relies on `.model_dump()` / `.model_validate()` round-trips being lossless.
- **Every error subclasses `GrampusError` with a `code=` string.** The `code` field is how automated systems identify errors programmatically. Never raise plain `ValueError` or `RuntimeError` from public APIs.
- **Never call `print()` or `logging.info()`.** Use `get_logger(__name__)` everywhere. The logger outputs structured JSON in production and console-friendly text in dev.
- **`ToolDefinition.to_function_schema()` output must match OpenAI/Anthropic tool-calling format exactly.** Both providers consume this directly. Breaking this silently breaks all tool calls across every provider.
- **`temperature` is validated in `AgentDefinition` to `[0.0, 2.0]`.** Do not remove the `@field_validator`.
- **`StreamChunk.is_final=True` on the last chunk, with `token_usage` populated.** All streaming consumers depend on this contract. Intermediate chunks must have `is_final=False` and may have empty `token_usage`.
- **`AgentState.last_event_id`** is set by `AgentRunner` after each event log write. It is used by `CausalTracer` for post-session failure diagnosis; do not remove this field.
- **Config priority order: init kwargs → env vars → YAML → coded defaults.** This order is enforced by `settings_customise_sources`. Do not add additional sources without updating the docstring.

---

## Extension guide

### Adding a new LLM provider

1. Create `src/grampus/core/models/myprovider.py`.
2. Subclass `ModelClient` and implement `async complete(…) -> ModelResponse` and `stream(…) -> AsyncIterator[StreamChunk]`.
3. Convert `list[Message]` to the provider's native format and `list[ToolDefinition]` via `td.to_function_schema()`.
4. Wrap all provider SDK exceptions in `ModelError(code="MODEL_API_ERROR", …)`.
5. Extract `TokenUsage` from the response — **use the API's reported token counts, not tiktoken estimates**.
6. Add the optional dependency to `pyproject.toml` under `[project.optional-dependencies]`.
7. Add tests in `tests/core/test_models.py` using `respx` to mock HTTP.

### Adding a new error type

1. Add the class to `errors.py`, subclassing the nearest semantic parent.
2. Document the `code` strings this error uses in its docstring.
3. Update `tests/core/test_errors.py` to verify `isinstance(NewError(…), GrampusError)` is True.

### Adding a new config section

1. Create a `BaseSettings` subclass with the right `env_prefix`.
2. Add it as a field on `GrampusConfig`.
3. Add tests in `tests/core/test_config.py` for env var loading and YAML loading.

---

## Dependency map

```
core/ depends on:      (nothing in grampus/)
core/ is imported by:  dapr/, memory/, tools/, orchestration/, safety/,
                       observability/, evaluation/, causal/, plugins/,
                       versioning/, server/, cli/
core/ must NOT import from: any other grampus sub-package
```

---

## ADR references

- **ADR-003** — Pydantic v2 for all data models
- **ADR-004** — Async-first architecture (`async complete()`, `stream()`)
- **ADR-008** — OpenTelemetry (`GrampusTracer` uses `TokenUsage` from here)
