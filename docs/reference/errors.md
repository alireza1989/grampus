# Error Reference

All Nexus exceptions inherit from `NexusError` and carry a machine-readable `code` string and optional `details` dict.

```python
from nexus.core.errors import NexusError

try:
    result = await runner.run(agent, user_input, session_id="s1")
except NexusError as e:
    print(f"Error: {e}")
    print(f"Code:    {e.code}")
    print(f"Details: {e.details}")
```

---

## Hierarchy

```
NexusError
├── ConfigError
├── MemoryError
│   └── MemorySecurityError
├── ToolError
│   ├── ToolNotFoundError
│   ├── ToolValidationError
│   └── ToolTimeoutError
├── OrchestrationError
│   └── BudgetExceededError
├── SafetyError
├── ModelError
└── DaprError
    ├── DaprConnectionError
    ├── ConcurrencyError
    ├── LockAcquisitionError
    └── StateSerializationError
```

---

## NexusError (base)

```python
NexusError(message: str, *, code: str, details: dict | None = None)
```

All exceptions carry:

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable error description |
| `code` | `str` | Machine-readable error code (snake_case) |
| `details` | `dict \| None` | Structured context (IDs, values, limits) |

---

## ConfigError

**Code:** `config.invalid` or `config.missing`

**Raised when:** A required configuration field is missing or has an invalid value.

```python
from nexus.core.errors import ConfigError

# Example details
{
    "field": "model.anthropic_api_key",
    "reason": "required field is not set"
}
```

**How to handle:** Check environment variables (`NEXUS_MODEL__ANTHROPIC_API_KEY`) and `nexus.yaml`.

---

## MemoryError

**Code:** `memory.store_failed`, `memory.retrieve_failed`, `memory.delete_failed`

**Raised when:** A memory read, write, or delete operation fails at the storage layer.

```python
from nexus.core.errors import MemoryError

# Example details
{
    "memory_type": "episodic",
    "operation": "store",
    "agent_id": "research-agent",
    "dapr_error": "connection refused"
}
```

**How to handle:** Check Dapr sidecar health (`http://localhost:3500/v1.0/healthz`) and PostgreSQL connectivity.

---

## MemorySecurityError

**Code:** `memory.security.injection_detected`, `memory.security.rate_limit_exceeded`, `memory.security.validation_failed`

**Raised when:** A memory write is blocked by the security layer (injection detected, rate limit, size anomaly).

```python
from nexus.core.errors import MemorySecurityError

# Example details
{
    "reason": "injection_pattern_detected",
    "pattern": "remember_in_future_sessions",
    "source_type": "TOOL_RESULT",
    "content_preview": "Ignore previous instructions..."
}
```

**How to handle:** Review the content being written. If legitimate, adjust the injection detection level in `safety_policy.yaml`.

---

## ToolError

**Code:** `tool.execution_failed`

**Raised when:** A tool function raises an unretriable exception.

```python
# Example details
{
    "tool_name": "web_search",
    "tool_call_id": "call_abc123",
    "error": "HTTPError: 429 Too Many Requests"
}
```

---

## ToolNotFoundError

**Code:** `tool.not_found`

**Raised when:** `ToolExecutor.execute()` or `ToolRegistry.get_or_raise()` is called with an unregistered tool name.

```python
# Example details
{
    "tool_name": "send_email",
    "registered_tools": ["web_search", "calculate"]
}
```

---

## ToolValidationError

**Code:** `tool.validation_failed`

**Raised when:** A required tool argument is missing or has the wrong type.

```python
# Example details
{
    "tool_name": "get_weather",
    "missing_arguments": ["city"],
    "received_arguments": {"units": "celsius"}
}
```

---

## ToolTimeoutError

**Code:** `tool.timeout`

**Raised when:** A tool execution exceeds `ToolExecutor.timeout_seconds` and all retries are exhausted.

```python
# Example details
{
    "tool_name": "slow_database_query",
    "timeout_seconds": 30.0,
    "attempts": 3
}
```

---

## OrchestrationError

**Code:** `orchestration.max_iterations_exceeded`, `orchestration.no_state_found`, `orchestration.agent_not_waiting`, `orchestration.crew_member_failed`, `orchestration.graph_node_failed`

**Raised when:** The agent loop exceeds `max_iterations` without producing a final answer, or a Crew/Graph operation fails.

```python
# Max iterations example details
{
    "agent_name": "research-agent",
    "max_iterations": 10,
    "last_action": "tool_call: web_search"
}

# Crew failure example details
{
    "failed_member": "critic",
    "error_code": "orchestration.max_iterations_exceeded"
}
```

**How to handle:** Increase `RunnerConfig.max_iterations`, simplify the agent's task, or decompose into a crew.

---

## BudgetExceededError

**Code:** `orchestration.budget_exceeded`

**Raised when:** `AgentDefinition.cost_budget_usd` is set and the accumulated cost exceeds it during a run.

```python
# Example details
{
    "budget_usd": 0.10,
    "accumulated_cost_usd": 0.1023,
    "agent_name": "research-agent",
    "steps_completed": 7
}
```

**How to handle:** Increase the budget, reduce tool calls, or use a cheaper model tier.

---

## SafetyError

**Code:** `INPUT_BLOCKED`, `TOOL_RESULT_BLOCKED`, `ACTION_BLOCKED`, `PII_BLOCKED`

**Raised when:** A safety check blocks a user input, tool result, or tool call.

```python
# Example details
{
    "violation_type": "injection",
    "severity": "critical",
    "pattern": "role_hijacking",
    "blocked_content_preview": "Ignore previous instructions..."
}
```

**How to handle:** Review the blocked content. If it's a false positive, adjust the injection detection level.

---

## ModelError

**Code:** `model.api_error`, `model.rate_limit`, `model.context_length_exceeded`, `model.invalid_response`

**Raised when:** The LLM API returns an error or an unexpected response.

```python
# Example details
{
    "model": "claude-sonnet-4-6",
    "provider": "anthropic",
    "status_code": 429,
    "provider_error": "rate_limit_error"
}
```

**How to handle:** Check your API key, rate limits, and token counts. For context length errors, reduce `WorkingMemory` token limit.

---

## DaprError

**Code:** `dapr.connection_failed`, `dapr.timeout`

**Raised when:** The Dapr sidecar is unreachable.

---

## ConcurrencyError

**Code:** `dapr.concurrency_conflict`

**Raised when:** An optimistic concurrency ETag mismatch occurs during a Dapr state write.

```python
# Example details
{
    "key": "episodic:research-agent:session-42:ep-001",
    "expected_etag": "v3",
    "actual_etag": "v4"
}
```

**How to handle:** Retry the operation with the latest ETag.

---

## LockAcquisitionError

**Code:** `dapr.lock_acquisition_failed`

**Raised when:** A distributed lock cannot be acquired within the timeout (another process holds it).

---

## StateSerializationError

**Code:** `dapr.serialization_failed`

**Raised when:** A Dapr state value cannot be deserialized into the expected Pydantic model.
