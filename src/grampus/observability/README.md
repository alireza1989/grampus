# `grampus/observability/` — Observability

This package provides three layers of observability: agent OTEL tracing (`GrampusTracer`), Prometheus-compatible metrics (`GrampusMetrics`), behavior monitoring (`BehaviorMonitor`), structured event log (`EventLog`), cost alerting (`AlertManager`), and optional Phoenix/Arize integration for LLM-specific observability.

All three layers are opt-in via `AgentRunner` parameters. With all at `None`, the runner runs with zero observability overhead.

---

## Key abstractions

| Class | File | Role |
|---|---|---|
| `GrampusTracer` | `tracer.py` | OTEL spans for all 6 agent event types; `otlp_endpoint=None` uses NoOpProvider |
| `GrampusMetrics` | `metrics.py` | Prometheus counters, gauges, histograms exposed at `/metrics` |
| `BehaviorMonitor` | `behavior.py` | Tracks per-agent patterns; detects cost spikes, tool usage shifts |
| `EventLog` | `events.py` | Append-only structured event log; Dapr-backed or in-memory |
| `AlertManager` | `alerts.py` | Cost and behavior alert rules with configurable notification channels |
| `NotificationChannel` | `notification.py` | Alert delivery: webhook, Slack, email |
| `PhoenixTracer` | `phoenix.py` | Optional Arize Phoenix integration for LLM-native observability |

---

## OTEL tracing

### Span types

| Span name | When | Key attributes |
|---|---|---|
| `agent.run` | Start of `AgentRunner.run()` | `agent_id`, `session_id`, `model` |
| `agent.llm_call` | Each model call | `model`, `input_tokens`, `output_tokens`, `cost_usd`, `stop_reason` |
| `agent.tool_call` | Each tool execution | `tool_name`, `duration_ms`, `ok` |
| `agent.memory_read` | Each `MemoryManager.recall()` | `query_hash`, `records_returned`, `memory_types` |
| `agent.memory_write` | Each `MemoryManager.remember()` | `source_type`, `trust_level`, `memory_type` |
| `agent.decision` | Conditional branches, uncertainty escalation | `decision_type`, `reason` |

### Usage

```python
from grampus.observability.tracer import GrampusTracer

tracer = GrampusTracer(
    service_name="grampus-agent",
    otlp_endpoint="http://localhost:4318",  # Jaeger, Honeycomb, etc.
    agent_id="agent-1",
    session_id="sess-abc",
)

# All span methods are context managers:
with tracer.agent_run(agent_id="agent-1", session_id="sess-abc", model="claude-haiku-4-5-20251001"):
    # everything inside gets a child span under agent.run
    with tracer.llm_call(model="claude-haiku-4-5-20251001"):
        response = await client.complete(...)
        tracer.record_llm_call(response.token_usage, stop_reason=response.stop_reason)

    with tracer.tool_call(tool_name="web_search"):
        result = await executor.execute(tool_call)
```

`otlp_endpoint=None` (the default) uses `NoOpTracerProvider` — zero network calls, zero overhead.

---

## Prometheus metrics

```python
from grampus.observability.metrics import GrampusMetrics

metrics = GrampusMetrics()
# Expose via FastAPI:
from fastapi.responses import Response
@app.get("/metrics")
async def prometheus_metrics():
    return Response(content=metrics.generate(), media_type="text/plain")
```

| Metric | Type | Description |
|---|---|---|
| `grampus_tokens_total` | Counter | Total tokens by model and agent |
| `grampus_cost_usd_total` | Counter | Total cost by model and agent |
| `grampus_tool_calls_total` | Counter | Tool calls by tool name and status |
| `grampus_errors_total` | Counter | Errors by type |
| `grampus_active_agents` | Gauge | Currently running agent sessions |
| `grampus_llm_latency_seconds` | Histogram | LLM call duration by model |
| `grampus_tool_latency_seconds` | Histogram | Tool execution duration |

---

## Event log

The `EventLog` is the append-only record of every agent action. It is the source of truth for audit trails, causal analysis (Phase F4), and compliance reporting.

```python
from grampus.observability.events import EventLog, EventType

# Open (or resume) a session's event log
event_log = await EventLog.open(
    state_store=dapr_store,
    agent_id="agent-1",
    session_id="sess-abc",
)

# Emit events (AgentRunner calls these automatically)
await event_log.append(EventType.LLM_CALL, data={...})
await event_log.append(EventType.TOOL_CALL, data={...})

# Retrieve for analysis or replay
events = await event_log.get_events_for_session(session_id="sess-abc", agent_id="agent-1")
```

**Event types (14 total):** `AGENT_START`, `AGENT_END`, `AGENT_ERROR`, `LLM_CALL`, `TOOL_CALL`, `TOOL_RESULT`, `MEMORY_READ`, `MEMORY_WRITE`, `MEMORY_SUMMARIZE`, `SAFETY_CHECK`, `BUDGET_CHECK`, `UNCERTAINTY_ESTIMATE`, `PLAN_CREATED`, `PLAN_SUBGOAL_COMPLETE`

**Storage:** each event is an `AgentEvent(frozen=True)` with a monotonic `sequence_number`. Key pattern: `{session_id}:{sequence_number}`. Events are never modified or deleted after writing.

---

## Behavior monitoring

`BehaviorMonitor` consumes events via Dapr pub/sub and detects anomalies by comparing current patterns to 7-day baselines:

- **Cost spike** — session cost exceeds `baseline_avg * cost_spike_factor`
- **Tool usage shift** — a tool is called more than 3× its baseline rate in this session
- **Memory access shift** — retrieval patterns diverge significantly from baseline

When an anomaly is detected, `AlertManager.fire()` is called, which dispatches to configured `NotificationChannel`s (webhook, Slack, email).

---

## Hard invariants

- **`EventLog.open()` is the correct entry point, not `EventLog()`** directly. `open()` initializes the sequence counter from Dapr (for session resumption). Direct `__init__` resets to 0 and would duplicate sequence numbers.
- **Events are `AgentEvent(frozen=True)` — never modified after creation.** This is the immutable audit trail requirement from ADR-005. There is no `update_event()` method.
- **`GrampusTracer.record_llm_call()` annotates the current active span**, not a new span. Call it inside an existing `tracer.llm_call()` context manager, not standalone.
- **`otlp_endpoint=None` (default) produces zero network overhead.** Never make OTLP the default in tests — it will try to connect to a non-existent endpoint.
- **Prometheus metrics are not pre-registered globally.** Each `GrampusMetrics()` instance has its own registry. Inject the same instance into all components that should share metrics.

---

## Dependency map

```
observability/ depends on:      core/, dapr/ (EventLog uses DaprStateStore)
observability/ is imported by:  orchestration/runner.py, cli/
observability/ must NOT import from: memory/ (circular via event_log → MemoryManager),
                                     tools/, safety/, evaluation/
```

---

## ADR references

- **ADR-005** — Event sourcing; `EventLog` is the append-only action record
- **ADR-008** — OpenTelemetry for all observability; 6 custom span types
