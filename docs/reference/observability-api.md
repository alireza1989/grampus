# Observability API Reference

## GrampusTracer

Wraps the OpenTelemetry SDK with agent-specific span types.

::: grampus.observability.tracer.GrampusTracer
    options:
      show_source: false
      members: [llm_call, tool_call, memory_read, memory_write, record_llm_call]

### Span context manager

```python
tracer = GrampusTracer(service_name="my-agent", otel_endpoint="http://localhost:4317")

with tracer.span("agent.custom_step", attributes={"step.name": "validate"}):
    do_work()

# Async
async with tracer.async_span("agent.llm_call", attributes={"model": "claude-sonnet-4-6"}):
    response = await llm.complete(messages)
```

### Span types and attributes

| Span type | Key attributes |
|-----------|---------------|
| `agent.run` | `agent.name`, `agent.model`, `session.id`, `agent.status` |
| `agent.llm_call` | `model`, `input_tokens`, `output_tokens`, `cost_usd`, `stop_reason` |
| `agent.tool_call` | `tool.name`, `tool.duration_ms`, `tool.success`, `tool.call_id` |
| `agent.memory_read` | `memory.type`, `memory.query`, `memory.results_count` |
| `agent.memory_write` | `memory.type`, `memory.source_type`, `memory.trust_level` |
| `agent.decision` | `agent.step`, `decision.action` |

---

## GrampusMetrics

Prometheus-compatible metrics endpoint.

::: grampus.observability.metrics.GrampusMetrics
    options:
      show_source: false
      members: [record_llm_call, record_tool_call, record_error, set_active_agents, to_prometheus_text, snapshot]

### Counter metrics

| Metric name | Labels | Description |
|-------------|--------|-------------|
| `grampus_total_tokens` | `model`, `agent_name` | Tokens consumed |
| `grampus_total_cost_usd` | `model`, `agent_name` | USD spent |
| `grampus_total_tool_calls` | `tool_name`, `agent_name` | Tool executions |
| `grampus_total_errors` | `error_code`, `agent_name` | Errors by type |
| `grampus_llm_call_count` | `model`, `agent_name` | Total LLM calls made |

### Gauge metrics

| Metric name | Labels | Description |
|-------------|--------|-------------|
| `grampus_active_agents` | `agent_name` | Currently running agents |

### Histogram metrics

| Metric name | Labels | Description |
|-------------|--------|-------------|
| `grampus_llm_latency_ms` | `model`, `agent_name` | LLM call latency in milliseconds |
| `grampus_tool_latency_ms` | `tool_name`, `agent_name` | Tool execution latency in milliseconds |

---

## EventLog

Append-only audit log for every agent action.

::: grampus.observability.events.EventLog
    options:
      show_source: false
      members: [append, replay, replay_since, event_count]

### AgentEvent

```python
@dataclass
class AgentEvent:
    event_id: str
    session_id: str
    agent_name: str
    event_type: str          # see event types table below
    summary: str             # human-readable one-line description
    payload: dict[str, Any]  # full event data
    timestamp: datetime
    step: int                # ReAct iteration number
```

### Event types

| Event type | Payload keys |
|-----------|-------------|
| `agent.started` | `agent_name`, `model`, `input` |
| `agent.completed` | `steps_taken`, `cost_usd`, `output_preview` |
| `agent.failed` | `error_code`, `error_message` |
| `llm.called` | `model`, `message_count`, `input_tokens` |
| `llm.responded` | `output_tokens`, `cost_usd`, `stop_reason` |
| `tool.called` | `tool_name`, `arguments` |
| `tool.completed` | `duration_ms`, `output_preview` |
| `tool.failed` | `error_code`, `error_message` |
| `memory.read` | `query`, `types`, `results_count` |
| `memory.written` | `memory_type`, `source_type`, `trust_level` |
| `safety.violation` | `violation_type`, `severity`, `blocked` |

---

## BehaviorMonitor

Tracks agent behavior patterns and detects anomalies.

::: grampus.observability.behavior.BehaviorMonitor
    options:
      show_source: false
      members: [record_turn, anomalies, profile]

### BehaviorAnomaly

```python
@dataclass
class BehaviorAnomaly:
    pattern: str              # "tool_usage_shift" | "cost_spike" | etc.
    severity: str             # "warning" | "critical"
    description: str          # human-readable explanation
    current_value: float      # observed metric value
    baseline_value: float     # expected (rolling average) value
    ratio: float              # current / baseline
```

### Monitored anomaly patterns

| Pattern | Trigger condition |
|---------|-----------------|
| `tool_usage_shift` | Tool X called > 2.5× or < 0.4× baseline frequency |
| `cost_spike` | Cost per run > 2.5× rolling average |
| `memory_access_anomaly` | Memory reads from unusual source types |
| `error_rate_spike` | Error rate > 2.5× baseline |
| `latency_spike` | P95 run duration > 2.5× baseline |
