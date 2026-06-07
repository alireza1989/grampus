# Agent Handoffs

Agent handoffs let one agent explicitly transfer control to another mid-conversation — passing accumulated context, intent, and session state to a specialist agent. Use handoffs when a single agent detects it has reached the boundary of its expertise and needs to delegate; use [Multi-Agent Crews](multi-agent-crew.md) when you want parallel or pipeline coordination planned upfront.

---

## When to use handoffs vs. crews

| Situation | Use |
|-----------|-----|
| Agent discovers mid-run it needs a specialist (e.g., billing question in a support agent) | Handoff |
| You want agents running in parallel or sequentially from the start | Crew |
| Dynamic routing based on intent detected at runtime | Handoff |
| Fixed pipeline: researcher → critic → writer | Crew |

---

## Basic handoff

```python
import asyncio
import os

from nexus.core.models.anthropic import AnthropicClient
from nexus.core.types import AgentDefinition
from nexus.orchestration.handoff import (
    AgentRegistry,
    HandoffExecutor,
    HandoffPolicy,
    create_handoff_tool,
)
from nexus.orchestration.runner import AgentRunner, RunnerConfig
from nexus.tools.executor import ToolExecutor
from nexus.tools.registry import ToolRegistry


def make_client() -> AnthropicClient:
    return AnthropicClient(api_key=os.environ["NEXUS_MODEL__ANTHROPIC_API_KEY"])


# ── Build the billing specialist ────────────────────────────────────────────
billing_registry = ToolRegistry()
billing_runner = AgentRunner(
    model_client=make_client(),
    tool_executor=ToolExecutor(billing_registry),
    config=RunnerConfig(max_iterations=8, enable_memory=False),
)

# ── Register it in the global AgentRegistry ─────────────────────────────────
registry = AgentRegistry()
registry.register(
    agent_id="billing-agent",
    runner=billing_runner,
    agent_def=AgentDefinition(
        name="billing-agent",
        model="claude-sonnet-4-6",
        system_prompt="You are a billing specialist. Help users with payments and invoices.",
        tools=[],
    ),
    description="Handles billing inquiries, payment processing, and invoice questions.",
    skills=["billing", "payments", "invoices"],
)

# ── Build a handoff tool the front-line agent can call ───────────────────────
policy = HandoffPolicy(
    max_depth=3,                   # prevent infinite agent loops
    allowed_agents=["billing-agent"],   # explicit allowlist (None = allow all)
    sanitize_context=True,         # scan context for injection before handoff
)
executor = HandoffExecutor(registry=registry, policy=policy)
handoff_tool = create_handoff_tool(
    target_agent_id="billing-agent",
    description="Transfer to the billing specialist for payment and invoice questions.",
    handoff_executor=executor,
)

# ── Wire the handoff tool into the front-line agent ──────────────────────────
front_registry = ToolRegistry()
front_registry.register_tool_definition(handoff_tool)

front_runner = AgentRunner(
    model_client=make_client(),
    tool_executor=ToolExecutor(front_registry),
    config=RunnerConfig(max_iterations=10, enable_memory=False),
)
front_def = AgentDefinition(
    name="support-agent",
    model="claude-sonnet-4-6",
    system_prompt=(
        "You are a customer support agent. For general questions answer directly. "
        "For billing or payment questions, use the handoff_to_billing_agent tool."
    ),
    tools=["handoff_to_billing-agent"],
)


async def main() -> None:
    result = await front_runner.run(front_def, "I need help with my last invoice.")
    print(result.output)


asyncio.run(main())
```

---

## AgentCard and A2A discovery

Every registered agent exposes an `AgentCard` — a machine-readable capability manifest compliant with the A2A (Agent-to-Agent) protocol v1.2. External frameworks can discover Nexus agents and invoke them without Nexus-specific code.

**Endpoints:**

| URL | Method | Description |
|-----|--------|-------------|
| `/.well-known/agent.json` | `GET` | AgentCard for the primary agent |
| `/a2a/agents` | `GET` | List all registered agents' AgentCards |
| `/a2a/agents/{agent_id}` | `GET` | AgentCard for a specific agent |

**Example AgentCard response:**

```json
{
  "id": "billing-agent",
  "name": "Billing Agent",
  "description": "Handles billing inquiries, payment processing, and invoice questions.",
  "version": "1.0.0",
  "skills": ["billing", "payments", "invoices"],
  "protocol": "a2a/1.2",
  "endpoint": "https://your-service.example.com/a2a/agents/billing-agent/invoke"
}
```

This enables LangGraph agents, CrewAI agents, or any A2A-compatible framework to invoke your Nexus agent directly. See [ADR-010](../architecture/decisions.md) for the rationale behind A2A support.

---

## Security

### HandoffPolicy

`HandoffPolicy` controls what is allowed during a handoff:

```python
from nexus.orchestration.handoff import HandoffPolicy

policy = HandoffPolicy(
    max_depth=3,                         # stop after 3 hops to prevent loops
    allowed_agents=["billing-agent", "tech-support"],  # None = allow all registered
    sanitize_context=True,               # regex-scan context before passing
    trust_degradation=True,              # context tagged as LLM_GENERATED (default)
)
```

### Trust degradation

Context passed to the target agent is tagged as `SourceType.LLM_GENERATED` (trust level 0.7), not `USER_INPUT` (trust level 0.9). This means if the target agent writes handoff context to memory, it receives a lower trust score than direct user input — limiting the blast radius of a compromised handoff chain.

### Injection sanitization

When `sanitize_context=True` (the default), the handoff executor scans the context string for injection patterns before passing it to the target agent. This prevents a compromised upstream agent from poisoning the target agent's context.

!!! warning "Never disable sanitize_context in production"
    Prompt injection via handoff context is a real attack vector. An upstream agent that has been injected can attempt to pass malicious instructions through the handoff context. Always keep `sanitize_context=True` in production environments.

---

## Handoff events

Every handoff produces structured events in the event log for full auditability:

| Event type | Triggered by |
|-----------|-------------|
| `handoff.initiated` | `HandoffExecutor.execute()` called |
| `handoff.context_sanitized` | Context passed injection scan |
| `handoff.completed` | Target agent returned a result |
| `handoff.failed` | Target agent raised an error or policy denied the handoff |

Query these events via the event log:

```python
from nexus.observability.events import EventLog

event_log = EventLog(state_store=state_store)
events = await event_log.get_events(
    session_id="session-42",
    event_type_prefix="handoff",
)
for event in events:
    print(f"[{event.timestamp}] {event.event_type}: {event.summary}")
```

---

## Multi-hop handoffs

Agents can hand off to agents that hand off further. The `max_depth` guard in `HandoffPolicy` caps the chain length to prevent infinite loops:

```
user → support-agent → billing-agent → payment-specialist
                                                ↑
                              max_depth=3 stops here
```

If `max_depth` is exceeded, the current agent receives a `HandoffDepthExceededError` and should respond to the user directly rather than delegating further.

---

## See also

- **[Multi-Agent Crew →](multi-agent-crew.md)** — Plan upfront coordination with sequential, parallel, and hierarchical patterns
- **[Observability guide →](observability.md)** — Trace handoff events with OTEL spans
- **[Architecture Decisions →](../architecture/decisions.md)** — ADR-010: MCP + A2A protocol support
