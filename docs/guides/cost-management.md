# Cost Management & Alerts

Nexus tracks cost at every level — individual model calls, tool steps, sessions, and agent lifetime — and can enforce hard budgets that stop an agent before it overspends. For production deployments, cost alerts fire notifications to Slack, email, or webhooks when thresholds are crossed.

---

## Cost tracking

`CostTracker` records token usage and USD cost for every LLM call. The data is available immediately on `ExecutionResult`:

```python
result = await runner.run(agent_def, "Research quantum computing.", session_id="s1")

print(f"Input tokens:  {result.token_usage.input_tokens}")
print(f"Output tokens: {result.token_usage.output_tokens}")
print(f"Total tokens:  {result.token_usage.total_tokens}")
print(f"Cost:          ${result.token_usage.cost_usd:.6f}")
print(f"Model:         {result.token_usage.model}")
```

Cost events are also published to the Dapr pub/sub bus, where the `BehaviorMonitor` and alert evaluator consume them.

---

## Budget enforcement

Set `cost_budget_usd` on `AgentDefinition` to stop the agent before it exceeds a per-run limit. When the budget is reached, `AgentRunner` raises `BudgetExceededError` before the next LLM call:

```python
from nexus.core.errors import BudgetExceededError
from nexus.core.types import AgentDefinition

agent_def = AgentDefinition(
    name="research-bot",
    model="claude-sonnet-4-6",
    system_prompt="You are a research assistant.",
    cost_budget_usd=0.50,   # hard stop at $0.50 per run
)

try:
    result = await runner.run(agent_def, "Research every aspect of quantum computing.")
except BudgetExceededError as e:
    print(f"Budget exceeded: {e}")
    print(f"Spent so far:    ${e.details['cost_usd_so_far']:.4f}")
    print(f"Limit:           ${e.details['budget_usd']:.4f}")
```

Use `nexus cost` on the CLI to review recent spending across all agents. See the [CLI reference](../reference/cli.md).

---

## Cost alerts

Cost alerts let you react to spending patterns before they become surprises. Define rules, attach notification channels, and let the `AlertEvaluator` watch for threshold crossings.

### Defining alert rules

```python
from nexus.observability.alerts import AlertEvaluator, AlertRule, AlertSeverity, ThresholdType
from nexus.observability.notification import (
    LogChannel,
    NotificationDispatcher,
    SlackChannel,
    SmtpChannel,
    WebhookChannel,
)

rules = [
    AlertRule(
        name="session-budget",
        threshold_type=ThresholdType.PER_SESSION_USD,
        threshold_usd=0.10,
        severity=AlertSeverity.WARNING,
        cooldown_seconds=3600,   # at most one alert per hour for this rule
    ),
    AlertRule(
        name="daily-spend-research-bot",
        agent_id="research-bot",       # None = applies to all agents
        threshold_type=ThresholdType.PER_DAY_USD,
        threshold_usd=5.00,
        severity=AlertSeverity.CRITICAL,
        cooldown_seconds=86400,
    ),
    AlertRule(
        name="per-run-spike",
        threshold_type=ThresholdType.PER_RUN_USD,
        threshold_usd=0.25,
        severity=AlertSeverity.WARNING,
    ),
]

dispatcher = NotificationDispatcher(
    channels=[
        SlackChannel(webhook_url="https://hooks.slack.com/services/..."),
        SmtpChannel(
            host="smtp.myco.com",
            port=587,
            username="alerts@myco.com",
            password="...",
            to_addrs=["ops@myco.com"],
        ),
        LogChannel(),   # always log — even if Slack/email fails
    ]
)

evaluator = AlertEvaluator(rules=rules, dispatcher=dispatcher)
```

Pass the evaluator to `AgentRunner` and it monitors every cost event automatically:

```python
runner = AgentRunner(
    model_client=client,
    tool_executor=executor,
    config=RunnerConfig(max_iterations=10),
    alert_evaluator=evaluator,
)
```

### Notification channels

| Channel | Requires | Behavior |
|---------|----------|----------|
| `WebhookChannel` | `url` | `POST` JSON payload; optional HMAC-SHA256 signature for verification |
| `SlackChannel` | Incoming webhook URL | Slack Block Kit message with severity color coding |
| `SmtpChannel` | SMTP host and credentials | Plain text email via stdlib `smtplib` — no extra dependencies |
| `LogChannel` | Nothing | `structlog` warning; always succeeds even if other channels fail |

### Threshold types

| Type | Resets | Use case |
|------|--------|----------|
| `PER_RUN_USD` | Each run | Alert on single expensive runs |
| `PER_SESSION_USD` | Each session | Track cumulative session spend |
| `PER_HOUR_USD` | Each hour (rolling) | Detect rate spikes |
| `PER_DAY_USD` | Midnight UTC | Daily budget monitoring |
| `PER_MONTH_USD` | First of month UTC | Monthly billing control |

---

## Alert rules via REST API

When the Nexus server is running (`nexus serve`), manage alert rules over HTTP:

```bash
# Create a rule
curl -X POST http://localhost:8000/alerts/rules \
  -H "Content-Type: application/json" \
  -d '{
    "name": "session-budget",
    "threshold_type": "per_session_usd",
    "threshold_usd": 0.10,
    "severity": "warning",
    "cooldown_seconds": 3600
  }'

# List all rules
curl http://localhost:8000/alerts/rules

# Enable or disable a rule
curl -X PATCH http://localhost:8000/alerts/rules/rule_abc123 \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'

# Delete a rule
curl -X DELETE http://localhost:8000/alerts/rules/rule_abc123

# View alert history (last 500 events)
curl "http://localhost:8000/alerts/history?limit=50"
```

---

## Alert management via CLI

```bash
# List all configured rules
nexus alerts list

# Add a new rule interactively
nexus alerts add \
  --name "daily-spend" \
  --threshold-usd 5.00 \
  --threshold-type per_day_usd \
  --severity critical \
  --agent-id research-bot \
  --cooldown 86400

# Enable or disable a rule
nexus alerts enable  <rule_id>
nexus alerts disable <rule_id>

# Remove a rule
nexus alerts remove <rule_id>

# Fire a test notification for a rule (verify channels work)
nexus alerts test <rule_id>
```

See the [CLI reference](../reference/cli.md) for all flags.

---

## See also

- **[Observability guide →](observability.md)** — Prometheus metrics for cost, Grafana dashboard
- **[CLI reference →](../reference/cli.md)** — `nexus cost` and `nexus alerts` commands
- **[Configuration reference →](../reference/config.md)** — Cost alert config in `nexus.yaml`
