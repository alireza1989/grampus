# `grampus/safety/` — Safety & Guardrails

This package implements the runtime safety middleware that intercepts every agent input, LLM output, tool call, and tool result. It is integrated into `AgentRunner` as an optional `SafetyPipeline` parameter — when `safety_pipeline=None`, the agent runs without any safety checks.

---

## Key abstractions

| Class | File | Role |
|---|---|---|
| `SafetyPipeline` | `pipeline.py` | Orchestrates all safety checks; the single entry point from `AgentRunner` |
| `PromptInjectionDetector` | `injection.py` | Multi-layer injection detection: regex + heuristic + optional semantic classifier |
| `PIIDetector` | `pii.py` | Detects PII in tool I/O: email, phone, SSN, credit card, addresses |
| `ActionGuard` | `action_guard.py` | Per-agent tool allowlist/denylist, rate limiting, cost guard |
| `PolicyLoader` | `policies.py` | Loads safety config from YAML policy files |

---

## Pipeline hooks (where each check fires)

```python
# 1. Pre-execution: check user input for injection
await safety_pipeline.check_input(messages)
# → raises SafetyError(code="INPUT_BLOCKED") on injection detected

# 2. Post-LLM: scan LLM output (DETECT only — no block)
flags = await safety_pipeline.check_llm_output(response)
# → returns list[SafetyFlag], never raises
# → logs detected issues, logs PII, does NOT block response

# 3. Pre-tool-call: check tool authorization
await safety_pipeline.check_tool_call(tool_name, agent_id)
# → raises SafetyError(code="ACTION_BLOCKED") if tool is denied

# 4. Post-tool-result: check tool output for injection and PII
await safety_pipeline.check_tool_result(tool_result, tool_name)
# → raises SafetyError(code="TOOL_RESULT_BLOCKED") on injection
# → redacts PII in-place if configured to redact
```

The asymmetry is intentional: user input and tool results are external (untrusted), so they block. LLM output is internal and blocking would abort the agent — instead, LLM output is flagged for monitoring.

---

## Injection detection

`PromptInjectionDetector` uses three layers in order:

**Layer 1 — Regex patterns (fast):**
- Instruction override patterns: "ignore previous instructions", "disregard your training", "you are now..."
- Memory poisoning patterns: "remember that", "in all future conversations", "always respond with"
- Role manipulation: "your new role is", "act as", "pretend you are"
- 13 compiled patterns total

**Layer 2 — Heuristic scoring:**
- Unusual command density (imperative verbs per sentence)
- Sudden topic shifts in a single message
- Suspicious character sequences typical of injection payloads

**Layer 3 — Semantic classifier (optional):**
- Requires `[safety]` extras (small fine-tuned classifier model)
- Off by default — enable via policy YAML

Detection levels: `strict` (flag on Layer 1 alone), `balanced` (default, require 2+ layers), `permissive` (require all 3 layers). Configure via policy YAML.

---

## PII detection

`PIIDetector` uses compiled regex patterns for:
- Email addresses
- Phone numbers (US + international formats)
- Social Security Numbers (SSN)
- Credit card numbers (Luhn-validated)
- Street addresses

Actions (configurable per PII type):
- `log` — log the detection, pass through
- `redact` — replace with `[REDACTED]` before returning
- `block` — raise `SafetyError`

Optional spaCy NER integration (requires `[pii]` extras) for person names and organization names.

---

## Policy YAML configuration

```yaml
safety:
  injection:
    level: balanced        # strict | balanced | permissive
    block_on_input: true
    block_on_tool_result: true
  pii:
    email: redact
    phone: redact
    ssn: block
    credit_card: block
    address: log
  action_guard:
    allowed_tools: []      # empty = all allowed
    denied_tools:
      - shell_execute
      - filesystem_write
    max_calls_per_minute: 60
    max_cost_usd: 5.0
```

Load with `PolicyLoader.from_file("policies/safety.yaml")`.

---

## Hard invariants

- **`check_input()` and `check_tool_result()` always raise on detected injection** — they never return a flag list. The runner must stop on these; they are not optional signals.
- **`check_llm_output()` never raises** — it returns flags only. Blocking LLM output would cause the runner to abort on the most common path. Use the flags for monitoring and alerting.
- **`check_tool_call()` raises `SafetyError(code="ACTION_BLOCKED")`** from `ActionGuard`. This is distinct from injection — it means the agent is not authorized to call this tool, regardless of the content.
- **PII redaction modifies the `ToolResult.output` in place** before it is added to the agent's messages. The original unredacted output is never stored in memory or the event log.
- **Safety overhead must be < 5ms per check** (benchmark test in `tests/safety/test_benchmarks.py`). The regex and heuristic layers are fast. Never add a network call to the hot path without gating it behind a config flag.

---

## Extension guide

### Adding a new injection pattern

Edit `injection.py` — add to the `_INJECTION_PATTERNS` list. Patterns are compiled at import time. Test the new pattern with `tests/safety/test_injection.py`.

### Adding a new PII type

Edit `pii.py` — add a new compiled regex and corresponding `PIIType` enum value. Add the action config key to the policy YAML schema.

### Adding a new tool guard

`ActionGuard` in `action_guard.py` checks: (1) allowlist, (2) denylist, (3) rate limit, (4) cost guard. To add a new type of guard, add a method to `ActionGuard` and call it from `check()`.

---

## Dependency map

```
safety/ depends on:      core/ (errors, logging, types)
safety/ is imported by:  orchestration/runner.py
safety/ must NOT import from: memory/, tools/ (direct), dapr/, evaluation/
```

---

## ADR references

- **ADR-006** — Memory write provenance (safety also validates memory writes — see `memory/validator.py`)
- **ADR-007** — Sandbox by default (complements safety by isolating code execution)
- **ADR-008** — OpenTelemetry spans for safety checks
