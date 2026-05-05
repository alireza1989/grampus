# Safety API Reference

## SafetyPipeline

The main safety middleware. Compose injection detection, PII detection, and action guard into a unified check pipeline.

::: nexus.safety.pipeline.SafetyPipeline
    options:
      show_source: false
      members: [check_input, check_tool_result, check_llm_output, check_tool_call, get_violations]

---

## SafetyPipelineConfig

```python
from nexus.safety.pipeline import SafetyPipelineConfig

config = SafetyPipelineConfig(
    check_user_input=True,       # run injection + PII on user input
    check_tool_results=True,     # run injection + PII on tool output
    check_llm_output=True,       # run PII on LLM response (injection not blocked)
    check_memory_writes=True,    # run injection on memory write content
    log_violations=True,         # emit structlog events for violations
)
```

---

## Injection detector

::: nexus.safety.injection.PromptInjectionDetector
    options:
      show_source: false
      members: [check]

### InjectionCheckResult

```python
@dataclass
class InjectionCheckResult:
    detected: bool
    pattern: str | None        # matched pattern name
    confidence: float          # 0.0–1.0
    blocked: bool              # True if level blocks this confidence
```

---

## PII detector

::: nexus.safety.pii.PIIDetector
    options:
      show_source: false
      members: [check]

### PIICheckResult

```python
@dataclass
class PIICheckResult:
    detected: bool
    types_found: list[str]        # ["email", "phone", ...]
    redacted_text: str            # original if action="log", else redacted
    blocked: bool                 # True if action="block" and PII found
```

---

## Action guard

::: nexus.safety.action_guard.SafetyActionGuard
    options:
      show_source: false
      members: [check]

### ActionPolicy

```python
from nexus.safety.action_guard import ActionPolicy

policy = ActionPolicy(
    allowed_tools=["web_search", "calculate"],  # explicit allowlist (None = allow all)
    denied_tools=[],                             # explicit denylist
    max_tool_calls_per_turn=20,                  # across all tools per turn
    max_consecutive_tool_calls=8,                # before requiring LLM step
    max_cost_per_action_usd=0.05,               # per-tool-call cost cap
)
```

---

## SafetyViolation

Structured record emitted for every detected issue:

```python
@dataclass
class SafetyViolation:
    violation_type: str    # "injection" | "pii" | "action_blocked"
    severity: str          # "critical" | "high" | "medium" | "low"
    detail: str            # human-readable description
    blocked: bool          # True = request was blocked, False = logged only
    timestamp: datetime
```

---

## Policy loader

::: nexus.safety.policies.PolicyLoader
    options:
      show_source: false
      members: [load, from_file]

### Example policy YAML

```yaml
# safety_policy.yaml
injection:
  level: balanced

pii:
  action: redact
  types:
    - email
    - phone
    - ssn
    - credit_card

action_guard:
  allowed_tools:
    - web_search
    - calculate
  max_tool_calls_per_turn: 20
  max_consecutive_tool_calls: 8
  max_cost_per_action_usd: 0.05

pipeline:
  check_user_input: true
  check_tool_results: true
  check_llm_output: true
  check_memory_writes: true
  log_violations: true
```

Loading:

```python
from nexus.safety.policies import load_safety_policy
from nexus.safety.pipeline import SafetyPipeline

safety_config = load_safety_policy("safety_policy.yaml")
pipeline = SafetyPipeline.from_config(safety_config)
```
