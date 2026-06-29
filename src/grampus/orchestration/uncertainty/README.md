# `grampus/orchestration/uncertainty/` — Uncertainty Quantification (ADR-013)

This sub-package implements dual-process uncertainty quantification (UQ) as an optional hook in `AgentRunner`. It detects when an agent is operating in an uncertain zone and applies a three-tier escalation ladder: PROCEED → PROCEED_WITH_LOG → PAUSE_FOR_HUMAN → ABORT. For irreversible tool calls (send_email, delete, deploy), it can pause and require human confirmation even at moderate uncertainty.

Grounded in: Kadavath et al. 2022 (P(True)), arXiv 2601.15703 (dual-process AUQ, Jan 2026), arXiv 2504.03579 (adaptive semantic entropy, 2025), arXiv 2412.01033 (SAUP propagation, ACL 2025).

---

## Key abstractions

| Class | File | Role |
|---|---|---|
| `UncertaintyMonitor` | `monitor.py` | Session-level controller — wires all components, exposes the two hook points |
| `UncertaintyEstimator` | `estimator.py` | System 1 (P(True) + verbalized fusion) + System 2 (adaptive semantic entropy) |
| `UncertaintyPropagator` | `propagator.py` | SAUP: propagates uncertainty across steps with situational weights |
| `UncertaintyPolicy` | `policy.py` | Maps propagated confidence → UncertaintyAction (escalation ladder) |
| `AgentBeliefState` | `types.py` | Session-wide confidence state; updated after each step |
| `StepUncertainty` | `types.py` | Per-step estimates: p_true, verbalized, fused, semantic, propagated |
| `UncertaintyAction` | `types.py` | Enum: `PROCEED, PROCEED_WITH_LOG, PAUSE_FOR_HUMAN, ABORT` |
| `UncertaintyLevel` | `types.py` | Enum: `LOW, MEDIUM, HIGH, CRITICAL` |
| `UncertaintySource` | `types.py` | Enum: `LLM_CALL, TOOL_CALL, MEMORY_READ, DECISION` — used for SAUP step weights |

---

## Two-hook integration in `AgentRunner`

```python
# Hook 1: post-LLM-call
step_unc, action = await uncertainty_monitor.observe_llm_response(
    response_text=response.content,
    step_id=step_id,
    prompt_messages=messages,
    model_client=model_client,
    step_type="llm_call",
)

if action == UncertaintyAction.ABORT:
    raise UncertaintyError("Agent confidence too low", code="UNCERTAINTY_CRITICAL")
elif action == UncertaintyAction.PAUSE_FOR_HUMAN:
    # Inject REFLECTION_PROMPT before next LLM call so the model
    # explicitly acknowledges its uncertainty
    messages.append(Message(role=Role.SYSTEM, content=REFLECTION_PROMPT))

# Hook 2: pre-tool-call (for irreversible tools)
action = await uncertainty_monitor.check_tool_call(
    tool_name="send_email",
    step_id=step_id,
)
if action == UncertaintyAction.PAUSE_FOR_HUMAN:
    # Pause before the irreversible action
    ...
```

---

## Dual-process estimation

```
Any LLM response
    │
    ▼
System 1 (always runs, fast):
    ├─ Verbalized confidence: extract self-reported float from response text
    │   (weight 0.4 — weak signal, but cheap)
    ├─ P(True): ask model "Is your answer correct?" → binary → float
    │   (weight 0.6 — ECE ≈ 0.10 on frontier models)
    └─ Fused confidence = 0.6·p_true + 0.4·verbalized
    │
    │  Fused in uncertain zone (0.35 < fused < 0.65)?
    │  No  → skip System 2
    │  Yes → activate System 2
    ▼
System 2 (triggered on uncertainty, slower):
    ├─ Sample 2 responses at temperature > 0
    ├─ Jaccard ≥ 0.60? → early-stop (saves ~47% sampling cost)
    ├─ Jaccard < 0.60? → sample up to max_samples
    └─ Semantic entropy confidence from agreement rate
    Pessimistic fusion: min(system1_fused, semantic_entropy_conf)
```

---

## SAUP uncertainty propagation

Each step type has a situational weight reflecting how much a grounding error at that step affects downstream reasoning:

| Step type | Weight |
|---|---|
| `DECISION` | 0.70 — highest; wrong decisions compound |
| `LLM_CALL` | 0.55 |
| `TOOL_CALL` | 0.45 |
| `MEMORY_READ` | 0.35 — lowest; retrieved facts can be overridden |

Propagation formula (simplified):
```
propagated = prior_propagated * (1 - step_weight * step_uncertainty)
```

This means a confident step cannot erase an uncertain history — the accumulated uncertainty monotonically influences the propagated score. This is the 20% AUROC improvement from the SAUP paper.

---

## Escalation ladder

```
propagated_confidence
    │
    ≥ HIGH_THRESHOLD (0.75)  → PROCEED
    ≥ MEDIUM_THRESHOLD (0.50) → PROCEED_WITH_LOG (log uncertainty, continue)
    ≥ LOW_THRESHOLD (0.30)   → PAUSE_FOR_HUMAN  (inject reflection prompt)
    < LOW_THRESHOLD (0.30)   → ABORT (raise UncertaintyError)

Exception: irreversible tools PAUSE at MEDIUM uncertainty
Irreversible tool names (configurable): send_email, delete*, deploy*, publish*
```

---

## Hard invariants

- **`uncertainty_monitor=None` (default) means zero overhead.** When not set in `AgentRunner`, neither hook fires and no extra LLM calls are made.
- **`UncertaintyMonitor.initialize(session_id, agent_id)` must be called at the start of each `runner.run()`.** It resets the `AgentBeliefState`. Without this call, propagated uncertainty from previous sessions would carry over.
- **System 2 sampling makes additional LLM calls with the same `model_client`.** These count toward budget. Cap `max_samples` if budget is tight.
- **`REFLECTION_PROMPT` is injected as a `Role.SYSTEM` message**, not a user message, so it does not appear in the user-visible conversation history.
- **OTEL spans** (`uncertainty.estimate`, `uncertainty.semantic`, `uncertainty.escalate`) are emitted when a `tracer` is provided. These enable confidence dashboards alongside cost and latency metrics.

---

## Dependency map

```
uncertainty/ depends on:     core/ (types, errors, logging)
uncertainty/ is imported by: orchestration/runner.py
uncertainty/ must NOT import from: memory/, tools/, safety/, evaluation/
```

---

## ADR references

- **ADR-013** — Dual-process uncertainty quantification: full design rationale
