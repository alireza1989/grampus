# Uncertainty Quantification

Uncertainty Quantification (UQ) gives every Grampus agent a real-time confidence signal and a three-tier escalation ladder. Instead of silently returning a low-quality answer when an LLM is unsure, UQ measures how confident the model actually is — step by step, across the entire run — and takes a principled action: proceed, log a warning, pause for human review, or abort.

---

## Why verbalized confidence is not enough

Most agentic frameworks ask the model to write `"confidence": 0.9` in its JSON output and treat that as the ground truth. Research shows this is unreliable:

| Signal | ECE (lower = better) | Notes |
|--------|----------------------|-------|
| Verbalized confidence | **0.377+** | Aligned models cluster at 90–100% regardless of accuracy (arXiv 2412.14737, KDD 2025) |
| P(True) self-evaluation | **~0.10** | Single follow-up call; works on any black-box API |
| Semantic entropy | **best AUROC** | N-sample entropy; most accurate but slower (Farquhar et al., *Nature* 2024) |

Grampus uses all three signals in a dual-process architecture: a fast path that always runs, and a slow path that activates only when the fast path is uncertain.

---

## Architecture: Dual-Process AUQ

```
LLM response
     │
     ▼
┌─────────────────────────────────────────────┐
│  System 1 (fast — always runs)              │
│  1. Extract verbalized confidence           │
│  2. P(True) follow-up call (optional)       │
│  3. Weighted, calibrated fusion             │
└───────────────┬─────────────────────────────┘
                │ fused ∈ trigger zone?
                ▼
┌─────────────────────────────────────────────┐
│  System 2 (slow — opt-in)                  │
│  4. Adaptive semantic entropy sampling      │
│     ├─ Start with 2 samples                │
│     ├─ Jaccard ≥ 0.6 → early stop         │
│     └─ Else extend to max_samples          │
│  5. Pessimistic fusion: min(fast, entropy) │
└───────────────┬─────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│  SAUP Propagation (across steps)            │
│  propagated = w·fused + (1-w)·cumulative   │
│  weights: decision=0.70, llm=0.55,         │
│           tool=0.45, memory_read=0.35       │
└───────────────┬─────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│  Three-Tier Escalation Policy               │
│  ≥ 0.80 → LOW    → PROCEED                 │
│  ≥ 0.60 → MEDIUM → PROCEED_WITH_LOG        │
│  ≥ 0.40 → HIGH   → PAUSE_FOR_HUMAN         │
│  < 0.40 → CRITICAL → ABORT (UncertaintyError)│
└─────────────────────────────────────────────┘
```

---

## Quick start

```python
from grampus.orchestration import AgentRunner, UncertaintyMonitor, UncertaintyPolicy

policy = UncertaintyPolicy(
    low_threshold=0.80,          # PROCEED below this
    medium_threshold=0.60,       # warn below this
    high_threshold=0.40,         # pause for human below this
    enable_p_true=True,          # run P(True) follow-up call
    irreversible_tool_names=["send_email", "delete_records", "deploy"],
)

monitor = UncertaintyMonitor(policy=policy)

runner = AgentRunner(
    model_client=client,
    tool_executor=executor,
    uncertainty_monitor=monitor,
)

result = await runner.run(agent_def, "Summarise this legal brief.", session_id="s1")

if result.status == AgentStatus.WAITING_FOR_HUMAN:
    meta = result.metadata.get("uncertainty", {})
    print(f"Paused — level: {meta['overall_level']}, confidence: {meta['cumulative_confidence']:.2f}")
```

---

## Escalation tiers

| Level | Propagated confidence | Action | What happens |
|-------|-----------------------|--------|--------------|
| **LOW** | ≥ 0.80 | `PROCEED` | Run continues normally |
| **MEDIUM** | ≥ 0.60 | `PROCEED_WITH_LOG` | Warning logged; run continues |
| **HIGH** | ≥ 0.40 | `PAUSE_FOR_HUMAN` | `status = WAITING_FOR_HUMAN`; optional reflection prompt injected |
| **CRITICAL** | < 0.40 | `ABORT` | `UncertaintyError` raised |

### Irreversible tool override

When a tool name matches any entry in `irreversible_tool_names` (case-insensitive substring match), MEDIUM escalates to `PAUSE_FOR_HUMAN`. LOW is always safe, even for irreversible tools.

```python
policy = UncertaintyPolicy(
    irreversible_tool_names=["send_email", "delete", "deploy", "transfer"],
)
```

If the agent is about to call `send_email_to_client` and cumulative confidence is 0.72 (MEDIUM), execution pauses — even though MEDIUM normally proceeds.

---

## Reflection injection

When HIGH uncertainty is detected on an LLM step and `inject_reflection_on_high=True` (the default), Grampus injects a System-2 reflection message before pausing:

```
Before you continue, assess your own uncertainty explicitly.
List the specific things you are NOT confident about in your current reasoning.
For each uncertain point state: (1) what you know, (2) what you don't know,
(3) what additional information would resolve the uncertainty.
```

The reflection appears as a `SYSTEM` message in `result.messages`. When you call `runner.resume()` with a human response, the next LLM call sees the reflection and the human's guidance together.

---

## Semantic entropy (slow path)

Enable semantic entropy sampling for high-stakes tasks where P(True) accuracy is not sufficient:

```python
policy = UncertaintyPolicy(
    enable_p_true=True,
    enable_semantic_sampling=True,   # opt-in slow path
)
estimator = UncertaintyEstimator(
    min_samples=2,                   # adaptive: start here
    max_samples=5,                   # extend to this if first pair disagrees
    early_stop_jaccard=0.60,         # first-pair agreement threshold
    semantic_trigger_low=0.50,       # only sample when fused is in this zone
    semantic_trigger_high=0.72,
)
monitor = UncertaintyMonitor(estimator=estimator, policy=policy)
```

The adaptive algorithm (arXiv 2504.03579) saves ~47% of sampling cost:

1. Sample 2 responses at temperature 0.8.
2. If first-pair Jaccard similarity ≥ 0.60 → stop early (model is clearly consistent).
3. Otherwise → extend to `max_samples`.
4. Compute Shannon entropy over semantic clusters. Cluster two responses together if Jaccard ≥ 0.40.
5. `confidence = 1.0 - H_norm`. Take `min(fast_path, entropy_conf)` (pessimistic fusion).

---

## SAUP propagation across steps

A single uncertain step should not be erased by subsequent confident steps. SAUP (arXiv 2412.01033, ACL 2025) weights each step type by its forward impact:

| Step type | Weight (`w`) | Effect |
|-----------|-------------|--------|
| `decision` | 0.70 | High influence — decisions cascade downstream |
| `llm_call` | 0.55 | Moderate — reasoning may drift |
| `tool_call` | 0.45 | Lower — results often grounding facts |
| `memory_read` | 0.35 | Lowest — retrieval rarely introduces new uncertainty |

Formula: `propagated(t) = w × fused(t) + (1 − w) × cumulative(t−1)`

A confident step 3 cannot erase a highly uncertain step 1 when w = 0.55.

---

## Graph integration: `uncertainty_guard_node`

Insert an explicit uncertainty checkpoint between graph nodes:

```python
from grampus.orchestration import uncertainty_guard_node, Graph, human_node

guard = uncertainty_guard_node(
    monitor,
    step_type="decision",          # used for SAUP weight lookup
    escalate_node="human_review",  # sets metadata["uncertainty_escalate"] = True
)

async def route(state):
    if state.metadata.get("uncertainty_escalate"):
        return "human_review"
    return "next_step"

graph = (
    Graph(graph_id="qa")
    .add_node("llm_step", llm_handler, entry=True)
    .add_node("guard", guard)
    .add_conditional_edge("guard", route, {"human_review": "human_review", "next_step": "final"})
    .add_node("human_review", human_node("Uncertain answer — please review."))
    .add_node("final", final_handler)
)
```

---

## Belief state and metadata

After each `runner.run()`, uncertainty metadata is attached to `result.metadata["uncertainty"]`:

```python
{
    "overall_level": "medium",       # UncertaintyLevel as string
    "cumulative_confidence": 0.74,   # EMA-propagated session-level confidence
    "total_steps": 4,
    "high_uncertainty_steps": 1,
    "last_step_id": "llm_3"
}
```

Access the full `AgentBeliefState` from the monitor after a run:

```python
belief = monitor.get_belief_state()
for step in belief.step_uncertainties:
    print(f"{step.step_id}: level={step.level} propagated={step.propagated_confidence:.3f}")
```

---

## OTEL spans

Three custom spans are emitted per step (when `tracer` is passed to `UncertaintyMonitor`):

| Span | Emitted when | Key attributes |
|------|-------------|----------------|
| `uncertainty.estimate` | Every step | `step_id`, `step_type`, `verbalized_confidence`, `p_true_confidence`, `fused_confidence`, `propagated_confidence`, `level`, `action`, `p_true_ran`, `samples_used` |
| `uncertainty.semantic` | Semantic sampling ran | `step_id`, `sample_count`, `early_stopped` |
| `uncertainty.escalate` | HIGH or CRITICAL level | `step_id`, `level`, `cumulative_confidence`, `irreversible` |

```python
from grampus.observability.tracer import GrampusTracer

tracer = GrampusTracer(service_name="my-agent", otlp_endpoint="http://localhost:4317")
monitor = UncertaintyMonitor(policy=policy, tracer=tracer)
```

---

## Handling a paused run

```python
result = await runner.run(agent_def, task, session_id="s1")

if result.status == AgentStatus.WAITING_FOR_HUMAN:
    # Inspect what caused the pause
    meta = result.metadata.get("uncertainty", {})
    print(f"Paused: {meta['overall_level']} confidence ({meta['cumulative_confidence']:.2f})")

    # Optionally read the reflection message
    for msg in result.messages:
        if msg.role == Role.SYSTEM and "NOT confident" in (msg.content or ""):
            print("Reflection:", msg.content)

    # Resume after human provides guidance
    human_guidance = "Focus on section 3.2 of the document."
    resumed = await runner.resume("my-agent", "s1", human_guidance)
```

---

## Configuration reference

### UncertaintyPolicy

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `low_threshold` | `float` | `0.80` | Propagated confidence floor for LOW (PROCEED) |
| `medium_threshold` | `float` | `0.60` | Floor for MEDIUM (PROCEED_WITH_LOG) |
| `high_threshold` | `float` | `0.40` | Floor for HIGH (PAUSE_FOR_HUMAN) |
| `enable_p_true` | `bool` | `True` | Run P(True) follow-up call after each LLM response |
| `enable_semantic_sampling` | `bool` | `False` | Enable adaptive semantic entropy slow path |
| `irreversible_tool_names` | `list[str]` | `[]` | Substrings; MEDIUM → PAUSE on match |
| `inject_reflection_on_high` | `bool` | `True` | Inject System-2 reflection before PAUSE |

### UncertaintyEstimator

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_p_true` | `bool` | `True` | Controls P(True) calls |
| `verbalized_weight` | `float` | `0.4` | Fusion weight for verbalized signal |
| `p_true_weight` | `float` | `0.6` | Fusion weight for P(True) signal |
| `verbalized_calibration_bias` | `float` | `0.25` | ECE correction for verbalized confidence |
| `p_true_calibration_bias` | `float` | `0.10` | ECE correction for P(True) |
| `min_samples` | `int` | `2` | Adaptive entropy: minimum samples before early-stop check |
| `max_samples` | `int` | `5` | Adaptive entropy: extend to this on disagreement |
| `semantic_trigger_low` | `float` | `0.50` | Lower bound of sampling trigger zone |
| `semantic_trigger_high` | `float` | `0.72` | Upper bound of sampling trigger zone |
| `early_stop_jaccard` | `float` | `0.60` | First-pair agreement threshold for early stop |

---

## Research citations

| Finding | Source | Baked-in design decision |
|---------|--------|--------------------------|
| Verbalized ECE ≥ 0.377 for frontier models | arXiv 2412.14737; ACM 3711896.3736569 | `verbalized_calibration_bias=0.25`; verbalized is weak signal (weight 0.4) |
| P(True) ECE ≈ 0.10 | Kadavath et al. 2022; validated 2023–2025 | P(True) is primary fast-path signal (weight 0.6) |
| Adaptive sampling saves 47% cost | arXiv 2504.03579 | `min_samples=2`, early-stop on Jaccard ≥ 0.6 |
| Semantic entropy best AUROC | Farquhar et al. 2024, *Nature* | Slow path; pessimistic fusion `min(fast, entropy)` |
| SAUP 20% AUROC improvement | arXiv 2412.01033; ACL 2025 pp. 6064–6073 | Per-step-type situational weights in `UncertaintyPropagator` |
| Dual-Process AUQ | arXiv 2601.15703, Jan 2026 | System 1 always; System 2 on uncertain zone |
| Three-tier escalation (production consensus) | Zylos Research April 2026 | `PROCEED → PROCEED_WITH_LOG → PAUSE_FOR_HUMAN → ABORT` |
