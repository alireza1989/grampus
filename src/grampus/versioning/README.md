# `grampus/versioning/` — Agent Versioning & A/B Testing (H50)

This package implements content-addressed agent versioning with deterministic A/B routing and pure-Python statistical significance testing. It tracks what `AgentDefinition` was deployed when, enables rollback, and runs controlled experiments — all backed by the existing Dapr infrastructure.

Described in ADR-025.

---

## Key abstractions

| Class | File | Role |
|---|---|---|
| `VersionStore` | `store.py` | Dapr-backed persistence for versions and deployment history |
| `VersionRouter` | `router.py` | Resolves which version to use per request; sticky deterministic A/B assignment |
| `ABTestManager` | `ab_testing.py` | Creates experiments, records results, evaluates auto-promotion |
| `VersionMetrics` | `metrics.py` | Aggregated stats per version: pass_rate, avg_cost_usd, avg_latency_seconds |
| `VersionStats` | `stats.py` | `two_proportion_z_test` and `welch_t_test` — pure Python, no scipy |
| `AgentVersion` | `types.py` | Stored version: version_id, agent_id, definition, created_at, tags |
| `DeploymentRecord` | `types.py` | One deployment event: version_id, deployed_at, deployed_by, notes |
| `ABExperiment` | `types.py` | Experiment config: control_version_id, treatment_version_id, split, target |
| `ABResult` | `types.py` | Experiment outcome: control_metric, treatment_metric, p_value, auto_promoted |
| `compute_version_id` | `store.py` | Pure function: SHA-256 over canonicalized `AgentDefinition` JSON |

---

## Content-addressed version IDs

```python
from grampus.versioning.store import compute_version_id

version_id = compute_version_id(agent_def)
# → deterministic SHA-256 hash over canonicalized JSON
# Same AgentDefinition always produces the same ID regardless of when/where computed
# Identical definitions are deduplicated at save time automatically
```

The canonicalization: key-sorted JSON with tool list sorted by tool name. This ensures two `AgentDefinition` instances with the same content but different field order produce the same version ID.

---

## Version storage

```python
from grampus.versioning.store import VersionStore

store = VersionStore(state_store=dapr_store)

# Save a version (idempotent — same content = same ID, no duplicate created)
version_id = await store.save_version(agent_def, agent_id="researcher")

# List versions (newest first, skips corrupt records with a warning)
versions = await store.list_versions(agent_id="researcher")

# Record a deployment event
await store.record_deployment(
    agent_id="researcher",
    version_id=version_id,
    deployed_by="alireza",
    notes="Prompt refinement for conciseness",
)

# Get deployment history (capped at 50 entries)
history = await store.get_deployment_history(agent_id="researcher")
```

---

## A/B testing

```python
from grampus.versioning.ab_testing import ABTestManager
from grampus.versioning.types import ABExperiment

ab_manager = ABTestManager(store=store)

# Create an experiment
experiment = await ab_manager.create_experiment(ABExperiment(
    id="exp-concise-prompt",
    agent_id="researcher",
    control_version_id=v1_id,
    treatment_version_id=v2_id,
    split=0.20,                        # 20% of traffic to treatment
    metric="eval_pass_rate",
    auto_promote_threshold=0.05,       # p < 0.05 → auto-promote treatment
    min_samples=100,                   # need 100 observations per arm
))

# Record eval run results (called from your eval pipeline)
await ab_manager.record_eval_result(
    experiment_id="exp-concise-prompt",
    version_id=v2_id,
    passed=True,
    cost_usd=0.0023,
    latency_seconds=1.4,
)

# Check experiment status
result = await ab_manager.evaluate_experiment("exp-concise-prompt")
if result.auto_promoted:
    print(f"Treatment auto-promoted! p={result.p_value:.4f}")
```

---

## Sticky deterministic A/B routing

```python
from grampus.versioning.router import VersionRouter

router = VersionRouter(ab_manager=ab_manager)

# In AgentRunner:
runner = AgentRunner(..., version_router=router)

# For each run, VersionRouter.resolve(agent_id, user_id) selects the version:
version_id = await router.resolve(agent_id="researcher", user_id="user-42")
# → SHA-256(experiment_id:user_id) % 100 < int(split * 100)
# → Same user always lands in the same bucket for a given experiment
# → No server-side session state required
```

The hash-based assignment is **deterministic and stateless** — the same `(experiment_id, user_id)` always maps to the same bucket. This prevents the same user from seeing different behaviors on consecutive requests (which would contaminate experiment results).

---

## Significance testing (pure Python)

```python
from grampus.versioning.stats import two_proportion_z_test, welch_t_test

# For pass/fail metrics (eval pass rate)
p_value = two_proportion_z_test(
    successes_a=control_passes,
    n_a=control_total,
    successes_b=treatment_passes,
    n_b=treatment_total,
)

# For continuous metrics (cost, latency)
# Note: uses a 10%-difference heuristic, not a true p-value (see ADR-025)
p_value = welch_t_test(
    mean_a=control_avg_cost,
    mean_b=treatment_avg_cost,
    n_a=control_n,
    n_b=treatment_n,
)
```

Both functions use only stdlib `math` — no scipy, no numpy. `two_proportion_z_test` uses `math.erfc` for the two-tailed p-value. `welch_t_test` uses Lentz's continued-fraction regularized incomplete beta function.

---

## Hard invariants

- **`compute_version_id` is pure** — no I/O, no randomness. Same input always → same output. Test by computing the same definition twice and asserting equality.
- **Deployment history is capped at 50 entries** (`VersionStore._MAX_HISTORY = 50`). Older entries are silently dropped. The authoritative audit trail is the Dapr event log (ADR-005).
- **`list_versions()` skips corrupt records with a warning**, rather than raising. This makes the function robust to partial Dapr failures — you always get back the versions that could be loaded.
- **`VersionRouter.resolve()` is wrapped in `contextlib.suppress`** — a broken experiment never crashes agent execution. If routing fails, the default (control) version is used.
- **A/B routing is for experiments only** — production deployments should use `store.record_deployment()` and pass the resolved `version_id` directly. Do not use `VersionRouter` for non-experimental production traffic.

---

## Dependency map

```
versioning/ depends on:     core/ (types, errors, logging), dapr/ (VersionStore),
                            evaluation/ (EvalSuite for auto-promotion scoring)
versioning/ is imported by: orchestration/runner.py (version_router param), cli/
versioning/ must NOT import from: memory/, tools/, safety/, observability/
```

---

## ADR references

- **ADR-025** — Content-addressed agent versioning with deterministic A/B routing: full design rationale
