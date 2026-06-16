# Causal Analysis Guide

## Why causal analysis matters for agents

When an agent fails, the event log records *what happened* but not *why*. A flat sequence of
events does not distinguish between root causes and cascading effects. Consider this scenario:
at step 3 a web-search tool returned truncated output; at step 5 a summarizer tool failed
because it received truncated input; at step 8 the agent returned an error response. Without
causal analysis you see three failures. With causal analysis you see one root cause — the
truncated search output at step 3 — that cascaded into two downstream effects.

The same ambiguity arises in counterfactual reasoning: "would the agent have succeeded if it
had skipped the database lookup at step 4?" Standard logs cannot answer this. A structural
causal model (SCM) populated from the agent's own reasoning can.

Grampus provides a two-tier causal analysis layer. Tier 1 diagnoses failures from the event log
with no LLM inference. Tier 2 builds a persistent SCM from LLM-labeled causal claims and
answers intervention queries in pure Python. Both tiers integrate into `AgentRunner` as
optional hooks — zero overhead when not configured.

---

## Tier 1 — Root cause diagnosis

`CausalTracer` reconstructs a causal graph from the Phase 9 event log (see `EventLog` in
`src/grampus/observability/events.py`) and diagnoses root causes using three edge types:

**SEQUENTIAL** — event B immediately follows event A in the same session (ordered by
`step_index`). Every consecutive step pair gets a sequential edge. This captures the basic
execution flow.

**DATA_DEPENDENCY** — the first 20 characters of A's `output_snippet` appear as a substring
in B's `input_snippet`. This captures data flow: if `tool_web_search` returned a URL that
`tool_fetch_page` consumed as its input, a data dependency edge connects them. A concrete
example: if tool A returned `"https://example.com/docs"` at step 2, and tool B received
`"https://example.com/docs/api"` as input at step 4, that shared prefix creates a
data-dependency edge from step 2 to step 4.

**FAILURE_CASCADE** — event A has a non-empty `error_message`, and event B has `failed=True`
within 3 steps after A. Weight 2.0 — the strongest signal for root cause. This captures the
most common failure pattern: an error in one component propagates to downstream failures.

**Root cause ranking** uses a composite score:

```
structural_score = 1 / (1 + downstream_count)
positional_score = 1 - (step_index / max(total_steps - 1, 1))
composite_score  = 0.6 × structural + 0.4 × positional
```

A node with fewer downstream effects (closer to a leaf) scores higher structurally. An
earlier-occurring node scores higher positionally. Candidates are ranked by composite score
descending — the top candidate is the most likely root cause.

---

## Tier 2 — Interventional queries

`CausalWorldModel` maintains a persistent SCM for one agent. As the agent runs, the LLM's
responses are observed by `CausalRelationExtractor`, which makes a single low-temperature
LLM call to extract stated causal claims. The extracted relations populate the SCM's
adjacency list. `SimpleCausalInference` then answers P(Y|do(X)) queries — "what would
happen to outcome Y if we intervene to set variable X to value v?" — using pure-Python
backdoor adjustment over the accumulated DAG.

The critical design principle: **the LLM labels causal structure; code does all causal
inference.** This is the correct architecture per the Rung Collapse proof (arXiv 2602.11675),
which established that LLMs cannot climb Pearl's causal ladder natively without external
scaffolding. `SimpleCausalInference.intervene()` never calls the LLM — it walks the DAG,
applies the backdoor criterion, and returns a qualitative answer.

A practical example:

```python
result = await world_model.query_intervention(InterventionQuery(
    natural_language="What would have happened if we skipped the database lookup?",
    target_variable="tool_database_lookup",
    outcome_variable="state_answer_quality",
    intervention_value="skipped",
))
# result.answer: "Setting tool_database_lookup to 'skipped' causally affects
#                 state_answer_quality via: tool_database_lookup → state_answer_quality."
# result.causal_path: ["tool_database_lookup", "state_answer_quality"]
# result.is_identifiable: True
```

The `target_variable` and `outcome_variable` must be variable IDs (slugified descriptions).
Use `grampus.causal.world_model._slugify("tool database lookup")` to convert a description to
its slug form.

---

## Quick start

Minimal working example with full F4 setup:

```python
from grampus.causal import (
    CausalTracer,
    CausalRelationExtractor,
    CausalWorldModel,
    InterventionQuery,
)
from grampus.orchestration.runner import AgentRunner, RunnerConfig

# Build the F4 components
extractor = CausalRelationExtractor(model_client=model_client)
world_model = CausalWorldModel(
    state_store=dapr_store,
    extractor=extractor,
    agent_id="my-agent",
)
tracer = CausalTracer(event_store=event_log)

# Pass to AgentRunner (additive — all other params unchanged)
runner = AgentRunner(
    model_client=model_client,
    tool_executor=tool_executor,
    causal_world_model=world_model,
    causal_tracer=tracer,
    config=RunnerConfig(max_iterations=10),
)

# Run the agent normally
result = await runner.run(agent_def, "research quantum computing", session_id="sess-1")

# After a failed run — diagnose root cause
diagnosis = await tracer.diagnose("sess-1", "my-agent", failure_event_id="evt-42")
if diagnosis.root_causes:
    top = diagnosis.root_causes[0]
    print(f"Root cause: {top.event_type} at step {top.causal_chain[0]}")
    print(f"Causal chain: {' → '.join(top.causal_chain)}")
    print(f"Composite score: {top.composite_score:.3f}")

# Intervention query on the accumulated world model
from grampus.causal.world_model import _slugify

result = await world_model.query_intervention(InterventionQuery(
    natural_language="What if we had skipped the database lookup?",
    target_variable=_slugify("database lookup"),
    outcome_variable=_slugify("answer quality"),
    intervention_value="skipped",
))
print(result.answer)
print(f"Identifiable: {result.is_identifiable}, confidence: {result.confidence:.3f}")
```

The event log must expose `get_events_for_session(session_id, agent_id) -> list[dict]`.
Each dict needs: `event_id`, `event_type`, `step_index` (int), `failed` (bool),
`input_snippet` (str|None), `output_snippet` (str|None), `error_message` (str|None).

---

## Limitations

**F4 cannot infer causal structure from data statistically.** The `CausalWorldModel` only
knows what the LLM explicitly labels as causal. It cannot discover hidden confounders or
latent variables from observational data. That would require PC algorithm or GES (Peter-Clark
/ Greedy Equivalence Search), which need many observations and significantly more compute.

**The world model starts empty.** A freshly initialized `CausalWorldModel` has no variables.
The SCM grows over sessions as the agent observes text containing causal claims. Interventional
queries on sparse models return `is_identifiable=False`. Quality improves with session count.

**`SimpleCausalInference` handles simple DAGs only.** The pure-Python backdoor adjustment
works well for agent world models with fewer than 200 variables and no latent confounders.
For complex causal structures, the optional DoWhy backend provides more robust inference.

---

## Using DoWhy for complex graphs (optional)

Install the optional dependency:

```bash
pip install grampus-ai[causal]
# or: uv add "grampus-ai[causal]"
```

Then wrap `SimpleCausalInference` with a DoWhy-backed implementation:

```python
import dowhy
from grampus.causal.inference import SimpleCausalInference

class DoWhyInference(SimpleCausalInference):
    def _compute_intervention(self, query):
        # Build a DoWhy CausalModel from self._graph.adjacency
        # and run identification + estimation
        import networkx as nx
        G = nx.DiGraph()
        for cause, effects in self._graph.adjacency.items():
            for effect in effects:
                G.add_edge(cause, effect)
        model = dowhy.CausalModel(
            data=None,  # observational data not required for identification
            treatment=query.target_variable,
            outcome=query.outcome_variable,
            graph=G,
        )
        identified = model.identify_effect()
        # ... estimation and result formatting
        return super()._compute_intervention(query)
```

For world models expected to grow beyond 200 variables, DoWhy handles latent confounders,
non-parametric identification, and IV-based estimation that the pure-Python implementation
does not support.
