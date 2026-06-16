# Artifact-Centric Collaboration

## What is artifact-centric collaboration?

When multiple agents work on the same deliverable — a research report, a codebase, a database schema — passing text strings between them breaks down at scale. Strings have no structure, no ownership, and no conflict detection. Two agents can silently write incompatible outputs that no one catches until after the fact.

The Specification Gap paper (arXiv 2603.24284, March 2026) quantified this: implicit shared specifications reduce two-agent integration accuracy by 25–39 percentage points. Agents that know "write something about background" produce outputs that cannot be composed without manual review.

Artifact-centric collaboration solves this by making the shared deliverable a first-class object. An `Artifact` is a versioned, schema-validated, ownership-tracked structure. Every section has an explicit specification. Every write is validated at write time — not post-hoc. Agents claim exclusive ownership of a section, write it, and release it with a MERGED status. Other agents can depend on MERGED sections and receive concise summaries, not the full history.

## When to use ArtifactCrew vs standard Crew

Use `ArtifactCrew` when:
- Multiple agents contribute to a **single deliverable** (report, codebase, schema, contract)
- Sections have **clear dependencies** (e.g., analysis depends on background)
- You need a **verifiable, versioned output** that can be audited

Use standard `Crew` when:
- Agents produce **independent outputs** that don't need to be integrated
- Tasks are **sequential pipelines** where each step takes the previous step's output as a string
- The deliverable is a **single LLM response**, not a structured document

## Quick start: research report

```python
from grampus.orchestration.artifact import (
    ArtifactCollaborator,
    ArtifactContentType,
    ArtifactCrew,
    ArtifactSchema,
    ArtifactStore,
    ConflictDetector,
    SchemaValidator,
    SectionLockManager,
    SectionSchema,
)

# 1. Define the artifact schema (schema-first — the Specification Gap insight)
schema = ArtifactSchema(
    artifact_type="research_report",
    description="A 4-section research report on quantum computing applications",
    sections=[
        SectionSchema(
            section_id="executive_summary",
            description=(
                "2-paragraph summary of findings. Must include: "
                "main conclusion, top 3 applications, one risk."
            ),
            content_type=ArtifactContentType.MARKDOWN,
        ),
        SectionSchema(
            section_id="background",
            description="Technical background on quantum computing. 400–600 words.",
            content_type=ArtifactContentType.MARKDOWN,
        ),
        SectionSchema(
            section_id="applications",
            description="Top 5 near-term applications with feasibility assessment for each.",
            content_type=ArtifactContentType.JSON,
            dependencies=["background"],
            required_fields=["applications"],
        ),
        SectionSchema(
            section_id="conclusion",
            description="Synthesis of background + applications into recommendations.",
            content_type=ArtifactContentType.MARKDOWN,
            dependencies=["background", "applications"],
        ),
    ],
    global_constraints=[
        "All sections must be consistent with each other on terminology.",
        "Do not contradict the executive_summary's stated main conclusion.",
    ],
)

# 2. Create the artifact store and the artifact itself
store = ArtifactStore(state_store=your_dapr_state_store, validator=SchemaValidator())
artifact = await store.create(schema)

# 3. Build agents and collaborators
agents = [agent_runner_1, agent_runner_2]
collaborators = [
    ArtifactCollaborator(
        agent_id="researcher-1",
        store=store,
        lock_manager=SectionLockManager(lock_factory, store),
        conflict_detector=ConflictDetector(store),
    ),
    ArtifactCollaborator(
        agent_id="researcher-2",
        store=store,
        lock_manager=SectionLockManager(lock_factory, store),
        conflict_detector=ConflictDetector(store),
    ),
]

# 4. Run the crew
crew = ArtifactCrew(agents=agents, collaborators=collaborators, store=store, max_retries=2)
completed = await crew.run(artifact.artifact_id, "Write a research report on quantum computing")

print(f"Completed at: {completed.completed_at}")
for section_id, section in completed.sections.items():
    print(f"{section_id}: version={section.version}, state={section.ownership_state}")
```

## Schema design tips

The Specification Gap finding is the most actionable result from this research: **the more explicit your section schema, the higher the integration accuracy**.

**Bad schema** (vague):
```python
SectionSchema(
    section_id="analysis",
    description="Write something about the analysis",
    content_type=ArtifactContentType.TEXT,
)
```

**Good schema** (explicit):
```python
SectionSchema(
    section_id="analysis",
    description=(
        "Quantitative analysis of the top 3 findings from the background section. "
        "For each finding: state the claim, cite the evidence, assess confidence (high/medium/low)."
    ),
    content_type=ArtifactContentType.JSON,
    dependencies=["background"],
    required_fields=["findings"],
    validation_rules=[
        "Each finding must have keys: claim, evidence, confidence",
        "confidence must be one of: high, medium, low",
        "Do not introduce findings not mentioned in the background section",
    ],
)
```

Rules of thumb:
- Always specify `content_type` explicitly — it drives schema validation
- For JSON sections, always list `required_fields` — they are validated before persistence
- Write `description` in imperative form, not as a vague label
- Use `validation_rules` for LLM self-check constraints that are hard to encode in JSON schema
- Use `dependencies` to express data flow — the crew enforces execution order

## Conflict handling

Three conflict types can occur during a write. The `resolution` field tells callers what to do:

| Conflict type | When it occurs | Resolution |
|---|---|---|
| `OWNERSHIP` | Agent writes to a section it doesn't own | `reject` — claim the section first |
| `VERSION_MISMATCH` | `expected_version` doesn't match stored version | `retry` — reload and try again |
| `SCHEMA_VALIDATION` | Content fails required_fields or max_tokens check | `reject` — fix the content |
| `DEPENDENCY_VERSION` | Writing before a dependency section is done | `retry` — wait for the dependency |

```python
from grampus.core.errors import ArtifactConflictError

try:
    completed = await crew.run(artifact_id, task_description)
except ArtifactConflictError as exc:
    print(f"Failed sections: {exc.details.get('failed_sections')}")
    print(f"Error code: {exc.code}")
    # Codes: SECTIONS_FAILED, SECTIONS_NOT_MERGED, CIRCULAR_DEPENDENCY
```

For individual writes via `ArtifactCollaborator`:
```python
result = await collaborator.write_section(artifact_id, section_id, content)
if not result.success:
    conflict = result.conflict
    if conflict.resolution == "retry":
        # Reload and try again after dependencies complete
        ...
    elif conflict.resolution == "reject":
        # The content is invalid — fix it before retrying
        ...
    elif conflict.resolution == "human_review":
        # Advisory conflict — can proceed but may want to review
        ...
```

## Graph integration via artifact_node()

`artifact_node()` lets you integrate a single-section artifact write into a `Graph` pipeline:

```python
from grampus.orchestration.artifact import ArtifactStore, ArtifactCollaborator
from grampus.orchestration.nodes import artifact_node, human_node, conditional_node
from grampus.orchestration.graph import Graph

# Each node claims, writes, and releases one section
background_node = artifact_node(store, collaborator_1, section_id="background")
conclusion_node = artifact_node(store, collaborator_2, section_id="conclusion")

def route_after_background(state):
    result = state.metadata.get("artifact_result", {})
    return "conclusion" if result.get("success") else "error"

graph = Graph()
graph.add_node("background", background_node)
graph.add_node("conclusion", conclusion_node)
graph.add_node("error", human_node("Background write failed — human review needed"))
graph.add_conditional_edge("background", route_after_background)

# Set artifact_id and task in state.metadata before running
state.metadata["artifact_id"] = artifact.artifact_id
state.metadata["artifact_task"] = "Write a research report"
final_state = await graph.execute(state)
```

The node reads `state.metadata["artifact_id"]` and `state.metadata["artifact_task"]`, and writes `state.metadata["artifact_result"]` (an `ArtifactEditResult` dict). Sets `state.status = AgentStatus.FAILED` on write failure so conditional edges can route to a `human_node`.

## How CAID scoped context prevents error propagation

When an agent calls `collaborator.get_scoped_context(artifact_id, section_id)`, it receives:

```
{
  "artifact_description": "A 4-section research report on quantum computing",
  "section_schema": { ... full SectionSchema for its section ... },
  "completed_dependencies": {
    "background": "Quantum computing uses qubits to represent information..."  # first 200 chars
  },
  "global_constraints": ["All sections must be consistent on terminology.", ...]
}
```

It does **not** receive:
- Other agents' in-progress work
- Full conversation history of completed sections
- Sections outside its dependency chain

This is the CAID insight (arXiv 2603.21489): scoped context reduces error propagation by confining each agent to its own section + completed dependencies only. A mistake in `analysis` cannot corrupt `background` because `background` never sees `analysis` content.
