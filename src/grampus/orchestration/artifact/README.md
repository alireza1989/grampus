# `grampus/orchestration/artifact/` — Artifact-Centric Collaboration (ADR-015)

This sub-package implements `ArtifactCrew` — a multi-agent collaboration pattern where multiple agents co-author a structured document or codebase. Each section has an explicit schema, agents claim exclusive ownership via distributed lock, and conflicts are detected at write time rather than post-hoc. This prevents the 25–39 percentage-point accuracy loss from implicit shared specifications (Specification Gap paper, arXiv 2603.24284, March 2026).

---

## Key abstractions

| Class | File | Role |
|---|---|---|
| `ArtifactCrew` | `crew.py` | Wave-based parallel execution: claims → writes → integration checks per wave |
| `ArtifactCollaborator` | `collaborator.py` | Per-agent interface: claim section → generate content → write section |
| `ArtifactStore` | `store.py` | Dapr-backed artifact CRUD; write-time schema validation + conflict detection |
| `SectionLockManager` | `lock_manager.py` | Atomic section claiming via Dapr distributed lock (at-most-one winner) |
| `SchemaValidator` | `schema.py` | Validates section content against `SectionSchema` before persisting |
| `ConflictDetector` | `conflict_detector.py` | Dependency-version checks at write time |
| `Artifact` | `types.py` | Top-level document: schema (immutable after creation) + sections (mutable via lifecycle) |
| `ArtifactSchema` | `types.py` | Schema for the full artifact: list of `SectionSchema` with dependency graph |
| `SectionSchema` | `types.py` | Schema for one section: description, content_type, required_fields, dependencies |
| `ArtifactSection` | `types.py` | One completed section: content, author_agent_id, ownership_state, timestamp |
| `ScopedContext` | `types.py` | Context passed to each agent: schema + assigned section + completed dep summaries |
| `ArtifactEditResult` | `types.py` | Return from one section write: ok, conflict, validation_errors |
| `OwnershipState` | `types.py` | MESI-inspired enum: `UNOWNED, CLAIMED, REVIEWING, MERGED` |

---

## How artifact collaboration works

```
ArtifactCrew.run(artifact_id, task_description)
    │
    ▼
Load Artifact from ArtifactStore (schema + current sections)
Build dependency DAG from section schemas
Kahn's algorithm → execution waves
    │
    ▼
For each wave (sequential between waves, parallel within):
    asyncio.gather(
        collaborator.edit_section(agent, artifact_id, section_id, task)
        for section_id in wave
    )
    │
    ▼
ArtifactCollaborator.edit_section(agent_runner, artifact_id, section_id, task):
    1. SectionLockManager.claim(artifact_id, section_id)
       → Dapr distributed lock — at-most-one winner, LockAcquisitionError on failure
       → OwnershipState: UNOWNED → CLAIMED
    2. Build ScopedContext:
       global schema + this section schema + one-line summaries of completed deps
       (Full artifact history is NEVER passed)
    3. agent_runner.run(scoped_context) → section content
    4. ArtifactStore.write_section(section_id, content)
       → SchemaValidator: JSON sections validate required_fields
       → ConflictDetector: check dependency version integrity
       → OwnershipState: CLAIMED → REVIEWING → MERGED
       → Raises ArtifactConflictError on validation failure
    5. SectionLockManager.release(artifact_id, section_id)
    │
    ▼
Integration check after each wave completes
    Verify all sections in the wave were successfully merged
    Failed sections retry up to max_retries, then added to failed list
    │
    ▼
Return completed Artifact or raise ArtifactConflictError with failed sections
```

---

## Usage

```python
from grampus.orchestration.artifact.store import ArtifactStore
from grampus.orchestration.artifact.collaborator import ArtifactCollaborator
from grampus.orchestration.artifact.crew import ArtifactCrew
from grampus.orchestration.artifact.types import Artifact, ArtifactSchema, SectionSchema

# Define the artifact schema (MUST be done before any agent is assigned)
schema = ArtifactSchema(
    name="competitive-analysis",
    sections=[
        SectionSchema(
            id="executive-summary",
            description="High-level summary of the competitive landscape",
            content_type="TEXT",
            required_fields=[],
            dependencies=[],
        ),
        SectionSchema(
            id="competitor-profiles",
            description="Detailed profile for each major competitor",
            content_type="MARKDOWN",
            required_fields=[],
            dependencies=["executive-summary"],
        ),
        SectionSchema(
            id="recommendations",
            description="Strategic recommendations based on analysis",
            content_type="JSON",
            required_fields=["actions", "priority", "timeline"],
            dependencies=["executive-summary", "competitor-profiles"],
        ),
    ],
)

# Create the artifact (schema is immutable after this point)
artifact = await store.create(schema)

# Build the crew
crew = ArtifactCrew(
    agents=[researcher, analyst, strategist],
    collaborators=[
        ArtifactCollaborator(store, lock_manager),
        ArtifactCollaborator(store, lock_manager),
        ArtifactCollaborator(store, lock_manager),
    ],
    store=store,
    max_retries=2,
)

completed_artifact = await crew.run(artifact_id=artifact.id, task_description="...")
```

---

## Content type validation

| Content type | Validation rule |
|---|---|
| `JSON` | Must parse as valid JSON; `required_fields` keys must be present at top level |
| `TEXT` | Any non-empty string |
| `MARKDOWN` | Any non-empty string |
| `CODE` | Any non-empty string |

`SchemaValidator.validate()` runs inside `ArtifactStore.write_section()` before persisting. Write-time validation is a hard requirement — post-hoc validation is insufficient (STORM finding: 18.7 point improvement for write-time vs post-hoc conflict detection).

---

## Hard invariants

- **`Artifact.schema` is immutable after creation.** You cannot add, remove, or modify section schemas after the artifact is created. Sections can only be added or updated through the `claim → write → release` lifecycle.
- **Circular dependencies in the section DAG raise `ArtifactConflictError(code="CIRCULAR_DEPENDENCY")` at wave-build time** — before any agent runs.
- **`SectionLockManager` uses the existing Dapr distributed lock** from Phase 2. Do not implement a custom locking mechanism.
- **Scoped context per agent** — each agent receives only its assigned section schema + summaries of completed dependencies. The full artifact content is never passed to any single agent.
- **`ArtifactConflictError` and `ArtifactSectionNotFoundError` are top-level peers of `OrchestrationError`** in the error hierarchy — not subclasses. This is intentional: artifact errors occur at a different abstraction level than runner errors.

---

## Dependency map

```
artifact/ depends on:     core/ (types, errors, logging), dapr/ (via lock and state store),
                          orchestration/runner.py (for ArtifactCollaborator)
artifact/ is imported by: orchestration/graph.py (via artifact_node())
artifact/ must NOT import from: memory/, tools/ (directly), safety/, evaluation/
```

---

## ADR references

- **ADR-015** — Artifact-centric collaboration: full design rationale (Specification Gap, STORM, Token Coherence, CodeCRDT, CAID)
