"""ArtifactCrew — wave-based parallel execution for artifact-centric collaboration."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Any

from nexus.core.errors import ArtifactConflictError
from nexus.core.logging import get_logger
from nexus.orchestration.artifact.collaborator import ArtifactCollaborator
from nexus.orchestration.artifact.schema import SchemaValidator
from nexus.orchestration.artifact.store import ArtifactStore
from nexus.orchestration.artifact.types import (
    Artifact,
    ArtifactEditResult,
    ArtifactSchema,
    ScopedContext,
)

if TYPE_CHECKING:
    from nexus.orchestration.runner import AgentRunner

_log = get_logger(__name__)


class ArtifactCrew:
    """Orchestrates multiple agents to collaboratively complete an artifact.

    Algorithm (CAID-inspired wave execution, arXiv 2603.21489):
    1. Load artifact schema → build dependency DAG
    2. Topological sort → execution waves (Kahn's algorithm)
       Wave 0: sections with no dependencies
       Wave 1: sections whose all deps are in wave 0, etc.
    3. For each wave (sequential between waves, parallel within a wave):
       a. asyncio.gather all sections in the wave
       b. Integration check after wave completes
       c. Failed sections retry up to max_retries, then add to failed list
    4. Return completed artifact or raise ArtifactConflictError with failed sections

    Args:
        agents: List of AgentRunner instances (round-robin assignment by default).
        collaborators: Parallel list of ArtifactCollaborator per agent.
        store: ArtifactStore for artifact state.
        max_retries: Max retry attempts per section on write failure.
        section_agent_map: Optional explicit {section_id → agent_id} assignment.
        tracer: Optional NexusTracer for OTEL spans.
    """

    def __init__(
        self,
        agents: list[AgentRunner],
        collaborators: list[ArtifactCollaborator],
        store: ArtifactStore,
        max_retries: int = 2,
        section_agent_map: dict[str, str] | None = None,
        tracer: Any | None = None,
    ) -> None:
        if len(agents) != len(collaborators):
            raise ValueError("agents and collaborators lists must have the same length")
        self._agents = agents
        self._collaborators = collaborators
        self._store = store
        self._max_retries = max_retries
        self._section_agent_map = section_agent_map or {}
        self._tracer = tracer
        self._validator = SchemaValidator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        artifact_id: str,
        task_description: str,
    ) -> Artifact:
        """Run full wave-based execution, returning the completed artifact.

        Args:
            artifact_id: ID of an already-created artifact.
            task_description: Overall task description injected into each agent prompt.

        Returns:
            Completed Artifact with all sections MERGED.

        Raises:
            ArtifactConflictError: If any sections fail after max_retries.
        """
        artifact = await self._store.load(artifact_id)
        waves = await self._build_waves(artifact.artifact_schema)

        _log.debug(
            "artifact_crew_run_start",
            artifact_id=artifact_id,
            num_sections=len(artifact.artifact_schema.sections),
            num_waves=len(waves),
            num_agents=len(self._agents),
        )

        if self._tracer:
            _emit_span(
                self._tracer,
                "artifact.crew.run",
                {
                    "artifact_id": artifact_id,
                    "num_sections": len(artifact.artifact_schema.sections),
                    "num_waves": len(waves),
                    "num_agents": len(self._agents),
                },
            )

        failed: list[str] = []
        completed: list[str] = []

        for wave_index, wave in enumerate(waves):
            _log.debug(
                "artifact_crew_wave_start",
                wave_index=wave_index,
                section_ids=wave,
            )

            if self._tracer:
                _emit_span(
                    self._tracer,
                    "artifact.crew.wave",
                    {"wave_index": wave_index, "section_ids": str(wave)},
                )

            wave_results = await self._run_wave(wave, artifact_id, task_description, wave_index)

            for result in wave_results:
                if result.success:
                    completed.append(result.section_id)
                else:
                    failed.append(result.section_id)

            if completed:
                integration_conflicts = await self._integration_check(artifact_id, completed)
                if integration_conflicts:
                    _log.warning(
                        "artifact_crew_integration_conflicts",
                        wave_index=wave_index,
                        conflicts=[c.model_dump() for c in integration_conflicts],
                    )

        if failed:
            raise ArtifactConflictError(
                f"ArtifactCrew: {len(failed)} section(s) failed after max_retries: {failed}",
                code="SECTIONS_FAILED",
                details={"failed_sections": failed},
            )

        return await self._store.complete_artifact(artifact_id)

    # ------------------------------------------------------------------
    # Wave building (Kahn's algorithm)
    # ------------------------------------------------------------------

    async def _build_waves(self, schema: ArtifactSchema) -> list[list[str]]:
        """Topological sort → execution waves using Kahn's algorithm.

        Args:
            schema: ArtifactSchema with sections and their dependencies.

        Returns:
            List of waves, each wave is a list of section_ids.

        Raises:
            ArtifactConflictError: code="CIRCULAR_DEPENDENCY" on cycle detection.
        """
        deps = schema.dependency_ids()
        in_degree: dict[str, int] = {s: len(deps[s]) for s in deps}
        queue: deque[str] = deque(s for s in deps if in_degree[s] == 0)
        waves: list[list[str]] = []
        visited = 0

        while queue:
            wave = list(queue)
            waves.append(wave)
            queue.clear()
            for node in wave:
                visited += 1
                for candidate, candidate_deps in deps.items():
                    if node in candidate_deps:
                        in_degree[candidate] -= 1
                        if in_degree[candidate] == 0 and candidate not in [
                            s for w in waves for s in w
                        ]:
                            queue.append(candidate)

        if visited < len(deps):
            raise ArtifactConflictError(
                "Circular dependency detected in artifact section DAG",
                code="CIRCULAR_DEPENDENCY",
                details={"section_ids": list(deps.keys())},
            )

        return waves

    # ------------------------------------------------------------------
    # Wave / section execution
    # ------------------------------------------------------------------

    async def _run_wave(
        self,
        wave: list[str],
        artifact_id: str,
        task_description: str,
        wave_index: int,
    ) -> list[ArtifactEditResult]:
        """Run all sections in a wave concurrently via asyncio.gather."""
        tasks = []
        for section_id in wave:
            agent, collaborator = self._assign_agent(section_id, wave_index)
            tasks.append(
                self._run_section(
                    section_id=section_id,
                    artifact_id=artifact_id,
                    agent=agent,
                    collaborator=collaborator,
                    task_description=task_description,
                )
            )
        results: list[ArtifactEditResult] = await asyncio.gather(*tasks)
        return list(results)

    async def _run_section(
        self,
        section_id: str,
        artifact_id: str,
        agent: AgentRunner,
        collaborator: ArtifactCollaborator,
        task_description: str,
        retry_count: int = 0,
    ) -> ArtifactEditResult:
        """Full claim → scoped_context → run → write → release lifecycle.

        Args:
            section_id: Section to complete.
            artifact_id: Target artifact.
            agent: AgentRunner to execute the section task.
            collaborator: ArtifactCollaborator bound to this agent.
            task_description: Overall task description.
            retry_count: Current retry attempt (0 = first attempt).

        Returns:
            ArtifactEditResult indicating success or failure.
        """
        _log.debug(
            "artifact_section_run",
            section_id=section_id,
            agent_id=collaborator._agent_id,
            retry_count=retry_count,
        )

        if self._tracer:
            _emit_span(
                self._tracer,
                "artifact.crew.section",
                {
                    "section_id": section_id,
                    "agent_id": collaborator._agent_id,
                    "retries": retry_count,
                },
            )

        claimed = await collaborator.claim_section(artifact_id, section_id)
        if not claimed:
            return ArtifactEditResult(
                success=False,
                op_type="claim",
                section_id=section_id,
                agent_id=collaborator._agent_id,
            )

        try:
            scoped = await collaborator.get_scoped_context(artifact_id, section_id)
            prompt = self._build_section_prompt(task_description, scoped)

            agent_def = _make_agent_def(collaborator._agent_id, section_id)
            session_id = f"{artifact_id}:{section_id}:{retry_count}"

            result = await agent.run(agent_def, prompt, session_id=session_id)
            content = result.output or ""

            write_result = await collaborator.write_section(artifact_id, section_id, content)

            if not write_result.success:
                conflict = write_result.conflict
                resolution = conflict.resolution if conflict else "reject"

                if resolution == "retry" and retry_count < self._max_retries:
                    await collaborator.release_section(artifact_id, section_id, mark_complete=False)
                    return await self._run_section(
                        section_id=section_id,
                        artifact_id=artifact_id,
                        agent=agent,
                        collaborator=collaborator,
                        task_description=task_description,
                        retry_count=retry_count + 1,
                    )

                await collaborator.release_section(artifact_id, section_id, mark_complete=False)
                return write_result

            await collaborator.release_section(artifact_id, section_id, mark_complete=True)
            return write_result

        except Exception as exc:
            _log.warning(
                "artifact_section_exception",
                section_id=section_id,
                agent_id=collaborator._agent_id,
                exc=str(exc),
            )
            await collaborator.release_section(artifact_id, section_id, mark_complete=False)
            return ArtifactEditResult(
                success=False,
                op_type="write",
                section_id=section_id,
                agent_id=collaborator._agent_id,
            )

    async def _integration_check(
        self,
        artifact_id: str,
        completed_section_ids: list[str],
    ) -> list[Any]:
        """After each wave: re-validate all completed sections against the artifact snapshot.

        Args:
            artifact_id: Target artifact.
            completed_section_ids: Sections to re-validate.

        Returns:
            List of SectionConflict found (empty = clean).
        """
        conflicts = []
        artifact = await self._store.get_snapshot(artifact_id)
        for sid in completed_section_ids:
            section = artifact.sections.get(sid)
            schema = artifact.artifact_schema.get_section(sid)
            if section is None or schema is None or section.content is None:
                continue
            conflict = self._validator.validate(section, schema)
            if conflict:
                conflicts.append(conflict)
        return conflicts

    # ------------------------------------------------------------------
    # Agent assignment and prompt building
    # ------------------------------------------------------------------

    def _assign_agent(
        self, section_id: str, wave_index: int
    ) -> tuple[AgentRunner, ArtifactCollaborator]:
        """Return (agent, collaborator) for this section.

        Uses section_agent_map if provided, otherwise round-robin by wave_index.

        Args:
            section_id: Section to assign.
            wave_index: Index within the wave for round-robin fallback.

        Returns:
            (AgentRunner, ArtifactCollaborator) pair.
        """
        if section_id in self._section_agent_map:
            target_id = self._section_agent_map[section_id]
            for i, c in enumerate(self._collaborators):
                if c._agent_id == target_id:
                    return self._agents[i], c

        idx = wave_index % len(self._agents)
        return self._agents[idx], self._collaborators[idx]

    def _build_section_prompt(self, task_description: str, scoped: ScopedContext) -> str:
        """Build the task prompt an agent receives for its section.

        Args:
            task_description: Overall artifact goal.
            scoped: CAID scoped context for this section.

        Returns:
            Complete prompt string.
        """
        lines = [
            f"Overall artifact goal: {task_description}",
            "",
            f"Your section: {scoped.section_schema.section_id}",
            f"What it must contain: {scoped.section_schema.description}",
            f"Content type: {scoped.section_schema.content_type.value}",
        ]

        if scoped.section_schema.required_fields:
            lines.append(f"Required fields: {', '.join(scoped.section_schema.required_fields)}")

        if scoped.section_schema.validation_rules:
            lines.append("Validation rules:")
            for rule in scoped.section_schema.validation_rules:
                lines.append(f"  - {rule}")

        if scoped.completed_dependencies:
            lines.append("")
            lines.append("Completed context (summaries only):")
            for dep_id, summary in scoped.completed_dependencies.items():
                lines.append(f"  - {dep_id}: {summary}")

        if scoped.global_constraints:
            lines.append("")
            lines.append("Global constraints:")
            for constraint in scoped.global_constraints:
                lines.append(f"  - {constraint}")

        if scoped.approach_hint:
            lines.append("")
            lines.append(f"Approach hint: {scoped.approach_hint}")

        lines.append("")
        lines.append("Write ONLY the content for your section. Do not include any other sections.")

        return "\n".join(lines)


# ------------------------------------------------------------------
# Module helpers
# ------------------------------------------------------------------


def _emit_span(tracer: Any, name: str, attrs: dict[str, Any]) -> None:
    """Emit a single OTEL span with the given attributes. Silently suppresses errors."""
    import contextlib

    with contextlib.suppress(Exception), tracer._tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            span.set_attribute(k, v)


def _make_agent_def(agent_id: str, section_id: str) -> Any:
    """Build a minimal AgentDefinition for section execution."""
    from nexus.core.types import AgentDefinition

    return AgentDefinition(
        name=agent_id,
        model="claude-sonnet-4-6",
        system_prompt=(
            f"You are an expert writer responsible for writing section '{section_id}' "
            "of a collaborative artifact. Follow all instructions exactly."
        ),
    )
