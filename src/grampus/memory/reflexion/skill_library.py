"""SkillLibrary — post-success skill extraction and lifecycle management (SAGE)."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

from grampus.core.logging import get_logger
from grampus.memory.reflexion.types import SkillExtractionResult
from grampus.memory.types import Procedure, ProcedureStep, ProcedureType

if TYPE_CHECKING:
    from grampus.core.types import AgentDefinition, ExecutionResult
    from grampus.memory.embeddings import EmbeddingService
    from grampus.memory.procedural import ProceduralMemory

_log = get_logger(__name__)

_DEMOTION_MIN_USES = 5
_DEMOTION_THRESHOLD = 0.4
_VALIDATION_THRESHOLD = 0.6
_VALIDATION_MIN_USES = 3

EXTRACT_SYSTEM_PROMPT = (
    "You are an agent skill extractor. "
    "Given a successfully completed task and its execution trace, "
    "extract a reusable, parameterized skill if the approach is generalizable. "
    "A skill must be: (1) named, (2) described in one sentence, "
    "(3) broken into concrete steps with tool names where applicable, "
    "(4) tagged with 1–3 domain tags. "
    "If the task was too specific to generalize, reply with "
    '{"extractable": false}. '
    "Otherwise reply with valid JSON only."
)

EXTRACT_USER_TEMPLATE = (
    "Task: {task}\n\n"
    "Tool calls made (in order):\n{tool_sequence}\n\n"
    "Final output:\n{output}\n\n"
    'Extract a reusable skill or reply with {{"extractable": false}}:'
)


class SkillLibrary:
    """Tier 2 of F1: post-success skill extraction (SAGE, arXiv 2512.17102).

    After a task succeeds, attempts to extract a reusable parameterized skill
    from the execution trace. Stores validated skills in ProceduralMemory as
    type=SKILL. Surfaces top matching skills as approach hints on future tasks.

    Skill lifecycle:
    - New skills start with validated=False (not yet surfaced as hints).
    - After ≥3 successful uses: validated=True (surfaced prominently).
    - After ≥5 uses with success_rate < 0.4: validated=False (demoted).
    - After ≥5 uses with success_rate < 0.2: deleted entirely.

    Args:
        procedural_memory: Per-agent ProceduralMemory store.
        embedding_service: For embedding task descriptions.
        min_extraction_quality: Minimum confidence to store a skill. Default 0.5.
        max_skills: Prune lowest-performing skills when count exceeds this. Default 100.
    """

    def __init__(
        self,
        procedural_memory: ProceduralMemory,
        embedding_service: EmbeddingService,
        *,
        min_extraction_quality: float = 0.5,
        max_skills: int = 100,
    ) -> None:
        self._mem = procedural_memory
        self._embed = embedding_service
        self._min_quality = min_extraction_quality
        self._max_skills = max_skills

    async def observe_success(
        self,
        agent_def: AgentDefinition,
        user_input: str,
        result: ExecutionResult,
        model_client: Any,
    ) -> SkillExtractionResult:
        """Extract a skill from a successful execution if the approach is generalizable.

        Never raises — returns extracted=False on any error.
        """
        try:
            tool_sequence = self._build_tool_sequence(result)
            output_snippet = (result.output or "")[:500]
            user_prompt = EXTRACT_USER_TEMPLATE.format(
                task=user_input[:300],
                tool_sequence=tool_sequence,
                output=output_snippet,
            )

            raw = await _call_llm(
                model_client,
                system=EXTRACT_SYSTEM_PROMPT,
                user=user_prompt,
                model=agent_def.model,
                temperature=0.2,
                max_tokens=400,
            )
            if not raw:
                return SkillExtractionResult(extracted=False, error="empty model response")

            data = json.loads(raw)
            if not data.get("extractable", True):
                return SkillExtractionResult(extracted=False)

            steps = [
                ProcedureStep(
                    action=s.get("action", ""),
                    tool_name=s.get("tool_name"),
                )
                for s in data.get("steps", [])
            ]
            domain_tags: list[str] = data.get("domain_tags", [])
            name: str = data.get("name", f"skill:{uuid.uuid4().hex[:8]}")
            description: str = data.get("description", "")

            embedding = await self._embed.embed(user_input)
            proc = Procedure(
                id=str(uuid.uuid4()),
                name=name,
                description=description,
                steps=steps,
                trigger_conditions=[user_input[:200]],
                agent_id=self._mem._agent_id,
                embedding=embedding,
                procedure_type=ProcedureType.SKILL,
                confidence=1.0,
                metadata={"validated": False, "domain_tags": domain_tags},
            )
            await self._mem.store(proc)
            await self._run_lifecycle()

            _log.debug("skill_extracted", agent=agent_def.name, skill=name, id=proc.id)
            return SkillExtractionResult(extracted=True, procedure_id=proc.id, skill_name=name)

        except Exception as exc:  # noqa: BLE001
            _log.warning("skill_observe_success_error", error=str(exc))
            return SkillExtractionResult(extracted=False, error=str(exc))

    async def get_approach_hints(
        self,
        task: str,
        model_client: Any,
        *,
        top_k: int = 2,
        validated_only: bool = True,
    ) -> list[Procedure]:
        """Return the top-k most relevant skills for this task.

        Returns an empty list on any error.
        """
        try:
            embedding = await self._embed.embed(task)
            candidates = await self._mem.find_similar(
                embedding,
                procedure_type=ProcedureType.SKILL,
                top_k=top_k * 5,
            )
            if validated_only:
                candidates = [c for c in candidates if c.metadata.get("validated")]
            return candidates[:top_k]
        except Exception as exc:  # noqa: BLE001
            _log.warning("skill_get_hints_error", error=str(exc))
            return []

    async def record_skill_outcome(self, procedure_id: str, *, success: bool) -> None:
        """Update success/failure count and run lifecycle logic.

        Called after a skill was used as a hint and the task completed.
        """
        await self._mem.record_outcome(procedure_id, success=success)
        proc = await self._mem.get(procedure_id)
        if proc is None:
            return

        total_uses = proc.success_count + proc.failure_count
        success_rate = proc.success_count / total_uses if total_uses > 0 else 1.0

        # Promote to validated after enough successes
        if (
            not proc.metadata.get("validated")
            and proc.success_count >= _VALIDATION_MIN_USES
            and success_rate >= _VALIDATION_THRESHOLD
        ):
            updated_meta = {**proc.metadata, "validated": True}
            updated = proc.model_copy(update={"metadata": updated_meta})
            await self._mem._save_procedure(updated)
            _log.debug("skill_promoted", procedure_id=procedure_id)
            return

        if total_uses >= _DEMOTION_MIN_USES:
            if success_rate < 0.2:
                await self._mem.delete(procedure_id)
                _log.debug("skill_deleted", procedure_id=procedure_id, success_rate=success_rate)
            elif success_rate < _DEMOTION_THRESHOLD:
                updated_meta = {**proc.metadata, "validated": False}
                updated = proc.model_copy(update={"metadata": updated_meta})
                await self._mem._save_procedure(updated)
                _log.debug("skill_demoted", procedure_id=procedure_id, success_rate=success_rate)

    async def run_sequential(
        self,
        tasks: list[str],
        agent_def: AgentDefinition,
        runner: Any,
        *,
        session_prefix: str = "sequential",
    ) -> list[ExecutionResult]:
        """SAGE-inspired sequential rollout: skills learned from task t accelerate task t+1.

        For each task:
        1. Fetch current approach hints (including newly validated skills).
        2. Inject hints into agent_def.system_prompt.
        3. Run runner.run().
        4. Call observe_success() or log failure.

        Returns list of ExecutionResults (one per task).
        """
        results: list[ExecutionResult] = []
        for i, task in enumerate(tasks):
            session_id = f"{session_prefix}:{i}:{uuid.uuid4().hex[:8]}"
            hints = await self.get_approach_hints(task, runner._model_client)
            hint_context = self.format_hints_as_context(hints)
            effective_prompt = agent_def.system_prompt or ""
            if hint_context:
                effective_prompt = hint_context + (
                    "\n\n" + effective_prompt if effective_prompt else ""
                )
            patched_def = agent_def.model_copy(update={"system_prompt": effective_prompt})

            try:
                result = await runner.run(patched_def, task, session_id=session_id)
                results.append(result)
                from grampus.core.types import AgentStatus

                if result.status == AgentStatus.COMPLETED:
                    import contextlib

                    with contextlib.suppress(Exception):
                        await self.observe_success(agent_def, task, result, runner._model_client)
            except Exception as exc:  # noqa: BLE001
                _log.warning("sequential_task_failed", task_index=i, error=str(exc))

        return results

    def format_hints_as_context(self, skills: list[Procedure]) -> str:
        """Format skills as a system message prefix.

        Returns empty string if skills is empty.
        """
        if not skills:
            return ""
        lines = ["Reusable approaches for similar tasks:"]
        for skill in skills:
            step_parts = " ".join(f"{j}. {s.action}" for j, s in enumerate(skill.steps, 1))
            lines.append(f"[{skill.name}]: {skill.description}")
            if step_parts:
                lines.append(f"  Steps: {step_parts}")
        return "\n".join(lines)

    def _build_tool_sequence(self, result: ExecutionResult) -> str:
        """Extract ordered tool call names + args summary from ExecutionResult.messages."""
        parts: list[str] = []
        for msg in result.messages:
            for tc in msg.tool_calls:
                args_summary = str(tc.arguments)[:100]
                parts.append(f"{tc.name}({args_summary})")
        return "\n".join(parts) if parts else "(no tool calls)"

    async def _run_lifecycle(self) -> None:
        """Prune lowest-confidence skills when count > max_skills."""
        all_skills = await self._mem.query_by_type(ProcedureType.SKILL)
        if len(all_skills) <= self._max_skills:
            return

        # Sort ascending by success_rate (lowest first), delete excess
        def _rate(p: Procedure) -> float:
            total = p.success_count + p.failure_count
            return p.success_count / total if total > 0 else 0.0

        all_skills.sort(key=_rate)
        to_delete = all_skills[: len(all_skills) - self._max_skills]
        for proc in to_delete:
            await self._mem.delete(proc.id)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


async def _call_llm(
    model_client: Any,
    *,
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call model_client.complete with a system + user message pair."""
    from grampus.core.types import Message, Role

    messages = [
        Message(role=Role.SYSTEM, content=system),
        Message(role=Role.USER, content=user),
    ]
    response = await model_client.complete(
        messages=messages,
        model=model,
        temperature=temperature,
    )
    return (response.content or "").strip()
