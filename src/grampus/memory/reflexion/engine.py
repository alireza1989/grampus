"""ReflexionEngine — post-failure verbal reflection stored in ProceduralMemory."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

from grampus.core.logging import get_logger
from grampus.memory.reflexion.types import ReflexionHookResult
from grampus.memory.types import Procedure, ProcedureStep, ProcedureType

if TYPE_CHECKING:
    from grampus.core.types import AgentDefinition, AgentState
    from grampus.memory.embeddings import EmbeddingService
    from grampus.memory.procedural import ProceduralMemory

_log = get_logger(__name__)

_REFLECT_SYSTEM = (
    "You are a self-improving AI agent. You just failed a task. "
    "Write a concise, actionable verbal reflection (2–4 sentences) explaining: "
    "(1) what went wrong, (2) what you should do differently next time. "
    "Be specific — mention tool names, reasoning steps, or assumptions that failed. "
    "Reply with only the reflection text, no preamble."
)

_QUALITY_SYSTEM = (
    "Rate the quality of this agent self-reflection on a scale from 0.0 to 1.0. "
    "High quality (>0.7): specific, actionable, names concrete failure points. "
    "Low quality (<0.3): vague, generic, or unhelpful. "
    'Reply with JSON only: {"quality": <float>}'
)


class ReflexionEngine:
    """Tier 1 of F1: post-failure verbal reflection (Reflexion, NeurIPS 2023).

    On each task failure, asks the model to verbalize what went wrong and stores
    the reflection as a REFLECTION-typed Procedure in ProceduralMemory. A second
    LLM call rates quality (ME-ICPO, arXiv 2603.01335); low-quality reflections
    are stored but not surfaced as hints.

    Reflection lifecycle:
    - Created with confidence = quality_confidence score from the rating call.
    - Reflections with confidence < quality_threshold are stored but not surfaced.
    - When max_reflections is exceeded, oldest reflections are pruned.

    Args:
        procedural_memory: Per-agent store where reflections are saved.
        embedding_service: Used to embed the task description for similarity lookup.
        max_reflections: Prune oldest reflections when this count is exceeded. Default 50.
        quality_threshold: Minimum confidence to surface a reflection as a hint. Default 0.3.
    """

    def __init__(
        self,
        procedural_memory: ProceduralMemory,
        embedding_service: EmbeddingService,
        *,
        max_reflections: int = 50,
        quality_threshold: float = 0.3,
    ) -> None:
        self._mem = procedural_memory
        self._embed = embedding_service
        self._max_reflections = max_reflections
        self._quality_threshold = quality_threshold

    async def observe_failure(
        self,
        agent_def: AgentDefinition,
        user_input: str,
        exc: BaseException,
        state: AgentState,
        model_client: Any,
    ) -> ReflexionHookResult:
        """Generate and store a verbal reflection from a task failure.

        Steps:
        1. Build a failure summary from exc + last N messages.
        2. Call model_client.complete() (temp=0.3, max_tokens=300) for reflection text.
        3. Call model_client.complete() (temp=0.0, max_tokens=60) to rate quality → confidence.
        4. Store a REFLECTION Procedure with confidence = quality score.
        5. Prune oldest reflections if over max_reflections.

        Never raises — returns stored=False on any error.
        """
        try:
            failure_summary = _build_failure_summary(user_input, exc, state)
            reflection_text = await _call_llm(
                model_client,
                system=_REFLECT_SYSTEM,
                user=failure_summary,
                model=agent_def.model,
                temperature=0.3,
                max_tokens=300,
            )
            if not reflection_text:
                return ReflexionHookResult(stored=False, error="empty reflection from model")

            quality = await _rate_quality(model_client, reflection_text, agent_def.model)

            embedding = await self._embed.embed(user_input)
            proc = Procedure(
                id=str(uuid.uuid4()),
                name=f"reflection:{uuid.uuid4().hex[:8]}",
                description=reflection_text,
                steps=[ProcedureStep(action=reflection_text)],
                trigger_conditions=[user_input[:200]],
                agent_id=self._mem._agent_id,
                embedding=embedding,
                procedure_type=ProcedureType.REFLECTION,
                confidence=quality,
            )
            await self._mem.store(proc)
            await self._prune_reflections()

            _log.debug(
                "reflexion_stored",
                agent=agent_def.name,
                quality=quality,
                procedure_id=proc.id,
            )
            return ReflexionHookResult(
                stored=True,
                procedure_id=proc.id,
                quality_confidence=quality,
            )
        except Exception as exc_inner:  # noqa: BLE001
            _log.warning("reflexion_observe_failure_error", error=str(exc_inner))
            return ReflexionHookResult(stored=False, error=str(exc_inner))

    async def get_relevant_reflections(
        self,
        query: str,
        model_client: Any,
        *,
        top_k: int = 3,
    ) -> list[Procedure]:
        """Return the top-k most similar reflections above quality_threshold.

        Returns an empty list on any error.
        """
        try:
            embedding = await self._embed.embed(query)
            results = await self._mem.find_similar(
                embedding,
                procedure_type=ProcedureType.REFLECTION,
                top_k=top_k * 3,
            )
            surfaced = [r for r in results if r.confidence >= self._quality_threshold]
            return surfaced[:top_k]
        except Exception as exc:  # noqa: BLE001
            _log.warning("reflexion_get_relevant_error", error=str(exc))
            return []

    def format_as_context(self, reflections: list[Procedure]) -> str:
        """Format reflections as a numbered system message prefix.

        Returns an empty string if *reflections* is empty.
        """
        if not reflections:
            return ""
        lines = ["Lessons from past failures:"]
        for i, r in enumerate(reflections, 1):
            lines.append(f"{i}. {r.description}")
        return "\n".join(lines)

    async def _prune_reflections(self) -> None:
        """Delete oldest reflections when count exceeds max_reflections."""
        all_refs = await self._mem.query_by_type(ProcedureType.REFLECTION)
        if len(all_refs) <= self._max_reflections:
            return
        # Sort by last_used ascending (oldest first), fall back to id for stability
        all_refs.sort(key=lambda p: p.last_used or p.id)
        to_delete = all_refs[: len(all_refs) - self._max_reflections]
        for proc in to_delete:
            await self._mem.delete(proc.id)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _build_failure_summary(user_input: str, exc: BaseException, state: AgentState) -> str:
    recent_msgs = state.messages[-6:] if len(state.messages) > 6 else state.messages
    msg_summary = "\n".join(f"[{m.role}]: {(m.content or '')[:200]}" for m in recent_msgs)
    return (
        f"Task: {user_input[:300]}\n\n"
        f"Error: {type(exc).__name__}: {str(exc)[:200]}\n\n"
        f"Last messages:\n{msg_summary}"
    )


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


async def _rate_quality(model_client: Any, reflection: str, model: str) -> float:
    """Ask the model to rate reflection quality. Returns 0.5 on any parse failure."""
    try:
        raw = await _call_llm(
            model_client,
            system=_QUALITY_SYSTEM,
            user=f"Reflection:\n{reflection}",
            model=model,
            temperature=0.0,
            max_tokens=60,
        )
        data = json.loads(raw)
        score = float(data.get("quality", 0.5))
        return max(0.0, min(1.0, score))
    except Exception:  # noqa: BLE001
        return 0.5
