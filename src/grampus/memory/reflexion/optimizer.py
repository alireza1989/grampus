"""PromptOptimizer — DSPy-style prompt optimization driven by EvalSuite results."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from grampus.core.logging import get_logger
from grampus.memory.reflexion.types import OptimizationCandidate, OptimizationResult

if TYPE_CHECKING:
    from grampus.core.types import AgentDefinition
    from grampus.evaluation.prompt_versions import PromptVersionManager
    from grampus.evaluation.suite import EvalSuite
    from grampus.memory.reflexion.engine import ReflexionEngine
    from grampus.memory.reflexion.skill_library import SkillLibrary

_log = get_logger(__name__)

_REWRITE_SYSTEM = (
    "You are a prompt engineer improving an AI agent's system prompt. "
    "Given failing eval cases and the current system prompt, rewrite the prompt "
    "to directly address the failure patterns. "
    "Reply with only the rewritten system prompt text, no preamble."
)


class PromptOptimizer:
    """DSPy-style prompt optimization driven by EvalSuite results.

    Proposes 3 candidate system prompt mutations, evaluates each via EvalSuite.run(),
    and stores the best as a new PromptVersion if it improves on the baseline.

    Three mutation strategies:
    1. ``append_reflection`` — append the single highest-confidence reflection.
    2. ``append_skill`` — append the single highest-performing validated skill hint.
    3. ``rewrite_failures`` — ask the LLM to rewrite based on failing eval cases.

    Args:
        reflexion_engine: For fetching top reflections.
        skill_library: For fetching top skills.
        prompt_manager: PromptVersionManager for the agent.
        eval_runner: EvalSuite instance pre-loaded with eval cases.
        model_client: For the rewrite_failures mutation.
        improvement_threshold: Only store new version if improvement > this. Default 0.05.
    """

    def __init__(
        self,
        reflexion_engine: ReflexionEngine,
        skill_library: SkillLibrary,
        prompt_manager: PromptVersionManager,
        eval_runner: EvalSuite,
        model_client: Any,
        *,
        improvement_threshold: float = 0.05,
    ) -> None:
        self._reflexion = reflexion_engine
        self._skills = skill_library
        self._prompt_mgr = prompt_manager
        self._eval = eval_runner
        self._model_client = model_client
        self._threshold = improvement_threshold

    async def optimize(
        self,
        agent_def: AgentDefinition,
        runner: Any,
    ) -> OptimizationResult:
        """Run the full optimization cycle.

        Never raises — returns improved=False on any error.
        """
        try:
            baseline_suite = await self._eval.run()
            baseline_score = baseline_suite.pass_rate

            failing_cases = [cr for cr in baseline_suite.case_results if not cr.passed]

            candidates = await self._build_candidates(agent_def, failing_cases)
            if not candidates:
                return OptimizationResult(
                    improved=False,
                    original_score=baseline_score,
                    best_score=baseline_score,
                    candidates=[],
                )

            best_candidate: OptimizationCandidate | None = None
            best_score = baseline_score

            for candidate in candidates:
                score = await self._run_with_prompt(agent_def, runner, candidate.prompt)
                candidate.eval_score = score
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            if best_candidate is not None and best_score > baseline_score + self._threshold:
                new_version = self._next_version_string()
                self._prompt_mgr.register(
                    new_version,
                    best_candidate.prompt,
                    notes=f"Auto-optimized via {best_candidate.strategy}; "
                    f"score {baseline_score:.3f} → {best_score:.3f}",
                )
                self._prompt_mgr.activate(new_version)
                _log.info(
                    "prompt_optimized",
                    agent=agent_def.name,
                    strategy=best_candidate.strategy,
                    baseline=baseline_score,
                    best=best_score,
                    version=new_version,
                )
                return OptimizationResult(
                    improved=True,
                    original_score=baseline_score,
                    best_score=best_score,
                    best_strategy=best_candidate.strategy,
                    new_version=new_version,
                    candidates=candidates,
                )

            return OptimizationResult(
                improved=False,
                original_score=baseline_score,
                best_score=best_score,
                candidates=candidates,
            )

        except Exception as exc:  # noqa: BLE001
            _log.warning("prompt_optimizer_error", error=str(exc))
            return OptimizationResult(
                improved=False,
                original_score=0.0,
                best_score=0.0,
                error=str(exc),
            )

    async def _build_candidates(
        self,
        agent_def: AgentDefinition,
        failing_cases: list[Any],
    ) -> list[OptimizationCandidate]:
        """Build 3 candidates. Each is a full system prompt string + metadata."""
        base = agent_def.system_prompt or ""
        candidates: list[OptimizationCandidate] = []

        # Strategy 1: append_reflection
        try:
            reflections = await self._reflexion.get_relevant_reflections(base, self._model_client)
            if reflections:
                note = f"\nNote from past experience: {reflections[0].description}"
                candidates.append(
                    OptimizationCandidate(
                        strategy="append_reflection",
                        prompt=base + note,
                    )
                )
        except Exception:  # noqa: BLE001
            pass

        # Strategy 2: append_skill
        try:
            skills = await self._skills.get_approach_hints(base, self._model_client)
            if skills:
                hint = f"\nFor tasks like this, a proven approach is: {skills[0].description}"
                candidates.append(
                    OptimizationCandidate(
                        strategy="append_skill",
                        prompt=base + hint,
                    )
                )
        except Exception:  # noqa: BLE001
            pass

        # Strategy 3: rewrite_failures
        try:
            if failing_cases:
                cases_text = "\n".join(
                    f"- {cr.case_name}: {cr.error or 'assertion failed'}"
                    for cr in failing_cases[:5]
                )
                user_msg = (
                    f"Current system prompt:\n{base}\n\n"
                    f"Failing cases:\n{cases_text}\n\n"
                    "Rewrite the system prompt to address these failures:"
                )
                rewritten = await _call_llm(
                    self._model_client,
                    system=_REWRITE_SYSTEM,
                    user=user_msg,
                    model=agent_def.model,
                    temperature=0.3,
                    max_tokens=600,
                )
                if rewritten:
                    candidates.append(
                        OptimizationCandidate(
                            strategy="rewrite_failures",
                            prompt=rewritten,
                        )
                    )
        except Exception:  # noqa: BLE001
            pass

        return candidates

    async def _run_with_prompt(
        self,
        agent_def: AgentDefinition,
        runner: Any,
        prompt: str,
    ) -> float:
        """Temporarily override system_prompt and run EvalSuite. Returns pass_rate."""
        patched = agent_def.model_copy(update={"system_prompt": prompt})
        original_def = self._eval._agent_def
        self._eval._agent_def = patched
        try:
            suite_result = await self._eval.run()
            return suite_result.pass_rate
        finally:
            self._eval._agent_def = original_def

    def _next_version_string(self) -> str:
        """Compute next semver from existing versions, e.g. '1.0.0' → '1.1.0'."""
        versions = self._prompt_mgr.history()
        if not versions:
            return "1.0.0"
        latest = versions[-1].version
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", latest)
        if not match:
            return "1.0.0"
        major, minor, patch = int(match[1]), int(match[2]), int(match[3])
        return f"{major}.{minor + 1}.{patch}"


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
