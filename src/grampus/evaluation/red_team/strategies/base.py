"""BaseAttackStrategy: ABC all strategies implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from grampus.evaluation.red_team.types import (
    AttackCategory,
    AttackPayload,
    RedTeamTargetConfig,
)


class BaseAttackStrategy(ABC):
    """
    Abstract base for all red-team attack strategies.

    A strategy generates a list of AttackPayloads for a given target.
    The RedTeamRunner (Part 2) submits each payload to the target agent
    and passes the response to RedTeamJudge.

    Strategies must be:
    - Stateless: generate() must not mutate self
    - Deterministic for a given seed (for reproducibility)
    - Safe to call concurrently

    Args:
        model_client: Optional LLM for strategies that need generation.
            Rule-based strategies (PromptInjection, Jailbreak) work without it.
    """

    def __init__(self, model_client: Any | None = None) -> None:
        self._model = model_client

    @property
    @abstractmethod
    def category(self) -> AttackCategory:
        """The OWASP category this strategy targets."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short slug identifier, e.g. 'prompt_injection'."""

    @abstractmethod
    async def generate(
        self,
        target: RedTeamTargetConfig,
        count: int = 5,
    ) -> list[AttackPayload]:
        """
        Generate up to `count` attack payloads for this target.

        Must never raise — return [] on error.
        May use target.system_prompt and target.available_tools
        to tailor attacks.
        """
