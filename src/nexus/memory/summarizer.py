"""Conversation summarization strategies for working memory compression."""

from __future__ import annotations

from enum import StrEnum

from nexus.core.models.base import ModelClient
from nexus.core.types import Message, Role
from nexus.memory.token_counter import TokenCounter

_SUMMARIZE_PROMPT = (
    "The following is a conversation history. Produce a concise summary that "
    "preserves all key facts, decisions, names, and context. Write only the "
    "summary — no preamble.\n\n{conversation}"
)


class SummarizationStrategy(StrEnum):
    """Strategy used to compress a conversation that exceeds the token budget."""

    TRUNCATE = "truncate"
    SUMMARIZE = "summarize"
    HYBRID = "hybrid"


class Summarizer:
    """Compresses message lists using a configurable strategy.

    Args:
        model_client: LLM client used by SUMMARIZE and HYBRID strategies.
        model_name: Model to call for summarization.
        strategy: One of TRUNCATE, SUMMARIZE, or HYBRID.
        token_counter: Counter used to measure message sizes.
        keep_last_n: Number of most-recent messages kept verbatim in HYBRID mode.
    """

    def __init__(
        self,
        model_client: ModelClient,
        model_name: str,
        strategy: SummarizationStrategy,
        token_counter: TokenCounter,
        *,
        keep_last_n: int = 10,
    ) -> None:
        self._client = model_client
        self._model = model_name
        self._strategy = strategy
        self._counter = token_counter
        self._keep_last_n = keep_last_n

    async def summarize(self, messages: list[Message], max_tokens: int) -> list[Message]:
        """Return a compressed version of *messages*.

        TRUNCATE returns unchanged if already within *max_tokens*.
        SUMMARIZE and HYBRID always compress — if you chose those strategies
        you want a summary regardless of current size.
        """
        if not messages:
            return []

        match self._strategy:
            case SummarizationStrategy.TRUNCATE:
                if self._counter.count_messages(messages) <= max_tokens:
                    return messages
                return self._truncate(messages, max_tokens)
            case SummarizationStrategy.SUMMARIZE:
                return await self._summarize(messages)
            case SummarizationStrategy.HYBRID:
                return await self._hybrid(messages)

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _truncate(self, messages: list[Message], max_tokens: int) -> list[Message]:
        """Drop oldest non-SYSTEM messages until within budget."""
        system = [m for m in messages if m.role == Role.SYSTEM]
        non_system = [m for m in messages if m.role != Role.SYSTEM]
        # Drop from the front until we fit
        while non_system and self._counter.count_messages(system + non_system) > max_tokens:
            non_system.pop(0)
        return system + non_system

    async def _summarize(self, messages: list[Message]) -> list[Message]:
        """Replace entire history with one LLM-generated summary message."""
        system = [m for m in messages if m.role == Role.SYSTEM]
        non_system = [m for m in messages if m.role != Role.SYSTEM]
        summary_msg = await self._call_llm_for_summary(non_system)
        return system + [summary_msg]

    async def _hybrid(self, messages: list[Message]) -> list[Message]:
        """Summarize old messages, keep the most recent *keep_last_n* verbatim."""
        system = [m for m in messages if m.role == Role.SYSTEM]
        non_system = [m for m in messages if m.role != Role.SYSTEM]
        if len(non_system) <= self._keep_last_n:
            return messages
        to_summarize = non_system[: -self._keep_last_n]
        recent = non_system[-self._keep_last_n :]
        summary_msg = await self._call_llm_for_summary(to_summarize)
        return system + [summary_msg] + recent

    async def _call_llm_for_summary(self, messages: list[Message]) -> Message:
        conversation = _format_for_summary(messages)
        prompt = _SUMMARIZE_PROMPT.format(conversation=conversation)
        response = await self._client.complete(
            messages=[Message(role=Role.USER, content=prompt)],
            model=self._model,
            temperature=0.0,
        )
        return Message(
            role=Role.ASSISTANT,
            content=response.content,
            metadata={
                "is_summary": True,
                "original_count": len(messages),
            },
        )


def _format_for_summary(messages: list[Message]) -> str:
    lines: list[str] = []
    for m in messages:
        role = m.role.upper()
        content = m.content or ""
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
