"""ModelClient abstract base class and ModelResponse type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from nexus.core.types import Message, TokenUsage, ToolCall, ToolDefinition


class ModelResponse(BaseModel):
    """Structured response from any model provider."""

    content: str | None
    tool_calls: list[ToolCall]
    token_usage: TokenUsage
    model: str
    stop_reason: str


class ModelClient(ABC):
    """Abstract base for LLM provider clients."""

    @abstractmethod
    async def complete(
        self,
        *,
        messages: list[Message],
        model: str,
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> ModelResponse:
        """Send a completion request and return the full response."""

    @abstractmethod
    def stream(
        self,
        *,
        messages: list[Message],
        model: str,
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion, yielding text chunks."""
