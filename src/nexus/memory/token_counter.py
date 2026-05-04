"""Model-aware token counting using tiktoken with character-based fallback."""

from __future__ import annotations

import json
import math

import tiktoken

from nexus.core.types import Message

# Map model name prefixes/substrings to tiktoken encoding names.
# Claude uses cl100k_base as the closest public approximation.
_MODEL_ENCODING: dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5": "cl100k_base",
    "claude": "cl100k_base",
    "text-embedding-ada": "cl100k_base",
}

_CHARS_PER_TOKEN = 4


class TokenCounter:
    """Count tokens for messages using tiktoken or a character-based fallback.

    Args:
        model: LLM model name used to select the correct tiktoken encoding.
            Unrecognised models fall back to ``ceil(chars / 4)`` estimation.
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        self._model = model
        encoding_name = _resolve_encoding(model)
        self.encoding_name = encoding_name
        if encoding_name == "fallback":
            self._encoder: tiktoken.Encoding | None = None
        else:
            self._encoder = tiktoken.get_encoding(encoding_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count_text(self, text: str) -> int:
        """Return the token count for a plain string."""
        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        return math.ceil(len(text) / _CHARS_PER_TOKEN)

    def count_messages(self, messages: list[Message]) -> int:
        """Return the total token count across all messages."""
        return sum(self._count_one(m) for m in messages)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_one(self, message: Message) -> int:
        parts: list[str] = [message.role]
        if message.content:
            parts.append(message.content)
        for tc in message.tool_calls:
            parts.append(tc.name)
            parts.append(json.dumps(tc.arguments))
        for tr in message.tool_results:
            if tr.output is not None:
                parts.append(str(tr.output))
            if tr.error:
                parts.append(tr.error)
        return self.count_text(" ".join(parts))


def _resolve_encoding(model: str) -> str:
    """Return a tiktoken encoding name for *model*, or ``'fallback'``."""
    lower = model.lower()
    for prefix, enc in _MODEL_ENCODING.items():
        if prefix in lower:
            return enc
    return "fallback"
