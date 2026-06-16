"""In-session working memory with auto-summarization and durable audit log."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel

from grampus.core.logging import get_logger
from grampus.core.types import Message
from grampus.memory.summarizer import Summarizer
from grampus.memory.token_counter import TokenCounter

_log = get_logger(__name__)

_DEFAULT_THRESHOLD = 0.8
_DEFAULT_MAX_TOKENS = 100_000


class _WindowState(BaseModel):
    messages: list[Message]


class WorkingMemory:
    """In-session message buffer with auto-summarization and durable audit log.

    Two separate Dapr state keys are maintained:
    - ``window`` — the live, possibly compressed, message list sent to the LLM.
    - ``history`` — append-only full audit log, never compressed or deleted.

    Summarization is triggered inside ``add()`` when the live window exceeds
    ``threshold_fraction * max_tokens``.  An ``asyncio.Lock`` prevents races
    when the same session receives concurrent messages.
    """

    def __init__(
        self,
        state_store: Any,
        token_counter: TokenCounter,
        summarizer: Summarizer,
        *,
        agent_id: str,
        session_id: str,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        threshold_fraction: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._store = state_store
        self._counter = token_counter
        self._summarizer = summarizer
        self._agent_id = agent_id
        self._session_id = session_id
        self._max_tokens = max_tokens
        self._threshold = threshold_fraction
        self._window: list[Message] = []
        self._history: list[Message] = []
        self._token_count: int = 0
        self._lock = asyncio.Lock()
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def token_count(self) -> int:
        """Current token count of the live window."""
        return self._token_count

    async def add(self, message: Message) -> None:
        """Append *message* to the live window and audit log, then auto-summarize if needed."""
        async with self._lock:
            self._window.append(message)
            self._history.append(message)
            self._token_count = self._counter.count_messages(self._window)
            await self._persist_window()
            await self._persist_history()
            if self._token_count >= self._threshold * self._max_tokens:
                await self._auto_summarize()

    async def get_messages(self) -> list[Message]:
        """Return a copy of the current live window."""
        return list(self._window)

    async def get_full_history(self) -> list[Message]:
        """Return the complete audit log from Dapr state."""
        return list(self._history)

    async def clear(self) -> None:
        """Reset the live window without touching the audit log."""
        async with self._lock:
            self._window = []
            self._token_count = 0
            await self._persist_window()
            _log.debug("working_memory_cleared", agent=self._agent_id, session=self._session_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _auto_summarize(self) -> None:
        compressed = await self._summarizer.summarize(self._window, max_tokens=self._max_tokens)
        self._window = compressed
        self._token_count = self._counter.count_messages(self._window)
        await self._persist_window()
        _log.debug(
            "working_memory_summarized",
            agent=self._agent_id,
            session=self._session_id,
            new_token_count=self._token_count,
        )

    async def _persist_window(self) -> None:
        data = json.dumps({"messages": [m.model_dump(mode="json") for m in self._window]}).encode()
        await self._store.save("working_window", self._window_key, data)

    async def _persist_history(self) -> None:
        data = json.dumps({"messages": [m.model_dump(mode="json") for m in self._history]}).encode()
        await self._store.save("working_history", self._history_key, data)

    @property
    def _window_key(self) -> str:
        return f"{self._agent_id}:{self._session_id}:window"

    @property
    def _history_key(self) -> str:
        return f"{self._agent_id}:{self._session_id}:history"
