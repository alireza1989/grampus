"""PlaygroundSession and PlaygroundTurn data models with save/load support."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from grampus.core.types import TokenUsage

if TYPE_CHECKING:
    from grampus.evaluation.suite import EvalCase

_SESSIONS_DIR = Path.home() / ".grampus" / "playground"


class PlaygroundTurn(BaseModel):
    """One user/assistant exchange within a playground session."""

    turn_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_input: str
    assistant_output: str
    model: str
    token_usage: TokenUsage | None = None
    duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PlaygroundSession(BaseModel):
    """An entire playground session accumulating multiple turns."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str | None = None
    model: str
    system_prompt: str = ""
    turns: list[PlaygroundTurn] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: list[str] = Field(default_factory=list)

    def total_cost_usd(self) -> float:
        """Sum cost_usd across all turns."""
        return sum(t.token_usage.cost_usd for t in self.turns if t.token_usage)

    def total_tokens(self) -> int:
        """Sum total_tokens across all turns."""
        return sum(t.token_usage.total_tokens for t in self.turns if t.token_usage)

    def to_eval_case(self, turn_index: int = -1) -> EvalCase:
        """Export one turn as an EvalCase (last turn by default)."""
        from grampus.evaluation.suite import EvalCase

        if not self.turns:
            raise ValueError("Session has no turns to export")
        turn = self.turns[turn_index]
        return EvalCase(
            name=f"playground_{turn.turn_id[:8]}",
            description=f"Exported from playground session {self.session_id[:8]}",
            input=turn.user_input,
            metadata={
                "expected_output": turn.assistant_output,
                "model": turn.model,
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
            },
        )

    def to_messages(self) -> list[dict[str, Any]]:
        """Reconstruct message list for replay (system + alternating user/assistant)."""
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for turn in self.turns:
            messages.append({"role": "user", "content": turn.user_input})
            messages.append({"role": "assistant", "content": turn.assistant_output})
        return messages

    def _filename(self) -> str:
        """Derive the JSON filename from name or session_id prefix."""
        return f"{self.name}.json" if self.name else f"{self.session_id[:8]}.json"

    def save(self, directory: Path | None = None) -> Path:
        """Persist session to <directory>/<name>.json, creating dir if needed."""
        dir_ = directory or _SESSIONS_DIR
        dir_.mkdir(parents=True, exist_ok=True)
        path = dir_ / self._filename()
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, name: str, directory: Path | None = None) -> PlaygroundSession:
        """Load a session by name or session_id prefix from disk."""
        dir_ = directory or _SESSIONS_DIR
        path = dir_ / f"{name}.json"
        if not path.exists():
            matches = sorted(dir_.glob(f"{name}*.json"))
            if not matches:
                raise FileNotFoundError(f"Session '{name}' not found in {dir_}")
            path = matches[0]
        data = json.loads(path.read_text())
        return cls.model_validate(data)
