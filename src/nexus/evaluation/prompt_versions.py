"""Prompt version management for agents."""

from __future__ import annotations

import difflib
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from nexus.core.logging import get_logger

logger = get_logger(__name__)


class PromptVersion(BaseModel):
    """A versioned system prompt snapshot.

    Attributes:
        id: Unique identifier.
        version: Semver string (e.g. "1.0.0").
        prompt: The system prompt text.
        agent_id: Agent this version belongs to.
        created_at: UTC creation timestamp.
        notes: Optional description of changes.
        is_active: Whether this is the currently active version.
        eval_score: Pass-rate score from the last eval run (populated externally).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: str
    prompt: str
    agent_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    notes: str = ""
    is_active: bool = False
    eval_score: float | None = None


class PromptDiff(BaseModel):
    """Line-level diff between two prompt versions.

    Attributes:
        from_version: Source version string.
        to_version: Target version string.
        added_lines: Lines present in to_version but not from_version.
        removed_lines: Lines present in from_version but not to_version.
        unchanged_lines: Count of lines unchanged between versions.
        similarity_ratio: 0.0–1.0 from difflib.SequenceMatcher.
    """

    from_version: str
    to_version: str
    added_lines: list[str]
    removed_lines: list[str]
    unchanged_lines: int
    similarity_ratio: float


class PromptVersionManager:
    """Tracks system prompt versions for an agent.

    All state is in-memory. Persistence can be layered in Phase 12.

    Args:
        agent_id: Scopes versions to this agent.
    """

    def __init__(self, *, agent_id: str) -> None:
        self._agent_id = agent_id
        self._versions: dict[str, PromptVersion] = {}
        self._active_version: str | None = None

    def register(self, version: str, prompt: str, *, notes: str = "") -> PromptVersion:
        """Register a new version.

        Args:
            version: Semver string for this prompt.
            prompt: The system prompt text.
            notes: Optional description.

        Returns:
            The newly created PromptVersion.

        Raises:
            ValueError: If the version string already exists.
        """
        if version in self._versions:
            raise ValueError(f"Version '{version}' already exists for agent '{self._agent_id}'")
        pv = PromptVersion(version=version, prompt=prompt, agent_id=self._agent_id, notes=notes)
        self._versions[version] = pv
        logger.info("prompt_version_registered", agent_id=self._agent_id, version=version)
        return pv

    def activate(self, version: str) -> PromptVersion:
        """Set a version as active, deactivating the previous one.

        Args:
            version: Version string to activate.

        Returns:
            The newly activated PromptVersion.

        Raises:
            ValueError: If the version does not exist.
        """
        if version not in self._versions:
            raise ValueError(f"Version '{version}' not found")
        if self._active_version and self._active_version in self._versions:
            prev = self._versions[self._active_version]
            self._versions[self._active_version] = prev.model_copy(update={"is_active": False})
        self._versions[version] = self._versions[version].model_copy(update={"is_active": True})
        self._active_version = version
        return self._versions[version]

    def active(self) -> PromptVersion | None:
        """Return the currently active version, or None."""
        if self._active_version is None:
            return None
        return self._versions.get(self._active_version)

    def get(self, version: str) -> PromptVersion | None:
        """Return a specific version by version string."""
        return self._versions.get(version)

    def history(self) -> list[PromptVersion]:
        """Return all versions sorted by created_at ascending."""
        return sorted(self._versions.values(), key=lambda v: v.created_at)

    def diff(self, from_version: str, to_version: str) -> PromptDiff:
        """Compute line-level diff between two versions.

        Args:
            from_version: Source version string.
            to_version: Target version string.

        Returns:
            PromptDiff with added/removed lines and similarity ratio.

        Raises:
            ValueError: If either version is not found.
        """
        from_pv = self._versions.get(from_version)
        to_pv = self._versions.get(to_version)
        if from_pv is None:
            raise ValueError(f"Version '{from_version}' not found")
        if to_pv is None:
            raise ValueError(f"Version '{to_version}' not found")
        return _compute_diff(from_pv, to_pv)

    def rollback(self) -> PromptVersion:
        """Activate the previous version (one step back in history).

        Returns:
            The newly activated PromptVersion.

        Raises:
            ValueError: If fewer than 2 versions exist.
        """
        hist = self.history()
        if len(hist) < 2:
            raise ValueError("Cannot rollback: fewer than 2 versions exist")
        current_active = self._active_version
        for i, pv in enumerate(hist):
            if pv.version == current_active and i > 0:
                return self.activate(hist[i - 1].version)
        return self.activate(hist[-2].version)

    def record_eval_score(self, version: str, score: float) -> None:
        """Attach an eval pass_rate score to a version.

        Args:
            version: Version string.
            score: Pass-rate score (0.0–1.0).
        """
        if version not in self._versions:
            raise ValueError(f"Version '{version}' not found")
        self._versions[version] = self._versions[version].model_copy(update={"eval_score": score})


def _compute_diff(from_pv: PromptVersion, to_pv: PromptVersion) -> PromptDiff:
    """Compute diff between two PromptVersion objects."""
    from_lines = from_pv.prompt.splitlines()
    to_lines = to_pv.prompt.splitlines()

    matcher = difflib.SequenceMatcher(None, from_lines, to_lines)
    ratio = matcher.ratio()

    added: list[str] = []
    removed: list[str] = []
    unchanged = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            unchanged += i2 - i1
        elif tag == "insert":
            added.extend(to_lines[j1:j2])
        elif tag == "delete":
            removed.extend(from_lines[i1:i2])
        elif tag == "replace":
            removed.extend(from_lines[i1:i2])
            added.extend(to_lines[j1:j2])

    return PromptDiff(
        from_version=from_pv.version,
        to_version=to_pv.version,
        added_lines=added,
        removed_lines=removed,
        unchanged_lines=unchanged,
        similarity_ratio=ratio,
    )
