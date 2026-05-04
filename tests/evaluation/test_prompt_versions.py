"""Tests for PromptVersionManager."""

from __future__ import annotations

import pytest


class TestPromptVersionManager:
    def test_register_creates_version(self) -> None:
        from nexus.evaluation.prompt_versions import PromptVersionManager

        mgr = PromptVersionManager(agent_id="agent-1")
        v = mgr.register("1.0.0", "You are a helpful assistant")
        assert v.version == "1.0.0"
        assert v.prompt == "You are a helpful assistant"
        assert v.agent_id == "agent-1"

    def test_register_duplicate_raises_value_error(self) -> None:
        from nexus.evaluation.prompt_versions import PromptVersionManager

        mgr = PromptVersionManager(agent_id="agent-1")
        mgr.register("1.0.0", "prompt v1")
        with pytest.raises(ValueError, match="already exists"):
            mgr.register("1.0.0", "prompt v1 duplicate")

    def test_activate_sets_active_flag(self) -> None:
        from nexus.evaluation.prompt_versions import PromptVersionManager

        mgr = PromptVersionManager(agent_id="agent-1")
        mgr.register("1.0.0", "prompt v1")
        v = mgr.activate("1.0.0")
        assert v.is_active is True

    def test_activate_deactivates_previous(self) -> None:
        from nexus.evaluation.prompt_versions import PromptVersionManager

        mgr = PromptVersionManager(agent_id="agent-1")
        mgr.register("1.0.0", "v1")
        mgr.register("1.1.0", "v2")
        mgr.activate("1.0.0")
        mgr.activate("1.1.0")
        assert mgr.get("1.0.0") is not None
        assert mgr.get("1.0.0").is_active is False  # type: ignore[union-attr]
        assert mgr.get("1.1.0").is_active is True  # type: ignore[union-attr]

    def test_active_returns_none_initially(self) -> None:
        from nexus.evaluation.prompt_versions import PromptVersionManager

        mgr = PromptVersionManager(agent_id="agent-1")
        assert mgr.active() is None

    def test_history_sorted_by_created_at(self) -> None:
        from nexus.evaluation.prompt_versions import PromptVersionManager

        mgr = PromptVersionManager(agent_id="agent-1")
        mgr.register("1.0.0", "v1")
        mgr.register("2.0.0", "v2")
        history = mgr.history()
        assert len(history) == 2
        assert history[0].version == "1.0.0"
        assert history[1].version == "2.0.0"

    def test_diff_shows_added_lines(self) -> None:
        from nexus.evaluation.prompt_versions import PromptVersionManager

        mgr = PromptVersionManager(agent_id="agent-1")
        mgr.register("1.0.0", "Line A\nLine B")
        mgr.register("1.1.0", "Line A\nLine B\nLine C")
        diff = mgr.diff("1.0.0", "1.1.0")
        assert "Line C" in diff.added_lines

    def test_diff_shows_removed_lines(self) -> None:
        from nexus.evaluation.prompt_versions import PromptVersionManager

        mgr = PromptVersionManager(agent_id="agent-1")
        mgr.register("1.0.0", "Line A\nLine B\nLine C")
        mgr.register("1.1.0", "Line A\nLine B")
        diff = mgr.diff("1.0.0", "1.1.0")
        assert "Line C" in diff.removed_lines

    def test_diff_similarity_ratio_1_for_identical(self) -> None:
        from nexus.evaluation.prompt_versions import PromptVersionManager

        mgr = PromptVersionManager(agent_id="agent-1")
        mgr.register("1.0.0", "Same prompt text")
        mgr.register("1.1.0", "Same prompt text")
        diff = mgr.diff("1.0.0", "1.1.0")
        assert diff.similarity_ratio == pytest.approx(1.0)

    def test_rollback_activates_previous_version(self) -> None:
        from nexus.evaluation.prompt_versions import PromptVersionManager

        mgr = PromptVersionManager(agent_id="agent-1")
        mgr.register("1.0.0", "v1")
        mgr.register("1.1.0", "v2")
        mgr.activate("1.0.0")
        mgr.activate("1.1.0")
        prev = mgr.rollback()
        assert prev.version == "1.0.0"
        assert prev.is_active is True

    def test_rollback_raises_when_insufficient_history(self) -> None:
        from nexus.evaluation.prompt_versions import PromptVersionManager

        mgr = PromptVersionManager(agent_id="agent-1")
        mgr.register("1.0.0", "v1")
        with pytest.raises(ValueError, match="fewer than 2"):
            mgr.rollback()

    def test_record_eval_score_attaches_score(self) -> None:
        from nexus.evaluation.prompt_versions import PromptVersionManager

        mgr = PromptVersionManager(agent_id="agent-1")
        mgr.register("1.0.0", "v1")
        mgr.record_eval_score("1.0.0", 0.85)
        v = mgr.get("1.0.0")
        assert v is not None
        assert v.eval_score == pytest.approx(0.85)
