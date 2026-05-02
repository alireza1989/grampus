"""Tests for nexus.memory.procedural — ProceduralMemory CRUD and record_outcome."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from nexus.memory.procedural import ProceduralMemory
from nexus.memory.types import Procedure, ProcedureStep


def make_step(
    action: str = "search web",
    tool_name: str | None = "web_search",
    parameters_template: dict | None = None,
    expected_outcome: str | None = "relevant results",
) -> ProcedureStep:
    return ProcedureStep(
        action=action,
        tool_name=tool_name,
        parameters_template=parameters_template or {"query": "{topic}"},
        expected_outcome=expected_outcome,
    )


def make_procedure(
    procedure_id: str | None = None,
    name: str = "research_topic",
    description: str = "Research a topic using web search then summarize",
    steps: list[ProcedureStep] | None = None,
    agent_id: str = "agent-1",
    success_count: int = 0,
    failure_count: int = 0,
    embedding: list[float] | None = None,
) -> Procedure:
    return Procedure(
        id=procedure_id or str(uuid.uuid4()),
        name=name,
        description=description,
        steps=steps or [make_step(), make_step(action="summarize", tool_name="summarizer")],
        trigger_conditions=["user asks to research", "lookup task"],
        success_count=success_count,
        failure_count=failure_count,
        agent_id=agent_id,
        embedding=embedding,
    )


@pytest.fixture()
def mock_store() -> AsyncMock:
    store = AsyncMock()
    store.save = AsyncMock(return_value=None)
    store.get = AsyncMock(return_value=(None, ""))
    store.delete = AsyncMock(return_value=None)
    return store


@pytest.fixture()
def memory(mock_store: AsyncMock) -> ProceduralMemory:
    return ProceduralMemory(state_store=mock_store, agent_id="agent-1")


def make_stateful_memory() -> ProceduralMemory:
    """Return a ProceduralMemory backed by a simple in-memory dict store."""
    proc_store: dict[str, Procedure] = {}

    async def fake_save(entity: str, key: str, model_or_bytes: object, **_: object) -> None:
        if isinstance(model_or_bytes, Procedure):
            proc_store[key] = model_or_bytes

    async def fake_get(entity: str, key: str, cls: type) -> tuple:
        proc = proc_store.get(key)
        return proc, ("etag" if proc else "")

    async def fake_delete(entity: str, key: str, **_: object) -> None:
        proc_store.pop(key, None)

    store = AsyncMock()
    store.save = AsyncMock(side_effect=fake_save)
    store.get = AsyncMock(side_effect=fake_get)
    store.delete = AsyncMock(side_effect=fake_delete)
    return ProceduralMemory(state_store=store, agent_id="agent-1")


class TestProceduralMemoryStore:
    async def test_store_returns_procedure(self, memory: ProceduralMemory) -> None:
        proc = make_procedure()
        result = await memory.store(proc)
        assert isinstance(result, Procedure)

    async def test_store_preserves_name_and_description(self, memory: ProceduralMemory) -> None:
        proc = make_procedure(name="my_proc", description="does stuff")
        result = await memory.store(proc)
        assert result.name == "my_proc"
        assert result.description == "does stuff"

    async def test_store_persists_to_state_store(
        self, memory: ProceduralMemory, mock_store: AsyncMock
    ) -> None:
        await memory.store(make_procedure())
        mock_store.save.assert_called()

    async def test_store_adds_to_index(
        self, memory: ProceduralMemory, mock_store: AsyncMock
    ) -> None:
        await memory.store(make_procedure())
        # one save for the procedure, one for the index
        assert mock_store.save.call_count >= 2

    async def test_store_new_procedure_appears_in_index(self, memory: ProceduralMemory) -> None:
        proc = make_procedure()
        await memory.store(proc)
        assert proc.id in memory._index

    async def test_list_all_empty_initially(self, memory: ProceduralMemory) -> None:
        assert await memory.list_all() == []

    async def test_store_preserves_steps(self, memory: ProceduralMemory) -> None:
        steps = [make_step(action="step_a"), make_step(action="step_b")]
        proc = make_procedure(steps=steps)
        result = await memory.store(proc)
        assert len(result.steps) == 2
        assert result.steps[0].action == "step_a"


class TestProceduralMemoryGet:
    async def test_get_returns_none_for_missing(
        self, memory: ProceduralMemory, mock_store: AsyncMock
    ) -> None:
        mock_store.get.return_value = (None, "")
        result = await memory.get("nonexistent")
        assert result is None

    async def test_get_returns_procedure_when_exists(
        self, memory: ProceduralMemory, mock_store: AsyncMock
    ) -> None:
        proc = make_procedure(procedure_id="p-1")
        mock_store.get.return_value = (proc, "etag1")
        result = await memory.get("p-1")
        assert result is not None
        assert result.id == "p-1"


class TestProceduralMemoryDelete:
    async def test_delete_calls_store_delete(
        self, memory: ProceduralMemory, mock_store: AsyncMock
    ) -> None:
        await memory.delete("p-1")
        mock_store.delete.assert_called()

    async def test_delete_removes_from_index(self, memory: ProceduralMemory) -> None:
        proc = make_procedure()
        await memory.store(proc)
        assert proc.id in memory._index
        await memory.delete(proc.id)
        assert proc.id not in memory._index

    async def test_delete_nonexistent_is_a_noop(
        self, memory: ProceduralMemory, mock_store: AsyncMock
    ) -> None:
        await memory.delete("does-not-exist")
        mock_store.delete.assert_called_once()


class TestProceduralMemoryListAll:
    async def test_list_all_returns_all_stored(
        self, memory: ProceduralMemory, mock_store: AsyncMock
    ) -> None:
        p1 = make_procedure(procedure_id="p1", name="proc_a")
        p2 = make_procedure(procedure_id="p2", name="proc_b")
        memory._index = ["p1", "p2"]

        async def fake_get(entity: str, key: str, cls: type) -> tuple:
            mapping: dict[str, Procedure] = {"p1": p1, "p2": p2}
            return mapping.get(key), "etag"

        mock_store.get = AsyncMock(side_effect=fake_get)
        results = await memory.list_all()
        assert len(results) == 2

    async def test_list_all_skips_missing_ids(
        self, memory: ProceduralMemory, mock_store: AsyncMock
    ) -> None:
        memory._index = ["ghost-id"]
        mock_store.get.return_value = (None, "")
        results = await memory.list_all()
        assert results == []


class TestProceduralMemoryRecordOutcome:
    async def test_record_outcome_success_increments_success_count(self) -> None:
        memory = make_stateful_memory()
        proc = make_procedure(procedure_id="p-1", success_count=0)
        await memory.store(proc)
        await memory.record_outcome("p-1", success=True)
        updated = await memory.get("p-1")
        assert updated is not None
        assert updated.success_count == 1

    async def test_record_outcome_failure_increments_failure_count(self) -> None:
        memory = make_stateful_memory()
        proc = make_procedure(procedure_id="p-1", failure_count=0)
        await memory.store(proc)
        await memory.record_outcome("p-1", success=False)
        updated = await memory.get("p-1")
        assert updated is not None
        assert updated.failure_count == 1

    async def test_record_outcome_updates_last_used(self) -> None:
        memory = make_stateful_memory()
        proc = make_procedure(procedure_id="p-1")
        assert proc.last_used is None
        await memory.store(proc)
        await memory.record_outcome("p-1", success=True)
        updated = await memory.get("p-1")
        assert updated is not None
        assert updated.last_used is not None

    async def test_record_outcome_does_not_alter_other_counts(self) -> None:
        memory = make_stateful_memory()
        proc = make_procedure(procedure_id="p-1", success_count=3, failure_count=1)
        await memory.store(proc)
        await memory.record_outcome("p-1", success=True)
        updated = await memory.get("p-1")
        assert updated is not None
        assert updated.failure_count == 1  # unchanged

    async def test_record_outcome_noop_for_missing_id(
        self, memory: ProceduralMemory, mock_store: AsyncMock
    ) -> None:
        mock_store.get.return_value = (None, "")
        # should not raise
        await memory.record_outcome("nonexistent", success=True)

    async def test_record_outcome_last_used_is_recent(self) -> None:
        memory = make_stateful_memory()
        before = datetime.now(UTC)
        proc = make_procedure(procedure_id="p-1")
        await memory.store(proc)
        await memory.record_outcome("p-1", success=True)
        updated = await memory.get("p-1")
        assert updated is not None
        assert updated.last_used is not None
        assert updated.last_used >= before
