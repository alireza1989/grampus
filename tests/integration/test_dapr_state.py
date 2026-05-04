"""Integration tests for the FakeStateStore (Dapr state store stand-in).

These tests run entirely in-process — no Docker required. They verify
CRUD semantics, ETag-based optimistic concurrency, and namespace isolation.
"""

from __future__ import annotations

import pytest

from nexus.core.errors import ConcurrencyError
from nexus.core.types import AgentState, AgentStatus
from tests.integration.conftest import FakeStateStore


@pytest.mark.integration
class TestDaprStateCRUD:
    async def test_save_and_get_pydantic_model(self, fake_state_store: FakeStateStore) -> None:
        state = AgentState(agent_id="agent1", session_id="s1")
        await fake_state_store.save("runner", "agent1:s1", state)
        result, etag = await fake_state_store.get("runner", "agent1:s1", AgentState)
        assert result is not None
        assert result.agent_id == "agent1"
        assert etag != ""

    async def test_save_and_get_bytes(self, fake_state_store: FakeStateStore) -> None:
        data = b'{"messages": []}'
        await fake_state_store.save("working", "key1", data)
        result, etag = await fake_state_store.get("working", "key1", bytes)
        assert result == data
        assert etag != ""

    async def test_delete_removes_key(self, fake_state_store: FakeStateStore) -> None:
        state = AgentState(agent_id="agent2", session_id="s2")
        await fake_state_store.save("runner", "agent2:s2", state)
        await fake_state_store.delete("runner", "agent2:s2")
        result, etag = await fake_state_store.get("runner", "agent2:s2", AgentState)
        assert result is None
        assert etag == ""

    async def test_get_missing_key_returns_none(self, fake_state_store: FakeStateStore) -> None:
        result, etag = await fake_state_store.get("runner", "nonexistent", AgentState)
        assert result is None
        assert etag == ""

    async def test_bulk_get_returns_all_values(self, fake_state_store: FakeStateStore) -> None:
        states = [AgentState(agent_id=f"a{i}", session_id=f"s{i}") for i in range(3)]
        for i, s in enumerate(states):
            await fake_state_store.save("runner", f"key{i}", s)

        results = await fake_state_store.bulk_get("runner", ["key0", "key1", "key2"], AgentState)
        assert len(results) == 3
        for val, etag in results:
            assert val is not None
            assert etag != ""

    async def test_overwrite_updates_value(self, fake_state_store: FakeStateStore) -> None:
        state1 = AgentState(agent_id="a1", session_id="s1")
        state2 = AgentState(agent_id="a2", session_id="s2")
        await fake_state_store.save("runner", "shared-key", state1)
        await fake_state_store.save("runner", "shared-key", state2)
        result, _ = await fake_state_store.get("runner", "shared-key", AgentState)
        assert result is not None
        assert result.agent_id == "a2"


@pytest.mark.integration
class TestDaprStateConcurrency:
    async def test_etag_mismatch_raises_concurrency_error(
        self, fake_state_store: FakeStateStore
    ) -> None:
        state = AgentState(agent_id="ca", session_id="s")
        await fake_state_store.save("runner", "ca:s", state)
        _, good_etag = await fake_state_store.get("runner", "ca:s", AgentState)

        with pytest.raises(ConcurrencyError):
            await fake_state_store.save("runner", "ca:s", state, etag="wrong-etag")

    async def test_correct_etag_allows_update(self, fake_state_store: FakeStateStore) -> None:
        state = AgentState(agent_id="cb", session_id="s")
        await fake_state_store.save("runner", "cb:s", state)
        _, etag = await fake_state_store.get("runner", "cb:s", AgentState)

        updated = state.model_copy(update={"status": AgentStatus.COMPLETED})
        await fake_state_store.save("runner", "cb:s", updated, etag=etag)

        result, _ = await fake_state_store.get("runner", "cb:s", AgentState)
        assert result is not None
        assert result.status == AgentStatus.COMPLETED

    async def test_save_without_etag_overwrites(self, fake_state_store: FakeStateStore) -> None:
        state1 = AgentState(agent_id="cc", session_id="s")
        await fake_state_store.save("runner", "cc:s", state1)
        state2 = state1.model_copy(update={"status": AgentStatus.FAILED})
        await fake_state_store.save("runner", "cc:s", state2)
        result, _ = await fake_state_store.get("runner", "cc:s", AgentState)
        assert result is not None
        assert result.status == AgentStatus.FAILED

    async def test_namespace_scoping_prevents_collision(
        self, fake_state_store: FakeStateStore
    ) -> None:
        state_a = AgentState(agent_id="same-id", session_id="sA")
        state_b = AgentState(agent_id="same-id", session_id="sB")
        await fake_state_store.save("ns_a", "same-key", state_a)
        await fake_state_store.save("ns_b", "same-key", state_b)

        result_a, _ = await fake_state_store.get("ns_a", "same-key", AgentState)
        result_b, _ = await fake_state_store.get("ns_b", "same-key", AgentState)
        assert result_a is not None and result_a.session_id == "sA"
        assert result_b is not None and result_b.session_id == "sB"

    async def test_etag_increments_per_write(self, fake_state_store: FakeStateStore) -> None:
        state = AgentState(agent_id="cd", session_id="s")
        await fake_state_store.save("runner", "cd:s", state)
        _, etag1 = await fake_state_store.get("runner", "cd:s", AgentState)
        await fake_state_store.save("runner", "cd:s", state, etag=etag1)
        _, etag2 = await fake_state_store.get("runner", "cd:s", AgentState)
        assert etag2 != etag1
