"""Tests for nexus.dapr.state — DaprStateStore namespaced CRUD."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from nexus.core.errors import ConcurrencyError, StateSerializationError
from nexus.dapr.state import DaprStateStore


class Item(BaseModel):
    name: str
    count: int


def make_state_response(data: bytes = b"", etag: str = "1") -> MagicMock:
    resp = MagicMock()
    resp.data = data
    resp.etag = etag  # Dapr SDK returns etag as a plain str
    return resp


def make_bulk_response(items: dict[str, tuple[bytes, str]]) -> MagicMock:
    resp = MagicMock()
    bulk_items = []
    for key, (data, etag) in items.items():
        item = MagicMock()
        item.key = key
        item.data = data
        item.etag = etag  # plain str
        bulk_items.append(item)
    resp.items = bulk_items
    return resp


@pytest.fixture()
def mock_gw() -> AsyncMock:
    gw = AsyncMock()
    gw.save_state = AsyncMock(return_value=None)
    gw.get_state = AsyncMock(return_value=make_state_response(b"", ""))
    gw.delete_state = AsyncMock(return_value=None)
    gw.save_bulk_state = AsyncMock(return_value=None)
    gw.get_bulk_state = AsyncMock(return_value=make_bulk_response({}))
    gw.execute_state_transaction = AsyncMock(return_value=None)
    return gw


@pytest.fixture()
def store(mock_gw: AsyncMock) -> DaprStateStore:
    return DaprStateStore(gateway=mock_gw, store_name="teststore", namespace="ns")


class TestDaprStateStoreKeyFormat:
    def test_key_includes_namespace(self, store: DaprStateStore) -> None:
        key = store._make_key("agent", "abc-123")
        assert "ns" in key

    def test_key_includes_entity_type(self, store: DaprStateStore) -> None:
        key = store._make_key("agent", "abc-123")
        assert "agent" in key

    def test_key_includes_id(self, store: DaprStateStore) -> None:
        key = store._make_key("agent", "abc-123")
        assert "abc-123" in key

    def test_different_namespaces_produce_different_keys(self) -> None:
        gw = AsyncMock()
        s1 = DaprStateStore(gateway=gw, store_name="store", namespace="ns1")
        s2 = DaprStateStore(gateway=gw, store_name="store", namespace="ns2")
        assert s1._make_key("x", "1") != s2._make_key("x", "1")


class TestDaprStateStoreSave:
    async def test_save_calls_gateway_save_state(
        self, store: DaprStateStore, mock_gw: AsyncMock
    ) -> None:
        item = Item(name="foo", count=1)
        await store.save("agent", "id1", item)
        mock_gw.save_state.assert_called_once()

    async def test_save_uses_correct_store_name(
        self, store: DaprStateStore, mock_gw: AsyncMock
    ) -> None:
        item = Item(name="foo", count=1)
        await store.save("agent", "id1", item)
        call_kwargs = mock_gw.save_state.call_args
        combined = str(call_kwargs)
        assert "teststore" in combined

    async def test_save_serializes_model_to_bytes(
        self, store: DaprStateStore, mock_gw: AsyncMock
    ) -> None:
        item = Item(name="bar", count=42)
        await store.save("agent", "id1", item)
        call_kwargs = mock_gw.save_state.call_args
        args, kwargs = call_kwargs
        value_bytes = kwargs.get("value") or args[2]
        parsed = json.loads(value_bytes)
        assert parsed["name"] == "bar"
        assert parsed["count"] == 42

    async def test_save_passes_etag_when_provided(
        self, store: DaprStateStore, mock_gw: AsyncMock
    ) -> None:
        item = Item(name="x", count=0)
        await store.save("agent", "id1", item, etag="etag-99")
        call_kwargs = mock_gw.save_state.call_args
        combined = str(call_kwargs)
        assert "etag-99" in combined

    async def test_save_raises_concurrency_error_on_etag_mismatch(
        self, store: DaprStateStore, mock_gw: AsyncMock
    ) -> None:
        from grpc import StatusCode

        class FakeRpcError(Exception):
            def code(self) -> StatusCode:
                return StatusCode.FAILED_PRECONDITION

        mock_gw.save_state.side_effect = FakeRpcError("ETag mismatch")
        item = Item(name="x", count=0)
        with pytest.raises(ConcurrencyError):
            await store.save("agent", "id1", item, etag="stale-etag")

    async def test_save_raises_concurrency_error_on_aborted(
        self, store: DaprStateStore, mock_gw: AsyncMock
    ) -> None:
        # PostgreSQL state store returns ABORTED for ETag conflicts;
        # the Dapr SDK wraps this in DaprGrpcError with .grpc_statuscode.
        from grpc import StatusCode

        class FakeDaprGrpcError(Exception):
            grpc_statuscode = StatusCode.ABORTED

        mock_gw.save_state.side_effect = FakeDaprGrpcError("etag mismatch")
        item = Item(name="x", count=0)
        with pytest.raises(ConcurrencyError):
            await store.save("agent", "id1", item, etag="stale-etag")


class TestDaprStateStoreGet:
    async def test_get_returns_model_when_data_exists(
        self, store: DaprStateStore, mock_gw: AsyncMock
    ) -> None:
        data = json.dumps({"name": "test", "count": 7}).encode()
        mock_gw.get_state.return_value = make_state_response(data, "etag1")
        result, etag = await store.get("agent", "id1", Item)
        assert result is not None
        assert result.name == "test"
        assert result.count == 7
        assert etag == "etag1"

    async def test_get_returns_none_when_no_data(
        self, store: DaprStateStore, mock_gw: AsyncMock
    ) -> None:
        mock_gw.get_state.return_value = make_state_response(b"", "")
        result, etag = await store.get("agent", "id1", Item)
        assert result is None

    async def test_get_raises_on_invalid_data(
        self, store: DaprStateStore, mock_gw: AsyncMock
    ) -> None:
        mock_gw.get_state.return_value = make_state_response(b"not-json", "1")
        with pytest.raises(StateSerializationError):
            await store.get("agent", "id1", Item)

    async def test_get_uses_namespaced_key(self, store: DaprStateStore, mock_gw: AsyncMock) -> None:
        mock_gw.get_state.return_value = make_state_response(b"", "")
        await store.get("agent", "my-id", Item)
        call_kwargs = mock_gw.get_state.call_args
        combined = str(call_kwargs)
        assert "ns" in combined
        assert "my-id" in combined


class TestDaprStateStoreDelete:
    async def test_delete_calls_gateway_delete_state(
        self, store: DaprStateStore, mock_gw: AsyncMock
    ) -> None:
        await store.delete("agent", "id1")
        mock_gw.delete_state.assert_called_once()

    async def test_delete_uses_namespaced_key(
        self, store: DaprStateStore, mock_gw: AsyncMock
    ) -> None:
        await store.delete("agent", "my-id")
        call_kwargs = mock_gw.delete_state.call_args
        combined = str(call_kwargs)
        assert "ns" in combined
        assert "my-id" in combined

    async def test_delete_passes_etag_when_provided(
        self, store: DaprStateStore, mock_gw: AsyncMock
    ) -> None:
        await store.delete("agent", "id1", etag="etag-42")
        call_kwargs = mock_gw.delete_state.call_args
        combined = str(call_kwargs)
        assert "etag-42" in combined


class TestDaprStateStoreGetBulk:
    async def test_get_bulk_returns_dict_of_models(
        self, store: DaprStateStore, mock_gw: AsyncMock
    ) -> None:
        d1 = json.dumps({"name": "a", "count": 1}).encode()
        d2 = json.dumps({"name": "b", "count": 2}).encode()
        ns_key1 = store._make_key("agent", "id1")
        ns_key2 = store._make_key("agent", "id2")
        mock_gw.get_bulk_state.return_value = make_bulk_response(
            {ns_key1: (d1, "e1"), ns_key2: (d2, "e2")}
        )
        results = await store.get_bulk("agent", ["id1", "id2"], Item)
        assert "id1" in results
        assert "id2" in results
        assert results["id1"].name == "a"
        assert results["id2"].name == "b"

    async def test_get_bulk_skips_empty_items(
        self, store: DaprStateStore, mock_gw: AsyncMock
    ) -> None:
        ns_key = store._make_key("agent", "id1")
        mock_gw.get_bulk_state.return_value = make_bulk_response({ns_key: (b"", "")})
        results = await store.get_bulk("agent", ["id1"], Item)
        assert "id1" not in results

    async def test_get_bulk_calls_gateway_with_store_name(
        self, store: DaprStateStore, mock_gw: AsyncMock
    ) -> None:
        await store.get_bulk("agent", ["id1"], Item)
        call_kwargs = mock_gw.get_bulk_state.call_args
        combined = str(call_kwargs)
        assert "teststore" in combined


class TestDaprStateStoreSaveBulk:
    async def test_save_bulk_calls_gateway(self, store: DaprStateStore, mock_gw: AsyncMock) -> None:
        items = [("id1", Item(name="a", count=1)), ("id2", Item(name="b", count=2))]
        await store.save_bulk("agent", items)
        mock_gw.save_bulk_state.assert_called_once()

    async def test_save_bulk_uses_namespaced_keys(
        self, store: DaprStateStore, mock_gw: AsyncMock
    ) -> None:
        items = [("my-id", Item(name="x", count=0))]
        await store.save_bulk("agent", items)
        call_kwargs = mock_gw.save_bulk_state.call_args
        combined = str(call_kwargs)
        assert "ns" in combined
        assert "my-id" in combined


class TestDaprStateStoreTransaction:
    async def test_transaction_upsert_calls_gateway(
        self, store: DaprStateStore, mock_gw: AsyncMock
    ) -> None:
        ops = [("upsert", "agent", "id1", Item(name="x", count=0))]
        await store.execute_transaction("agent", ops)
        mock_gw.execute_state_transaction.assert_called_once()

    async def test_transaction_delete_calls_gateway(
        self, store: DaprStateStore, mock_gw: AsyncMock
    ) -> None:
        ops = [("delete", "agent", "id1", None)]
        await store.execute_transaction("agent", ops)
        mock_gw.execute_state_transaction.assert_called_once()

    async def test_transaction_uses_correct_store_name(
        self, store: DaprStateStore, mock_gw: AsyncMock
    ) -> None:
        ops = [("upsert", "agent", "id1", Item(name="x", count=0))]
        await store.execute_transaction("agent", ops)
        call_kwargs = mock_gw.execute_state_transaction.call_args
        combined = str(call_kwargs)
        assert "teststore" in combined
