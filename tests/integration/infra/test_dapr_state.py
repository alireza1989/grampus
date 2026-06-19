"""DaprStateStore serialization and namespacing tests (DaprGateway mocked with AsyncMock).

DaprGateway uses the Dapr gRPC SDK, not httpx, so tests mock the gateway
directly with AsyncMock — the same pattern as tests/dapr/test_state.py.
Goal: verify our serialization, key construction, and ETag logic are correct.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from grampus.core.errors import ConcurrencyError
from grampus.dapr.state import DaprStateStore

pytestmark = pytest.mark.integration


class _Widget(BaseModel):
    name: str
    value: int


def _resp(data: bytes = b"", etag: str = "") -> MagicMock:
    """Build a mock Dapr StateResponse."""
    r = MagicMock()
    r.data = data
    r.etag = etag
    return r


def _bulk_resp(items: list[tuple[str, bytes]]) -> MagicMock:
    """Build a mock Dapr BulkStatesResponse."""
    resp = MagicMock()
    mock_items = []
    for key, data in items:
        item = MagicMock()
        item.key = key
        item.data = data
        mock_items.append(item)
    resp.items = mock_items
    return resp


class TestSaveAndGetRoundtrip:
    async def test_save_and_get_roundtrip(self):
        gw = AsyncMock()
        store = DaprStateStore(gateway=gw, store_name="mystore", namespace="ns-rtrip")

        widget = _Widget(name="bolt", value=42)
        serialized = widget.model_dump_json().encode()

        gw.save_state = AsyncMock(return_value=None)
        gw.get_state = AsyncMock(return_value=_resp(serialized, "etag-1"))

        await store.save("widget", "w1", widget)
        result, etag = await store.get("widget", "w1", _Widget)

        assert result is not None
        assert result.name == "bolt"
        assert result.value == 42
        assert etag == "etag-1"
        gw.save_state.assert_called_once()
        gw.get_state.assert_called_once()


class TestETagConcurrencyRejectsStaleWrite:
    async def test_etag_concurrency_rejects_stale_write(self):
        from grpc import StatusCode

        class _FakeDaprGrpcError(Exception):
            grpc_statuscode = StatusCode.ABORTED

        gw = AsyncMock()
        store = DaprStateStore(gateway=gw, store_name="mystore", namespace="ns-etag")

        widget = _Widget(name="bolt", value=1)

        # First save succeeds (no ETag check)
        gw.save_state = AsyncMock(return_value=None)
        await store.save("widget", "w1", widget)

        # Second save with stale ETag fails
        gw.save_state = AsyncMock(side_effect=_FakeDaprGrpcError("ETag mismatch"))
        with pytest.raises(ConcurrencyError):
            await store.save("widget", "w1", widget, etag="stale-etag")


class TestNamespaceScopingIsolatesKeys:
    async def test_namespace_scoping_isolates_keys(self):
        captured_keys: list[str] = []

        async def _capture(store_name: str, key: str, value: bytes, **kwargs: object) -> None:
            captured_keys.append(key)

        gw = AsyncMock()
        gw.save_state = AsyncMock(side_effect=_capture)

        store_x = DaprStateStore(gateway=gw, store_name="mystore", namespace="ns-x")
        store_y = DaprStateStore(gateway=gw, store_name="mystore", namespace="ns-y")

        widget = _Widget(name="same-id", value=1)
        await store_x.save("widget", "w1", widget)
        await store_y.save("widget", "w1", widget)

        assert len(captured_keys) == 2
        assert captured_keys[0] != captured_keys[1], (
            "Different namespaces must produce different state keys"
        )
        assert "ns-x" in captured_keys[0]
        assert "ns-y" in captured_keys[1]


class TestBulkSaveAndGet:
    async def test_bulk_save_and_get(self):
        gw = AsyncMock()
        store = DaprStateStore(gateway=gw, store_name="mystore", namespace="ns-bulk")

        if not hasattr(store, "save_bulk") or not hasattr(store, "get_bulk"):
            pytest.skip("No bulk API on DaprStateStore")

        widgets = [(f"id{i}", _Widget(name=f"widget-{i}", value=i)) for i in range(3)]

        gw.save_bulk_state = AsyncMock(return_value=None)
        await store.save_bulk("widget", widgets)
        gw.save_bulk_state.assert_called_once()

        # Build mock bulk response with the full namespaced keys
        bulk_items = []
        for eid, widget in widgets:
            full_key = store._make_key("widget", eid)
            bulk_items.append((full_key, widget.model_dump_json().encode()))

        gw.get_bulk_state = AsyncMock(return_value=_bulk_resp(bulk_items))
        results = await store.get_bulk("widget", [f"id{i}" for i in range(3)], _Widget)

        assert len(results) == 3
        for i in range(3):
            assert f"id{i}" in results
            assert results[f"id{i}"].name == f"widget-{i}"
            assert results[f"id{i}"].value == i


class TestDeleteRemovesRecord:
    async def test_delete_removes_record(self):
        gw = AsyncMock()
        store = DaprStateStore(gateway=gw, store_name="mystore", namespace="ns-del")

        gw.delete_state = AsyncMock(return_value=None)
        gw.get_state = AsyncMock(return_value=_resp(b"", ""))

        await store.delete("widget", "w1")
        result, _ = await store.get("widget", "w1", _Widget)

        assert result is None
        gw.delete_state.assert_called_once()
