"""Tests for nexus.dapr.client — DaprGateway async wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.dapr.client import DaprGateway


def make_state_response(data: bytes = b"", etag: str = "1") -> MagicMock:
    resp = MagicMock()
    resp.data = data
    resp.etag = MagicMock()
    resp.etag.value = etag
    return resp


def make_bulk_response(items: dict[str, bytes]) -> MagicMock:
    resp = MagicMock()
    bulk_items = []
    for key, data in items.items():
        item = MagicMock()
        item.key = key
        item.data = data
        item.etag = MagicMock()
        item.etag.value = "1"
        bulk_items.append(item)
    resp.items = bulk_items
    return resp


def make_lock_response(success: bool) -> MagicMock:
    resp = MagicMock()
    resp.success = success
    return resp


def make_unlock_response(status: int = 0) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    return resp


@pytest.fixture()
def mock_sdk_client() -> MagicMock:
    """A mock of the underlying synchronous DaprClient SDK."""
    client = MagicMock()
    client.save_state = MagicMock(return_value=MagicMock())
    client.get_state = MagicMock(return_value=make_state_response(b"", "1"))
    client.delete_state = MagicMock(return_value=MagicMock())
    client.save_bulk_state = MagicMock(return_value=MagicMock())
    client.get_bulk_state = MagicMock(return_value=make_bulk_response({}))
    client.execute_state_transaction = MagicMock(return_value=MagicMock())
    client.publish_event = MagicMock(return_value=MagicMock())
    client.try_lock = MagicMock(return_value=make_lock_response(True))
    client.unlock = MagicMock(return_value=make_unlock_response(0))
    return client


@pytest.fixture()
def gateway(mock_sdk_client: MagicMock) -> DaprGateway:
    return DaprGateway(_client=mock_sdk_client)


class TestDaprGatewayInit:
    def test_accepts_injected_client(self, mock_sdk_client: MagicMock) -> None:
        gw = DaprGateway(_client=mock_sdk_client)
        assert gw is not None

    def test_default_client_created_without_injection(self) -> None:
        with patch("nexus.dapr.client.DaprSDKClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            gw = DaprGateway()
            assert gw is not None
            mock_cls.assert_called_once()


class TestDaprGatewaySaveState:
    async def test_calls_sdk_save_state(
        self, gateway: DaprGateway, mock_sdk_client: MagicMock
    ) -> None:
        await gateway.save_state("store", "key1", b"data")
        mock_sdk_client.save_state.assert_called_once()

    async def test_passes_store_name_and_key(
        self, gateway: DaprGateway, mock_sdk_client: MagicMock
    ) -> None:
        await gateway.save_state("mystore", "mykey", b"v")
        call_kwargs = mock_sdk_client.save_state.call_args
        assert call_kwargs is not None
        args, kwargs = call_kwargs
        all_args = list(args) + list(kwargs.values())
        combined = str(all_args) + str(kwargs)
        assert "mystore" in combined
        assert "mykey" in combined

    async def test_passes_etag_when_provided(
        self, gateway: DaprGateway, mock_sdk_client: MagicMock
    ) -> None:
        await gateway.save_state("store", "key", b"v", etag="etag-123")
        call_kwargs = mock_sdk_client.save_state.call_args
        combined = str(call_kwargs)
        assert "etag-123" in combined

    async def test_returns_none_on_success(self, gateway: DaprGateway) -> None:
        result = await gateway.save_state("store", "key", b"v")
        assert result is None


class TestDaprGatewayGetState:
    async def test_returns_state_response(
        self, gateway: DaprGateway, mock_sdk_client: MagicMock
    ) -> None:
        mock_sdk_client.get_state.return_value = make_state_response(b'{"x":1}', "etag1")
        resp = await gateway.get_state("store", "key")
        assert resp.data == b'{"x":1}'
        assert resp.etag.value == "etag1"

    async def test_calls_sdk_with_store_and_key(
        self, gateway: DaprGateway, mock_sdk_client: MagicMock
    ) -> None:
        await gateway.get_state("teststore", "testkey")
        call_kwargs = mock_sdk_client.get_state.call_args
        combined = str(call_kwargs)
        assert "teststore" in combined
        assert "testkey" in combined


class TestDaprGatewayDeleteState:
    async def test_calls_sdk_delete_state(
        self, gateway: DaprGateway, mock_sdk_client: MagicMock
    ) -> None:
        await gateway.delete_state("store", "key")
        mock_sdk_client.delete_state.assert_called_once()

    async def test_returns_none_on_success(self, gateway: DaprGateway) -> None:
        result = await gateway.delete_state("store", "key")
        assert result is None


class TestDaprGatewayGetBulkState:
    async def test_returns_bulk_response(
        self, gateway: DaprGateway, mock_sdk_client: MagicMock
    ) -> None:
        mock_sdk_client.get_bulk_state.return_value = make_bulk_response({"k1": b"v1", "k2": b"v2"})
        resp = await gateway.get_bulk_state("store", ["k1", "k2"])
        assert len(resp.items) == 2

    async def test_passes_keys_to_sdk(
        self, gateway: DaprGateway, mock_sdk_client: MagicMock
    ) -> None:
        await gateway.get_bulk_state("store", ["a", "b", "c"])
        call_kwargs = mock_sdk_client.get_bulk_state.call_args
        combined = str(call_kwargs)
        assert "a" in combined


class TestDaprGatewaySaveBulkState:
    async def test_calls_sdk_save_bulk_state(
        self, gateway: DaprGateway, mock_sdk_client: MagicMock
    ) -> None:
        items = [("k1", b"v1"), ("k2", b"v2")]
        await gateway.save_bulk_state("store", items)
        mock_sdk_client.save_bulk_state.assert_called_once()

    async def test_returns_none(self, gateway: DaprGateway) -> None:
        result = await gateway.save_bulk_state("store", [("k", b"v")])
        assert result is None


class TestDaprGatewayExecuteStateTransaction:
    async def test_calls_sdk_execute_transaction(
        self, gateway: DaprGateway, mock_sdk_client: MagicMock
    ) -> None:
        ops = [{"operation": "upsert", "request": {"key": "k", "value": b"v"}}]
        await gateway.execute_state_transaction("store", ops)
        mock_sdk_client.execute_state_transaction.assert_called_once()

    async def test_returns_none(self, gateway: DaprGateway) -> None:
        result = await gateway.execute_state_transaction("store", [])
        assert result is None


class TestDaprGatewayPublishEvent:
    async def test_calls_sdk_publish_event(
        self, gateway: DaprGateway, mock_sdk_client: MagicMock
    ) -> None:
        await gateway.publish_event("pubsub", "topic", b'{"msg":"hi"}')
        mock_sdk_client.publish_event.assert_called_once()

    async def test_passes_pubsub_topic_data(
        self, gateway: DaprGateway, mock_sdk_client: MagicMock
    ) -> None:
        await gateway.publish_event("my-pubsub", "my-topic", b"payload")
        call_kwargs = mock_sdk_client.publish_event.call_args
        combined = str(call_kwargs)
        assert "my-pubsub" in combined
        assert "my-topic" in combined


class TestDaprGatewayTryLock:
    async def test_returns_true_when_lock_acquired(
        self, gateway: DaprGateway, mock_sdk_client: MagicMock
    ) -> None:
        mock_sdk_client.try_lock.return_value = make_lock_response(True)
        result = await gateway.try_lock("store", "resource", "owner", 30)
        assert result is True

    async def test_returns_false_when_lock_not_acquired(
        self, gateway: DaprGateway, mock_sdk_client: MagicMock
    ) -> None:
        mock_sdk_client.try_lock.return_value = make_lock_response(False)
        result = await gateway.try_lock("store", "resource", "owner", 30)
        assert result is False

    async def test_passes_resource_and_owner(
        self, gateway: DaprGateway, mock_sdk_client: MagicMock
    ) -> None:
        await gateway.try_lock("store", "my-resource", "lock-owner-1", 60)
        call_kwargs = mock_sdk_client.try_lock.call_args
        combined = str(call_kwargs)
        assert "my-resource" in combined
        assert "lock-owner-1" in combined


class TestDaprGatewayUnlock:
    async def test_returns_true_on_success_status(
        self, gateway: DaprGateway, mock_sdk_client: MagicMock
    ) -> None:
        mock_sdk_client.unlock.return_value = make_unlock_response(0)
        result = await gateway.unlock("store", "resource", "owner")
        assert result is True

    async def test_returns_false_on_nonzero_status(
        self, gateway: DaprGateway, mock_sdk_client: MagicMock
    ) -> None:
        mock_sdk_client.unlock.return_value = make_unlock_response(1)
        result = await gateway.unlock("store", "resource", "owner")
        assert result is False


class TestDaprGatewayIsHealthy:
    async def test_returns_true_when_healthy(self, gateway: DaprGateway) -> None:
        with patch("nexus.dapr.client.is_sidecar_healthy", AsyncMock(return_value=True)):
            result = await gateway.is_healthy("localhost", 3500)
        assert result is True

    async def test_returns_false_when_unhealthy(self, gateway: DaprGateway) -> None:
        with patch("nexus.dapr.client.is_sidecar_healthy", AsyncMock(return_value=False)):
            result = await gateway.is_healthy("localhost", 3500)
        assert result is False

    async def test_default_host_port_used(self, gateway: DaprGateway) -> None:
        with patch("nexus.dapr.client.is_sidecar_healthy", AsyncMock(return_value=True)) as mock_fn:
            await gateway.is_healthy()
            mock_fn.assert_called_once()
