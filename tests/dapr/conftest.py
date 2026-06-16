"""Shared fixtures for Dapr layer tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from grampus.core.config import DaprConfig


@pytest.fixture()
def dapr_config() -> DaprConfig:
    return DaprConfig(host="localhost", port=3500, grpc_port=50001)


def make_mock_state_response(data: bytes = b"", etag: str = "1") -> MagicMock:
    """Build a mock StateResponse with configurable data and etag."""
    resp = MagicMock()
    resp.data = data
    resp.etag = MagicMock()
    resp.etag.value = etag
    resp.text = data.decode() if data else ""
    return resp


def make_mock_bulk_response(items: dict[str, bytes]) -> MagicMock:
    """Build a mock BulkStatesResponse."""
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


def make_mock_lock_response(success: bool) -> MagicMock:
    resp = MagicMock()
    resp.success = success
    return resp


def make_mock_unlock_response(status: int = 0) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    return resp


@pytest.fixture()
def mock_gateway() -> AsyncMock:
    """A fully mocked DaprGateway for unit tests."""
    gw = AsyncMock()
    gw.save_state = AsyncMock(return_value=MagicMock())
    gw.get_state = AsyncMock(return_value=make_mock_state_response(b"", ""))
    gw.delete_state = AsyncMock(return_value=MagicMock())
    gw.save_bulk_state = AsyncMock(return_value=MagicMock())
    gw.get_bulk_state = AsyncMock(return_value=make_mock_bulk_response({}))
    gw.execute_state_transaction = AsyncMock(return_value=MagicMock())
    gw.publish_event = AsyncMock(return_value=MagicMock())
    gw.try_lock = AsyncMock(return_value=make_mock_lock_response(True))
    gw.unlock = AsyncMock(return_value=make_mock_unlock_response(0))
    gw.is_healthy = AsyncMock(return_value=True)
    return gw
