"""Tests for nexus.dapr.lock — DaprLock async context manager."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nexus.core.errors import LockAcquisitionError
from nexus.dapr.lock import DaprLock


@pytest.fixture()
def mock_gw() -> AsyncMock:
    gw = AsyncMock()
    gw.try_lock = AsyncMock(return_value=True)
    gw.unlock = AsyncMock(return_value=True)
    return gw


@pytest.fixture()
def lock(mock_gw: AsyncMock) -> DaprLock:
    return DaprLock(
        gateway=mock_gw,
        store_name="lockstore",
        resource_id="my-resource",
        lock_owner="owner-1",
        expiry_seconds=30,
    )


class TestDaprLockAcquire:
    async def test_acquires_lock_successfully(self, lock: DaprLock, mock_gw: AsyncMock) -> None:
        mock_gw.try_lock.return_value = True
        async with lock:
            mock_gw.try_lock.assert_called_once()

    async def test_raises_when_lock_not_acquired(self, lock: DaprLock, mock_gw: AsyncMock) -> None:
        mock_gw.try_lock.return_value = False
        with pytest.raises(LockAcquisitionError):
            async with lock:
                pass

    async def test_passes_resource_id_to_gateway(self, lock: DaprLock, mock_gw: AsyncMock) -> None:
        async with lock:
            call_kwargs = mock_gw.try_lock.call_args
            combined = str(call_kwargs)
            assert "my-resource" in combined

    async def test_passes_lock_owner_to_gateway(self, lock: DaprLock, mock_gw: AsyncMock) -> None:
        async with lock:
            call_kwargs = mock_gw.try_lock.call_args
            combined = str(call_kwargs)
            assert "owner-1" in combined

    async def test_passes_expiry_seconds_to_gateway(
        self, lock: DaprLock, mock_gw: AsyncMock
    ) -> None:
        async with lock:
            call_kwargs = mock_gw.try_lock.call_args
            combined = str(call_kwargs)
            assert "30" in combined

    async def test_error_contains_resource_id(self, mock_gw: AsyncMock) -> None:
        mock_gw.try_lock.return_value = False
        lock = DaprLock(
            gateway=mock_gw,
            store_name="s",
            resource_id="special-resource",
            lock_owner="o",
            expiry_seconds=5,
        )
        with pytest.raises(LockAcquisitionError) as exc_info:
            async with lock:
                pass
        assert "special-resource" in str(exc_info.value)


class TestDaprLockRelease:
    async def test_releases_lock_after_context(self, lock: DaprLock, mock_gw: AsyncMock) -> None:
        async with lock:
            pass
        mock_gw.unlock.assert_called_once()

    async def test_releases_lock_even_on_exception(
        self, lock: DaprLock, mock_gw: AsyncMock
    ) -> None:
        with pytest.raises(ValueError):
            async with lock:
                raise ValueError("body error")
        mock_gw.unlock.assert_called_once()

    async def test_passes_resource_id_to_unlock(self, lock: DaprLock, mock_gw: AsyncMock) -> None:
        async with lock:
            pass
        call_kwargs = mock_gw.unlock.call_args
        combined = str(call_kwargs)
        assert "my-resource" in combined

    async def test_passes_lock_owner_to_unlock(self, lock: DaprLock, mock_gw: AsyncMock) -> None:
        async with lock:
            pass
        call_kwargs = mock_gw.unlock.call_args
        combined = str(call_kwargs)
        assert "owner-1" in combined

    async def test_unlock_failure_is_logged_not_raised(
        self, lock: DaprLock, mock_gw: AsyncMock
    ) -> None:
        mock_gw.unlock.return_value = False
        async with lock:
            pass

    async def test_unlock_exception_does_not_swallow_body_exception(
        self, lock: DaprLock, mock_gw: AsyncMock
    ) -> None:
        mock_gw.unlock.side_effect = RuntimeError("unlock failed")
        with pytest.raises(ValueError, match="body error"):
            async with lock:
                raise ValueError("body error")


class TestDaprLockStoreNameUsage:
    async def test_uses_configured_store_name(self, mock_gw: AsyncMock) -> None:
        lock = DaprLock(
            gateway=mock_gw,
            store_name="custom-lockstore",
            resource_id="r",
            lock_owner="o",
            expiry_seconds=10,
        )
        async with lock:
            combined_lock = str(mock_gw.try_lock.call_args)
            assert "custom-lockstore" in combined_lock
        combined_unlock = str(mock_gw.unlock.call_args)
        assert "custom-lockstore" in combined_unlock
