"""Distributed lock as an async context manager backed by Dapr."""

from __future__ import annotations

from types import TracebackType
from typing import Self

from nexus.core.errors import LockAcquisitionError
from nexus.core.logging import get_logger
from nexus.dapr.client import DaprGateway

_log = get_logger(__name__)


class DaprLock:
    """Async context manager for a Dapr distributed lock.

    Usage::

        lock = DaprLock(gateway, store_name="lockstore",
                        resource_id="job-42", lock_owner="worker-1",
                        expiry_seconds=30)
        async with lock:
            # exclusive section
            ...

    Raises:
        LockAcquisitionError: If the lock cannot be acquired on entry.
    """

    def __init__(
        self,
        gateway: DaprGateway,
        *,
        store_name: str,
        resource_id: str,
        lock_owner: str,
        expiry_seconds: int,
    ) -> None:
        self._gw = gateway
        self._store = store_name
        self._resource = resource_id
        self._owner = lock_owner
        self._expiry = expiry_seconds

    async def __aenter__(self) -> Self:
        acquired = await self._gw.try_lock(self._store, self._resource, self._owner, self._expiry)
        if not acquired:
            raise LockAcquisitionError(
                f"Could not acquire lock on '{self._resource}'",
                code="LOCK_ACQUISITION_ERROR",
                details={
                    "store_name": self._store,
                    "resource_id": self._resource,
                    "lock_owner": self._owner,
                },
            )
        _log.debug("lock_acquired", resource=self._resource, owner=self._owner)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        try:
            success = await self._gw.unlock(self._store, self._resource, self._owner)
            if not success:
                _log.warning(
                    "lock_unlock_failed",
                    resource=self._resource,
                    owner=self._owner,
                )
        except Exception:
            _log.warning(
                "lock_unlock_exception",
                resource=self._resource,
                owner=self._owner,
                exc_info=True,
            )
        _log.debug("lock_released", resource=self._resource, owner=self._owner)
        return False
