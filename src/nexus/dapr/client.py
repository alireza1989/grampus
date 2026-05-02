"""Async wrapper around the synchronous Dapr gRPC SDK client."""

from __future__ import annotations

import asyncio
from typing import Any

from dapr.clients import DaprClient as DaprSDKClient

from nexus.core.logging import get_logger
from nexus.dapr.health import is_sidecar_healthy

_log = get_logger(__name__)

_DEFAULT_HOST = "localhost"
_DEFAULT_PORT = 3500


class DaprGateway:
    """Async facade over the synchronous Dapr gRPC SDK.

    All blocking SDK calls are dispatched via ``asyncio.to_thread`` so they
    don't block the event loop.  Pass ``_client`` in tests to inject a mock.
    """

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        *,
        _client: Any | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._injected_client: Any | None = _client
        self._lazy_sdk_client: Any | None = None

    @property
    def _client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        if self._lazy_sdk_client is None:
            self._lazy_sdk_client = DaprSDKClient()
        return self._lazy_sdk_client

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    async def save_state(
        self,
        store_name: str,
        key: str,
        value: bytes,
        *,
        etag: str | None = None,
        state_metadata: dict[str, str] | None = None,
    ) -> None:
        """Save a single state entry."""
        kwargs: dict[str, Any] = {
            "store_name": store_name,
            "key": key,
            "value": value,
        }
        if etag is not None:
            kwargs["etag"] = etag
        if state_metadata is not None:
            kwargs["state_metadata"] = state_metadata
        await asyncio.to_thread(self._client.save_state, **kwargs)
        _log.debug("dapr_save_state", store=store_name, key=key)

    async def get_state(
        self,
        store_name: str,
        key: str,
        *,
        state_metadata: dict[str, str] | None = None,
    ) -> Any:
        """Return a StateResponse for the given key."""
        kwargs: dict[str, Any] = {"store_name": store_name, "key": key}
        if state_metadata is not None:
            kwargs["state_metadata"] = state_metadata
        resp = await asyncio.to_thread(self._client.get_state, **kwargs)
        _log.debug("dapr_get_state", store=store_name, key=key)
        return resp

    async def delete_state(
        self,
        store_name: str,
        key: str,
        *,
        etag: str | None = None,
        state_metadata: dict[str, str] | None = None,
    ) -> None:
        """Delete a state entry."""
        kwargs: dict[str, Any] = {"store_name": store_name, "key": key}
        if etag is not None:
            kwargs["etag"] = etag
        if state_metadata is not None:
            kwargs["state_metadata"] = state_metadata
        await asyncio.to_thread(self._client.delete_state, **kwargs)
        _log.debug("dapr_delete_state", store=store_name, key=key)

    async def get_bulk_state(
        self,
        store_name: str,
        keys: list[str],
        *,
        parallelism: int = 1,
        state_metadata: dict[str, str] | None = None,
    ) -> Any:
        """Return a BulkStatesResponse for multiple keys."""
        kwargs: dict[str, Any] = {
            "store_name": store_name,
            "keys": keys,
            "parallelism": parallelism,
        }
        if state_metadata is not None:
            kwargs["states_metadata"] = state_metadata
        resp = await asyncio.to_thread(self._client.get_bulk_state, **kwargs)
        _log.debug("dapr_get_bulk_state", store=store_name, key_count=len(keys))
        return resp

    async def save_bulk_state(
        self,
        store_name: str,
        items: list[tuple[str, bytes]],
    ) -> None:
        """Save multiple state entries in a single call."""
        from dapr.clients.grpc._state import StateItem

        states = [StateItem(key=k, value=v) for k, v in items]
        await asyncio.to_thread(self._client.save_bulk_state, store_name=store_name, states=states)
        _log.debug("dapr_save_bulk_state", store=store_name, item_count=len(items))

    async def execute_state_transaction(
        self,
        store_name: str,
        operations: list[dict[str, Any]],
        *,
        transactional_metadata: dict[str, str] | None = None,
    ) -> None:
        """Execute a transactional batch of state operations."""
        kwargs: dict[str, Any] = {
            "store_name": store_name,
            "operations": operations,
        }
        if transactional_metadata is not None:
            kwargs["transactional_metadata"] = transactional_metadata
        await asyncio.to_thread(self._client.execute_state_transaction, **kwargs)
        _log.debug("dapr_execute_transaction", store=store_name, op_count=len(operations))

    # ------------------------------------------------------------------
    # Pub/Sub
    # ------------------------------------------------------------------

    async def publish_event(
        self,
        pubsub_name: str,
        topic_name: str,
        data: bytes,
        *,
        data_content_type: str = "application/json",
        publish_metadata: dict[str, str] | None = None,
    ) -> None:
        """Publish an event to a Dapr pub/sub topic."""
        kwargs: dict[str, Any] = {
            "pubsub_name": pubsub_name,
            "topic_name": topic_name,
            "data": data,
            "data_content_type": data_content_type,
        }
        if publish_metadata is not None:
            kwargs["publish_metadata"] = publish_metadata
        await asyncio.to_thread(self._client.publish_event, **kwargs)
        _log.debug("dapr_publish_event", pubsub=pubsub_name, topic=topic_name)

    # ------------------------------------------------------------------
    # Distributed Lock
    # ------------------------------------------------------------------

    async def try_lock(
        self,
        store_name: str,
        resource_id: str,
        lock_owner: str,
        expiry_in_seconds: int,
    ) -> bool:
        """Attempt to acquire a distributed lock. Returns True if acquired."""
        resp = await asyncio.to_thread(
            self._client.try_lock,
            store_name=store_name,
            resource_id=resource_id,
            lock_owner=lock_owner,
            expiry_in_seconds=expiry_in_seconds,
        )
        _log.debug(
            "dapr_try_lock",
            store=store_name,
            resource=resource_id,
            acquired=resp.success,
        )
        return bool(resp.success)

    async def unlock(
        self,
        store_name: str,
        resource_id: str,
        lock_owner: str,
    ) -> bool:
        """Release a distributed lock. Returns True if successfully released."""
        resp = await asyncio.to_thread(
            self._client.unlock,
            store_name=store_name,
            resource_id=resource_id,
            lock_owner=lock_owner,
        )
        success: bool = resp.status == 0
        _log.debug(
            "dapr_unlock",
            store=store_name,
            resource=resource_id,
            success=success,
        )
        return success

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def is_healthy(
        self,
        host: str | None = None,
        port: int | None = None,
    ) -> bool:
        """Return True if the Dapr sidecar is healthy."""
        return await is_sidecar_healthy(
            host if host is not None else self._host,
            port if port is not None else self._port,
        )
