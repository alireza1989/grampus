"""Namespace-scoped Dapr state store with Pydantic serialization and ETags."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from grampus.core.errors import ConcurrencyError
from grampus.core.logging import get_logger
from grampus.dapr.client import DaprGateway
from grampus.dapr.serialization import empty_response, from_dapr_bytes, to_dapr_bytes

_log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_KEY_SEP = ":"


class DaprStateStore:
    """Typed state store with namespace isolation and optimistic concurrency.

    All keys are formatted as ``{namespace}:{entity_type}:{id}`` to prevent
    cross-component collisions when multiple layers share the same Dapr store.
    """

    def __init__(
        self,
        gateway: DaprGateway,
        store_name: str,
        namespace: str,
    ) -> None:
        self._gw = gateway
        self._store = store_name
        self._ns = namespace

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(self, entity_type: str, entity_id: str) -> str:
        return _KEY_SEP.join([self._ns, entity_type, entity_id])

    def _strip_key(self, raw_key: str) -> str:
        """Reverse _make_key — return only the entity_id portion."""
        parts = raw_key.split(_KEY_SEP, 2)
        return parts[2] if len(parts) == 3 else raw_key

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def save(
        self,
        entity_type: str,
        entity_id: str,
        model: BaseModel,
        *,
        etag: str | None = None,
    ) -> None:
        """Persist a Pydantic model.

        Raises:
            ConcurrencyError: If *etag* is provided and doesn't match the
                stored version (optimistic concurrency failure).
        """
        key = self._make_key(entity_type, entity_id)
        value = to_dapr_bytes(model)
        try:
            await self._gw.save_state(self._store, key, value, etag=etag)
        except Exception as exc:
            _raise_if_concurrency_error(exc)
            raise
        _log.debug("state_saved", store=self._store, key=key)

    async def get(
        self,
        entity_type: str,
        entity_id: str,
        cls: type[T],
    ) -> tuple[T | None, str]:
        """Load a state entry.

        Returns:
            A ``(model, etag)`` tuple.  *model* is ``None`` when no data is
            stored for the key; *etag* is an empty string in that case.

        Raises:
            StateSerializationError: If stored bytes cannot be deserialized.
        """
        key = self._make_key(entity_type, entity_id)
        resp = await self._gw.get_state(self._store, key)
        _log.debug("state_got", store=self._store, key=key)
        if empty_response(resp):
            return None, ""
        return from_dapr_bytes(resp.data, cls), resp.etag

    async def delete(
        self,
        entity_type: str,
        entity_id: str,
        *,
        etag: str | None = None,
    ) -> None:
        """Delete a state entry."""
        key = self._make_key(entity_type, entity_id)
        await self._gw.delete_state(self._store, key, etag=etag)
        _log.debug("state_deleted", store=self._store, key=key)

    # ------------------------------------------------------------------
    # Bulk
    # ------------------------------------------------------------------

    async def get_bulk(
        self,
        entity_type: str,
        entity_ids: list[str],
        cls: type[T],
    ) -> dict[str, T]:
        """Load multiple entries. Missing/empty keys are omitted from result."""
        keys = [self._make_key(entity_type, eid) for eid in entity_ids]
        resp = await self._gw.get_bulk_state(self._store, keys)
        results: dict[str, T] = {}
        for item in resp.items:
            if not item.data:
                continue
            entity_id = self._strip_key(item.key)
            results[entity_id] = from_dapr_bytes(item.data, cls)
        _log.debug("state_bulk_got", store=self._store, count=len(results))
        return results

    async def save_bulk(
        self,
        entity_type: str,
        items: list[tuple[str, BaseModel]],
    ) -> None:
        """Save multiple entries in a single call."""
        pairs = [(self._make_key(entity_type, eid), to_dapr_bytes(model)) for eid, model in items]
        await self._gw.save_bulk_state(self._store, pairs)
        _log.debug("state_bulk_saved", store=self._store, count=len(pairs))

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    async def execute_transaction(
        self,
        entity_type: str,
        operations: list[tuple[str, str, str, BaseModel | None]],
    ) -> None:
        """Execute a batch of upsert/delete operations atomically.

        Each operation is a tuple of ``(op_type, entity_type, entity_id, model)``
        where *op_type* is ``"upsert"`` or ``"delete"`` and *model* may be
        ``None`` for deletes.
        """
        ops: list[dict[str, Any]] = []
        for op_type, _etype, entity_id, model in operations:
            key = self._make_key(entity_type, entity_id)
            if op_type == "upsert" and model is not None:
                ops.append(
                    {
                        "operationType": "upsert",
                        "request": {"key": key, "value": to_dapr_bytes(model)},
                    }
                )
            elif op_type == "delete":
                ops.append(
                    {
                        "operationType": "delete",
                        "request": {"key": key},
                    }
                )
        await self._gw.execute_state_transaction(self._store, ops)
        _log.debug("state_transaction", store=self._store, op_count=len(ops))


def _raise_if_concurrency_error(exc: Exception) -> None:
    """Re-raise as ConcurrencyError when the underlying cause is an ETag mismatch.

    The Dapr SDK wraps gRPC errors in DaprGrpcError. PostgreSQL state store
    returns ABORTED; some other stores use FAILED_PRECONDITION. Both indicate
    an optimistic concurrency conflict.
    """
    try:
        from grpc import StatusCode

        _CONCURRENCY_CODES = {StatusCode.ABORTED, StatusCode.FAILED_PRECONDITION}

        # DaprGrpcError (from dapr.clients.exceptions) exposes .grpc_statuscode
        grpc_code = getattr(exc, "grpc_statuscode", None)
        if grpc_code in _CONCURRENCY_CODES:
            raise ConcurrencyError(
                "ETag mismatch — concurrent modification detected",
                code="CONCURRENCY_ERROR",
                details={},
                hint="Reload the record before writing — another process modified it since you last read it.",
            ) from exc

        # Raw grpc.RpcError fallback (e.g. injected in unit tests)
        code_fn = getattr(exc, "code", None)
        if callable(code_fn) and code_fn() in _CONCURRENCY_CODES:
            raise ConcurrencyError(
                "ETag mismatch — concurrent modification detected",
                code="CONCURRENCY_ERROR",
                details={},
                hint="Reload the record before writing — another process modified it since you last read it.",
            ) from exc
    except ImportError:
        pass
