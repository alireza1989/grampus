"""In-memory state store for the demo (no Dapr sidecar required)."""

from __future__ import annotations

from typing import Any

from nexus.core.errors import ConcurrencyError


class FakeStateStore:
    """In-memory state store conforming to DaprStateStore duck-type interface."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[Any, str]] = {}
        self._etag_counter = 0

    async def save(
        self,
        entity: str,
        key: str,
        value: Any,
        *,
        etag: str | None = None,
        **_: Any,
    ) -> None:
        full_key = f"{entity}:{key}"
        if etag is not None:
            existing = self._data.get(full_key)
            if existing and existing[1] != etag:
                raise ConcurrencyError(
                    f"ETag mismatch for {full_key}",
                    code="CONCURRENCY_ERROR",
                )
        self._etag_counter += 1
        self._data[full_key] = (value, str(self._etag_counter))

    async def get(self, entity: str, key: str, cls: Any) -> tuple[Any, str]:
        full_key = f"{entity}:{key}"
        result = self._data.get(full_key)
        if result is None:
            return None, ""
        value, etag = result
        return value, etag

    async def delete(self, entity: str, key: str, **_: Any) -> None:
        self._data.pop(f"{entity}:{key}", None)

    async def bulk_get(self, entity: str, keys: list[str], cls: Any) -> list[tuple[Any, str]]:
        return [await self.get(entity, k, cls) for k in keys]

    async def save_bulk(self, entity: str, items: list[tuple[str, Any]]) -> None:
        for eid, model in items:
            await self.save(entity, eid, model)

    async def execute_transaction(
        self,
        entity: str,
        operations: list[tuple[str, str, str, Any]],
    ) -> None:
        for op_type, _etype, entity_id, model in operations:
            if op_type == "upsert" and model is not None:
                await self.save(entity, entity_id, model)
            elif op_type == "delete":
                await self.delete(entity, entity_id)
