"""Integration tests for the Dapr layer — require a live Dapr sidecar.

Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import pytest

from nexus.dapr.client import DaprGateway
from nexus.dapr.health import is_sidecar_healthy, wait_for_sidecar
from nexus.dapr.lock import DaprLock
from nexus.dapr.state import DaprStateStore

try:
    from pydantic import BaseModel

    class _Item(BaseModel):
        name: str
        count: int

except ImportError:
    pass

pytestmark = pytest.mark.integration


@pytest.fixture()
async def gateway() -> DaprGateway:
    gw = DaprGateway(host="localhost", port=3500)
    await wait_for_sidecar("localhost", 3500, timeout_seconds=30.0)
    return gw


@pytest.fixture()
def state_store(gateway: DaprGateway) -> DaprStateStore:
    return DaprStateStore(
        gateway=gateway,
        store_name="statestore",
        namespace="integration-test",
    )


@pytest.mark.integration
async def test_sidecar_is_healthy() -> None:
    result = await is_sidecar_healthy("localhost", 3500)
    assert result is True


@pytest.mark.integration
async def test_state_save_and_get(state_store: DaprStateStore) -> None:
    item = _Item(name="integration", count=42)
    await state_store.save("item", "it-1", item)
    result, etag = await state_store.get("item", "it-1", _Item)
    assert result is not None
    assert result.name == "integration"
    assert result.count == 42
    assert etag != ""


@pytest.mark.integration
async def test_state_delete(state_store: DaprStateStore) -> None:
    item = _Item(name="to-delete", count=0)
    await state_store.save("item", "it-del", item)
    await state_store.delete("item", "it-del")
    result, _ = await state_store.get("item", "it-del", _Item)
    assert result is None


@pytest.mark.integration
async def test_state_optimistic_concurrency(state_store: DaprStateStore) -> None:
    from nexus.core.errors import ConcurrencyError

    item = _Item(name="concurrent", count=1)
    await state_store.save("item", "it-cc", item)
    _, etag = await state_store.get("item", "it-cc", _Item)

    updated = _Item(name="concurrent", count=2)
    await state_store.save("item", "it-cc", updated, etag=etag)

    with pytest.raises(ConcurrencyError):
        await state_store.save("item", "it-cc", updated, etag=etag)


@pytest.mark.integration
async def test_state_bulk_get_set(state_store: DaprStateStore) -> None:
    items = [("bi-1", _Item(name="a", count=1)), ("bi-2", _Item(name="b", count=2))]
    await state_store.save_bulk("item", items)
    results = await state_store.get_bulk("item", ["bi-1", "bi-2"], _Item)
    assert len(results) == 2
    assert results["bi-1"].name == "a"
    assert results["bi-2"].name == "b"


@pytest.mark.integration
async def test_distributed_lock(gateway: DaprGateway) -> None:
    from nexus.core.errors import LockAcquisitionError

    lock1 = DaprLock(
        gateway=gateway,
        store_name="lockstore",
        resource_id="integration-lock",
        lock_owner="worker-1",
        expiry_seconds=5,
    )
    lock2 = DaprLock(
        gateway=gateway,
        store_name="lockstore",
        resource_id="integration-lock",
        lock_owner="worker-2",
        expiry_seconds=5,
    )
    async with lock1:
        with pytest.raises(LockAcquisitionError):
            async with lock2:
                pass
