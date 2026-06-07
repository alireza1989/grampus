"""QdrantVectorStore — async Qdrant v1.9+ adapter."""

from __future__ import annotations

import uuid as _uuid
from typing import Any

from nexus.core.errors import ToolError
from nexus.memory.vector.base import VectorRecord, VectorSearchResult, VectorStore

_NS = _uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # uuid.NAMESPACE_URL
_INSTALL_HINT = "Install Qdrant support: pip install 'nexus-ai[qdrant]'"


class _PointIdsList:
    """Minimal stub for qdrant_client.models.PointIdsList used in tests."""

    def __init__(self, points: list[str]) -> None:
        self.points = points


class _PointStub:
    """Minimal PointStruct stub for unit tests."""

    def __init__(self, id: str, vector: list[float], payload: dict[str, Any]) -> None:
        self.id = id
        self.vector = vector
        self.payload = payload


class _QdrantModelStub:
    """Minimal stub for qdrant_client.models namespace used in unit tests."""

    class Distance:
        COSINE = "Cosine"

    @staticmethod
    def VectorParams(size: int, distance: Any) -> Any:
        return {"size": size, "distance": distance}

    @staticmethod
    def PointStruct(id: str, vector: list[float], payload: dict[str, Any]) -> _PointStub:
        return _PointStub(id=id, vector=vector, payload=payload)

    @staticmethod
    def PointIdsList(points: list[str]) -> _PointIdsList:
        return _PointIdsList(points)


def _to_qdrant_uuid(nexus_id: str) -> str:
    """Convert a Nexus string ID to a deterministic UUID5."""
    return str(_uuid.uuid5(_NS, nexus_id))


class QdrantVectorStore(VectorStore):
    """Vector store backed by Qdrant (AsyncQdrantClient, v1.9+).

    String IDs are mapped to deterministic UUID5 values because Qdrant point
    IDs must be unsigned int or UUID strings.  The original Nexus ID is stored
    in the point payload under ``_nexus_id``.

    Pass ``_client`` to inject a pre-built mock for unit tests.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_name: str = "nexus_memory",
        _client: Any = None,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._collection_name = collection_name
        self._injected_client = _client

    def _get_qdrant(self) -> Any:
        try:
            import qdrant_client  # noqa: PLC0415

            return qdrant_client
        except (ImportError, TypeError) as err:
            raise ToolError(
                "qdrant_client SDK not installed",
                code="missing_sdk_qdrant",
                hint=_INSTALL_HINT,
            ) from err

    def _get_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        qdrant_client = self._get_qdrant()
        return qdrant_client.AsyncQdrantClient(url=self._url, api_key=self._api_key)

    def _require_sdk(self) -> None:
        """Raise ToolError if the SDK is not installed and no client is injected."""
        if self._injected_client is None:
            self._get_qdrant()

    def _get_models(self) -> Any:
        """Return the qdrant_client.models namespace, or a test stub."""
        if self._injected_client is None:
            from qdrant_client import models  # noqa: PLC0415

            return models
        return _QdrantModelStub()

    async def ensure_collection(self, dimension: int) -> None:
        """Create the Qdrant collection if it does not exist."""
        self._require_sdk()
        models = self._get_models()
        client = self._get_client()
        collections = await client.get_collections()
        names = [c.name for c in collections.collections]
        if self._collection_name not in names:
            await client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )

    async def upsert(self, records: list[VectorRecord]) -> None:
        """Upsert records into the Qdrant collection."""
        self._require_sdk()
        models = self._get_models()
        client = self._get_client()
        points = [
            models.PointStruct(
                id=_to_qdrant_uuid(r.id),
                vector=r.vector,
                payload={**r.payload, "_nexus_id": r.id},
            )
            for r in records
        ]
        await client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

    async def search(
        self,
        vector: list[float],
        top_k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Query the Qdrant collection for nearest neighbours."""
        self._require_sdk()
        client = self._get_client()
        results = await client.query_points(
            collection_name=self._collection_name,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            VectorSearchResult(
                id=str(p.payload.get("_nexus_id", p.id)),
                score=float(p.score),
                payload=dict(p.payload),
            )
            for p in results.points
        ]

    async def delete(self, ids: list[str]) -> None:
        """Delete records by Nexus ID (converted to UUID5)."""
        self._require_sdk()
        models = self._get_models()
        client = self._get_client()
        await client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(
                points=[_to_qdrant_uuid(i) for i in ids],
            ),
        )

    async def close(self) -> None:
        """Close the underlying Qdrant client connection."""
        if self._injected_client is not None:
            await self._injected_client.close()
