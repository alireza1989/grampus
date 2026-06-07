"""WeaviateVectorStore — async Weaviate v4 adapter."""

from __future__ import annotations

import contextlib
import uuid as _uuid
from typing import Any

from nexus.core.errors import ToolError
from nexus.memory.vector.base import VectorRecord, VectorSearchResult, VectorStore

_NS = _uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # uuid.NAMESPACE_URL
_INSTALL_HINT = "Install Weaviate support: pip install 'nexus-ai[weaviate]'"


def _to_weaviate_uuid(nexus_id: str) -> str:
    """Convert a Nexus string ID to a deterministic UUID5."""
    return str(_uuid.uuid5(_NS, nexus_id))


class WeaviateVectorStore(VectorStore):
    """Vector store backed by Weaviate (v4 async client).

    String IDs are mapped to deterministic UUID5 values because Weaviate
    requires UUID object IDs.  The original Nexus ID is stored in the object
    payload under ``_nexus_id`` so it can be recovered on search.

    Pass ``_client`` to inject a pre-built mock for unit tests.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        api_key: str | None = None,
        collection_name: str = "NexusMemory",
        _client: Any = None,
    ) -> None:
        self._host = host
        self._port = port
        self._api_key = api_key
        self._collection_name = collection_name
        self._injected_client = _client

    def _get_weaviate(self) -> Any:
        try:
            import weaviate  # noqa: PLC0415

            return weaviate
        except (ImportError, TypeError) as err:
            raise ToolError(
                "weaviate SDK not installed",
                code="missing_sdk_weaviate",
                hint=_INSTALL_HINT,
            ) from err

    def _make_client_ctx(self) -> Any:
        """Return an async context manager yielding a Weaviate client."""
        if self._injected_client is not None:
            return self._injected_client
        weaviate = self._get_weaviate()
        if self._api_key:
            return weaviate.use_async_with_weaviate_cloud(
                cluster_url=f"http://{self._host}:{self._port}",
                auth_credentials=weaviate.classes.init.Auth.api_key(self._api_key),
            )
        return weaviate.use_async_with_local(host=self._host, port=self._port)

    def _require_sdk(self) -> None:
        """Raise ToolError if the SDK is not installed and no client is injected."""
        if self._injected_client is None:
            self._get_weaviate()

    async def ensure_collection(self, dimension: int) -> None:
        """Create the Weaviate collection if it does not exist."""
        self._require_sdk()
        client_ctx = self._make_client_ctx()
        async with client_ctx as client:
            exists = await client.collections.exists(self._collection_name)
            if not exists:
                if self._injected_client is None:
                    weaviate = self._get_weaviate()
                    vectorizer = weaviate.classes.config.Configure.Vectorizer.none()
                else:
                    vectorizer = None
                await client.collections.create(
                    name=self._collection_name,
                    vectorizer_config=vectorizer,
                )

    async def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or replace records in Weaviate."""
        self._require_sdk()
        if self._injected_client is None:
            from weaviate.classes.data import DataObject  # noqa: PLC0415
        else:
            # Build a simple namespace object for tests
            class DataObject:  # type: ignore[no-redef]
                def __init__(self, **kwargs: Any) -> None:
                    for k, v in kwargs.items():
                        setattr(self, k, v)

        client_ctx = self._make_client_ctx()
        async with client_ctx as client:
            col = client.collections.get(self._collection_name)
            data_objects = [
                DataObject(
                    uuid=_to_weaviate_uuid(r.id),
                    properties={**r.payload, "_nexus_id": r.id},
                    vector=r.vector,
                )
                for r in records
            ]
            response = await col.data.insert_many(data_objects)

            # Handle duplicates: retry failed inserts as replaces
            if response.errors:
                for idx_str, _err in response.errors.items():
                    idx = int(idx_str)
                    record = records[idx]
                    await col.data.replace(
                        uuid=_to_weaviate_uuid(record.id),
                        properties={**record.payload, "_nexus_id": record.id},
                        vector=record.vector,
                    )

    async def search(
        self,
        vector: list[float],
        top_k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for nearest neighbours using vector similarity."""
        self._require_sdk()
        if self._injected_client is None:
            from weaviate.classes.query import MetadataQuery  # noqa: PLC0415
        else:

            class MetadataQuery:  # type: ignore[no-redef]
                def __init__(self, **kwargs: Any) -> None: ...

        client_ctx = self._make_client_ctx()
        async with client_ctx as client:
            col = client.collections.get(self._collection_name)
            response = await col.query.near_vector(
                near_vector=vector,
                limit=top_k,
                return_metadata=MetadataQuery(score=True, distance=True),
            )

        results = []
        for obj in response.objects:
            nexus_id = obj.properties.get("_nexus_id", "")
            score = obj.metadata.score if obj.metadata.score is not None else 0.0
            results.append(
                VectorSearchResult(
                    id=str(nexus_id),
                    score=float(score),
                    payload=dict(obj.properties),
                )
            )
        return results

    async def delete(self, ids: list[str]) -> None:
        """Delete records by Nexus ID (converted to UUID5)."""
        self._require_sdk()
        client_ctx = self._make_client_ctx()
        async with client_ctx as client:
            col = client.collections.get(self._collection_name)
            for nexus_id in ids:
                with contextlib.suppress(Exception):
                    await col.data.delete_by_id(_to_weaviate_uuid(nexus_id))
