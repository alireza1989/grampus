"""PineconeVectorStore — async Pinecone v9 adapter."""

from __future__ import annotations

from typing import Any

from nexus.core.errors import ToolError
from nexus.memory.vector.base import VectorRecord, VectorSearchResult, VectorStore

_INSTALL_HINT = "Install Pinecone support: pip install 'nexus-ai[pinecone]'"


class PineconeVectorStore(VectorStore):
    """Vector store backed by Pinecone (cloud-hosted, serverless or pod-based).

    All network I/O uses the PineconeAsyncio SDK (v9+).
    Pass ``_client`` to inject a pre-built mock for unit tests.
    """

    def __init__(
        self,
        api_key: str,
        index_host: str,
        namespace: str = "nexus",
        cloud: str = "aws",
        region: str = "us-east-1",
        _client: Any = None,
    ) -> None:
        self._api_key = api_key
        self._index_host = index_host
        self._namespace = namespace
        self._cloud = cloud
        self._region = region
        self._injected_client = _client

    def _get_pinecone(self) -> Any:
        try:
            import pinecone  # noqa: PLC0415

            return pinecone
        except (ImportError, TypeError) as err:
            raise ToolError(
                "pinecone SDK not installed",
                code="missing_sdk_pinecone",
                hint=_INSTALL_HINT,
            ) from err

    def _make_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        pinecone = self._get_pinecone()
        return pinecone.PineconeAsyncio(api_key=self._api_key)

    def _require_sdk(self) -> None:
        """Raise ToolError if the SDK is not installed and no client is injected."""
        if self._injected_client is None:
            self._get_pinecone()

    async def ensure_collection(self, dimension: int) -> None:
        """Create the Pinecone index if it does not exist."""
        self._require_sdk()
        client = self._make_client()
        async with client as pc:
            try:
                await pc.describe_index(self._index_host)
            except Exception:
                if self._injected_client is None:
                    pinecone = self._get_pinecone()
                    spec = pinecone.ServerlessSpec(cloud=self._cloud, region=self._region)
                else:
                    spec = None
                await pc.create_index(
                    name=self._index_host,
                    dimension=dimension,
                    metric="cosine",
                    spec=spec,
                )

    async def upsert(self, records: list[VectorRecord]) -> None:
        """Upsert records into the Pinecone index."""
        self._require_sdk()
        client = self._make_client()
        vectors = [(r.id, r.vector, r.payload) for r in records]
        async with client as pc:  # noqa: SIM117
            async with pc.IndexAsyncio(host=self._index_host) as idx:
                await idx.upsert(vectors=vectors, namespace=self._namespace)

    async def search(
        self,
        vector: list[float],
        top_k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Query the Pinecone index for nearest neighbours."""
        self._require_sdk()
        client = self._make_client()
        async with client as pc:  # noqa: SIM117
            async with pc.IndexAsyncio(host=self._index_host) as idx:
                response = await idx.query(
                    vector=vector,
                    top_k=top_k,
                    namespace=self._namespace,
                    filter=filter,
                    include_metadata=True,
                )
        return [
            VectorSearchResult(
                id=m.id,
                score=m.score,
                payload=m.metadata or {},
            )
            for m in response.matches
        ]

    async def delete(self, ids: list[str]) -> None:
        """Delete records from the Pinecone index."""
        self._require_sdk()
        client = self._make_client()
        async with client as pc:  # noqa: SIM117
            async with pc.IndexAsyncio(host=self._index_host) as idx:
                await idx.delete(ids=ids, namespace=self._namespace)
