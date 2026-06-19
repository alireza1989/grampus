"""Shared fixtures for infrastructure integration tests.

All tests in this directory require Docker (testcontainers).
Run with: pytest -m integration tests/integration/infra/
"""

from __future__ import annotations

import asyncpg
import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def pg_container():
    """Start pgvector/pgvector:pg16 container for the whole session."""
    with PostgresContainer(image="pgvector/pgvector:pg16") as pg:
        yield pg


@pytest.fixture(scope="session")
def redis_container():
    """Start redis:7 container for the whole session."""
    with RedisContainer(image="redis:7") as redis:
        yield redis


@pytest_asyncio.fixture(scope="session")
async def pg_url(pg_container: PostgresContainer) -> str:
    """Connection URL with pgvector extension pre-enabled."""
    url = (
        f"postgresql://{pg_container.username}:{pg_container.password}"
        f"@{pg_container.get_container_host_ip()}:{pg_container.get_exposed_port(5432)}"
        f"/{pg_container.dbname}"
    )
    conn = await asyncpg.connect(url)
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.close()
    return url


@pytest_asyncio.fixture()
async def asyncpg_pool(pg_url: str):  # type: ignore[misc]
    """Fresh asyncpg pool per test, closed after the test."""
    pool = await asyncpg.create_pool(pg_url, min_size=1, max_size=3)
    yield pool
    await pool.close()
