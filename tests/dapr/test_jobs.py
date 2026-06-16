"""Unit tests for DaprJobsClient using mocked httpx."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from grampus.core.errors import DaprJobsError, GrampusError
from grampus.dapr.jobs import DaprJobsClient


def _mock_response(status_code: int, body: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = json.dumps(body) if body else ""
    resp.json = MagicMock(return_value=body or {})
    return resp


def _make_async_client(response: MagicMock) -> MagicMock:
    """Return a mock httpx.AsyncClient context manager yielding a client with fixed response."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=response)
    mock_client.get = AsyncMock(return_value=response)
    mock_client.delete = AsyncMock(return_value=response)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.mark.asyncio
async def test_schedule_job_calls_correct_endpoint() -> None:
    resp = _mock_response(204)
    cm = _make_async_client(resp)
    with patch("grampus.dapr.jobs.httpx.AsyncClient", return_value=cm):
        client = DaprJobsClient(host="localhost", port=3500)
        await client.schedule("daily", cron="0 9 * * *", data={"input": "report"})

    call_args = cm.__aenter__.return_value.post.call_args
    url: str = call_args[0][0]
    assert "/v1.0-alpha1/jobs/daily" in url
    body: dict = call_args[1]["json"]
    assert body["schedule"] == "0 9 * * *"
    assert body["data"]["@type"] == "type.googleapis.com/google.protobuf.StringValue"
    assert json.loads(body["data"]["value"]) == {"input": "report"}


@pytest.mark.asyncio
async def test_schedule_job_with_repeats_and_ttl() -> None:
    resp = _mock_response(200)
    cm = _make_async_client(resp)
    with patch("grampus.dapr.jobs.httpx.AsyncClient", return_value=cm):
        client = DaprJobsClient()
        await client.schedule("once", cron="@daily", data={}, repeats=3, ttl="24h")

    body: dict = cm.__aenter__.return_value.post.call_args[1]["json"]
    assert body["repeats"] == 3
    assert body["ttl"] == "24h"


@pytest.mark.asyncio
async def test_schedule_job_raises_on_error() -> None:
    resp = _mock_response(500)
    resp.text = "Internal error"
    cm = _make_async_client(resp)
    with patch("grampus.dapr.jobs.httpx.AsyncClient", return_value=cm):
        client = DaprJobsClient()
        with pytest.raises(DaprJobsError) as exc_info:
            await client.schedule("fail", cron="@hourly", data={})
    assert exc_info.value.code == "JOBS_SCHEDULE_FAILED"
    assert "500" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_job_found() -> None:
    job_data = {"name": "daily", "schedule": "0 9 * * *"}
    resp = _mock_response(200, job_data)
    cm = _make_async_client(resp)
    with patch("grampus.dapr.jobs.httpx.AsyncClient", return_value=cm):
        client = DaprJobsClient()
        result = await client.get("daily")
    assert result == job_data


@pytest.mark.asyncio
async def test_get_job_not_found() -> None:
    resp = _mock_response(404)
    cm = _make_async_client(resp)
    with patch("grampus.dapr.jobs.httpx.AsyncClient", return_value=cm):
        client = DaprJobsClient()
        result = await client.get("missing")
    assert result is None


@pytest.mark.asyncio
async def test_get_job_raises_on_unexpected_error() -> None:
    resp = _mock_response(503)
    cm = _make_async_client(resp)
    with patch("grampus.dapr.jobs.httpx.AsyncClient", return_value=cm):
        client = DaprJobsClient()
        with pytest.raises(DaprJobsError) as exc_info:
            await client.get("oops")
    assert exc_info.value.code == "JOBS_GET_FAILED"


@pytest.mark.asyncio
async def test_delete_job_found() -> None:
    resp = _mock_response(204)
    cm = _make_async_client(resp)
    with patch("grampus.dapr.jobs.httpx.AsyncClient", return_value=cm):
        client = DaprJobsClient()
        deleted = await client.delete("daily")
    assert deleted is True


@pytest.mark.asyncio
async def test_delete_job_not_found() -> None:
    resp = _mock_response(404)
    cm = _make_async_client(resp)
    with patch("grampus.dapr.jobs.httpx.AsyncClient", return_value=cm):
        client = DaprJobsClient()
        deleted = await client.delete("missing")
    assert deleted is False


@pytest.mark.asyncio
async def test_delete_job_raises_on_error() -> None:
    resp = _mock_response(500)
    cm = _make_async_client(resp)
    with patch("grampus.dapr.jobs.httpx.AsyncClient", return_value=cm):
        client = DaprJobsClient()
        with pytest.raises(DaprJobsError) as exc_info:
            await client.delete("bad")
    assert exc_info.value.code == "JOBS_DELETE_FAILED"


def test_dapr_jobs_error_is_grampus_error() -> None:
    err = DaprJobsError("something went wrong", code="JOBS_TEST")
    assert isinstance(err, GrampusError)
    assert err.code == "JOBS_TEST"
