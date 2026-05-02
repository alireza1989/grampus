"""Tests for nexus.dapr.health — sidecar health checking."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from nexus.core.errors import DaprConnectionError
from nexus.dapr.health import is_sidecar_healthy, wait_for_sidecar


class TestIsSidecarHealthy:
    async def test_returns_true_on_200(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("nexus.dapr.health.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client
            result = await is_sidecar_healthy("localhost", 3500)
        assert result is True

    async def test_returns_false_on_503(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        with patch("nexus.dapr.health.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client
            result = await is_sidecar_healthy("localhost", 3500)
        assert result is False

    async def test_returns_false_on_connect_error(self) -> None:
        with patch("nexus.dapr.health.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client_cls.return_value = mock_client
            result = await is_sidecar_healthy("localhost", 3500)
        assert result is False

    async def test_returns_false_on_timeout(self) -> None:
        with patch("nexus.dapr.health.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client_cls.return_value = mock_client
            result = await is_sidecar_healthy("localhost", 3500)
        assert result is False

    async def test_hits_correct_healthz_url(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("nexus.dapr.health.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client
            await is_sidecar_healthy("myhost", 9999)
            call_url = mock_client.get.call_args[0][0]
        assert "myhost" in call_url
        assert "9999" in call_url
        assert "healthz" in call_url


class TestWaitForSidecar:
    async def test_succeeds_immediately_when_healthy(self) -> None:
        with patch("nexus.dapr.health.is_sidecar_healthy", AsyncMock(return_value=True)):
            await wait_for_sidecar("localhost", 3500, timeout_seconds=5.0)

    async def test_succeeds_after_retries(self) -> None:
        # First two calls fail, third succeeds
        side_effects = [False, False, True]
        call_count = 0

        async def fake_healthy(host: str, port: int) -> bool:
            nonlocal call_count
            result = side_effects[min(call_count, len(side_effects) - 1)]
            call_count += 1
            return result

        with (
            patch("nexus.dapr.health.is_sidecar_healthy", fake_healthy),
            patch("nexus.dapr.health.asyncio.sleep", AsyncMock()),
        ):
            await wait_for_sidecar("localhost", 3500, timeout_seconds=10.0)
        assert call_count == 3

    async def test_raises_on_timeout(self) -> None:
        with (
            patch("nexus.dapr.health.is_sidecar_healthy", AsyncMock(return_value=False)),
            patch("nexus.dapr.health.asyncio.sleep", AsyncMock()),
            pytest.raises(DaprConnectionError) as exc_info,
        ):
            await wait_for_sidecar("localhost", 3500, timeout_seconds=1.0)
        assert exc_info.value.code == "DAPR_CONNECTION_ERROR"

    async def test_zero_timeout_checks_once(self) -> None:
        call_count = 0

        async def fake_healthy(host: str, port: int) -> bool:
            nonlocal call_count
            call_count += 1
            return False

        with (
            patch("nexus.dapr.health.is_sidecar_healthy", fake_healthy),
            pytest.raises(DaprConnectionError),
        ):
            await wait_for_sidecar("localhost", 3500, timeout_seconds=0.0)
        assert call_count >= 1

    async def test_error_contains_host_and_port(self) -> None:
        with (
            patch("nexus.dapr.health.is_sidecar_healthy", AsyncMock(return_value=False)),
            patch("nexus.dapr.health.asyncio.sleep", AsyncMock()),
            pytest.raises(DaprConnectionError) as exc_info,
        ):
            await wait_for_sidecar("myhost", 1234, timeout_seconds=0.0)
        details = exc_info.value.details
        assert details.get("host") == "myhost"
        assert details.get("port") == 1234
