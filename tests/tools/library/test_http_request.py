"""Tests for grampus.tools.library.http_request."""

from __future__ import annotations

import importlib
import json

import httpx
import pytest

from grampus.tools.library.http_request import http_request

http_mod = importlib.import_module("grampus.tools.library.http_request")


def _make_transport(
    status_code: int = 200,
    body: str | bytes = '{"hello": "world"}',
    content_type: str = "application/json",
) -> httpx.MockTransport:
    if isinstance(body, str):
        body = body.encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status_code,
            content=body,
            headers={"content-type": content_type},
        )

    return httpx.MockTransport(handler)


class TestGetRequest:
    async def test_get_request_returns_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            http_mod,
            "_make_client",
            lambda timeout: httpx.AsyncClient(transport=_make_transport()),
        )
        result = await http_request(url="http://example.com/api", method="GET")
        assert result["ok"] is True
        assert result["status_code"] == 200

    async def test_json_response_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            http_mod,
            "_make_client",
            lambda timeout: httpx.AsyncClient(transport=_make_transport()),
        )
        result = await http_request(url="http://example.com/api", method="GET")
        assert result["ok"] is True
        assert result["body"] == {"hello": "world"}

    async def test_non_json_response_returns_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            http_mod,
            "_make_client",
            lambda timeout: httpx.AsyncClient(
                transport=_make_transport(body="plain text", content_type="text/plain")
            ),
        )
        result = await http_request(url="http://example.com", method="GET")
        assert result["ok"] is True
        assert result["body"] == "plain text"


class TestPostRequest:
    async def test_post_request_sends_json_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        received: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            received["body"] = json.loads(request.content)
            return httpx.Response(200, json={"status": "ok"})

        monkeypatch.setattr(
            http_mod,
            "_make_client",
            lambda timeout: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        await http_request(
            url="http://example.com/api",
            method="POST",
            body={"key": "value"},
        )
        assert received["body"] == {"key": "value"}

    async def test_post_returns_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            http_mod,
            "_make_client",
            lambda timeout: httpx.AsyncClient(transport=_make_transport()),
        )
        result = await http_request(url="http://example.com/api", method="POST", body={"x": 1})
        assert result["ok"] is True


class TestTruncation:
    async def test_response_body_truncated_at_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        big_body = "x" * 20_000
        monkeypatch.setattr(
            http_mod,
            "_make_client",
            lambda timeout: httpx.AsyncClient(
                transport=_make_transport(body=big_body, content_type="text/plain")
            ),
        )
        result = await http_request(url="http://example.com", method="GET")
        assert result["ok"] is True
        assert len(result["body"]) == 10_000


class TestErrors:
    async def test_network_error_returns_err(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(
            http_mod,
            "_make_client",
            lambda timeout: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        result = await http_request(url="http://example.com", method="GET")
        assert result["ok"] is False
        assert "error" in result

    async def test_invalid_method_returns_err(self) -> None:
        result = await http_request(url="http://example.com", method="HACK")
        assert result["ok"] is False

    async def test_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("timeout")

        monkeypatch.setattr(
            http_mod,
            "_make_client",
            lambda timeout: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        result = await http_request(url="http://example.com", method="GET")
        assert result["ok"] is False
