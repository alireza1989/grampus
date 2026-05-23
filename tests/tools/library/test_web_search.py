"""Tests for nexus.tools.library.web_search."""

from __future__ import annotations

import importlib

import httpx
import pytest

from nexus.tools.library.web_search import web_search

ws_mod = importlib.import_module("nexus.tools.library.web_search")

_DDGO_RESPONSE_WITH_RESULTS = {
    "RelatedTopics": [
        {"FirstURL": "https://example.com/1", "Text": "Result one about Python"},
        {"FirstURL": "https://example.com/2", "Text": "Result two about Python"},
        {"FirstURL": "https://example.com/3", "Text": "Result three about Python"},
    ]
}

_DDGO_RESPONSE_EMPTY = {"RelatedTopics": []}

_DDGO_RESPONSE_WITH_CATEGORY = {
    "RelatedTopics": [
        {"Name": "Category", "Topics": []},
        {"FirstURL": "https://example.com/1", "Text": "Result with URL"},
    ]
}


def _make_transport(payload: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


class TestWebSearch:
    async def test_web_search_returns_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            ws_mod,
            "_make_client",
            lambda: httpx.AsyncClient(transport=_make_transport(_DDGO_RESPONSE_WITH_RESULTS)),
        )
        result = await web_search(query="python programming")
        assert result["ok"] is True
        assert result["count"] > 0
        assert len(result["results"]) > 0

    async def test_web_search_result_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            ws_mod,
            "_make_client",
            lambda: httpx.AsyncClient(transport=_make_transport(_DDGO_RESPONSE_WITH_RESULTS)),
        )
        result = await web_search(query="python")
        r = result["results"][0]
        assert "title" in r
        assert "url" in r
        assert "snippet" in r

    async def test_web_search_empty_results_ok_not_err(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            ws_mod,
            "_make_client",
            lambda: httpx.AsyncClient(transport=_make_transport(_DDGO_RESPONSE_EMPTY)),
        )
        result = await web_search(query="xyzzy123abc")
        assert result["ok"] is True
        assert result["count"] == 0
        assert result["results"] == []
        assert "note" in result

    async def test_web_search_max_results_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        many_results = {
            "RelatedTopics": [
                {"FirstURL": f"https://example.com/{i}", "Text": f"Result {i}"} for i in range(20)
            ]
        }
        monkeypatch.setattr(
            ws_mod,
            "_make_client",
            lambda: httpx.AsyncClient(transport=_make_transport(many_results)),
        )
        result = await web_search(query="test", max_results=3)
        assert result["ok"] is True
        assert len(result["results"]) <= 3

    async def test_web_search_skips_category_topics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            ws_mod,
            "_make_client",
            lambda: httpx.AsyncClient(transport=_make_transport(_DDGO_RESPONSE_WITH_CATEGORY)),
        )
        result = await web_search(query="test")
        assert result["ok"] is True
        assert all("url" in r for r in result["results"])

    async def test_web_search_network_error_returns_err(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(
            ws_mod,
            "_make_client",
            lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        result = await web_search(query="test")
        assert result["ok"] is False

    async def test_web_search_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("timeout")

        monkeypatch.setattr(
            ws_mod,
            "_make_client",
            lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        result = await web_search(query="test")
        assert result["ok"] is False
