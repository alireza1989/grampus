"""Web search tool via DuckDuckGo Instant Answer API."""

from __future__ import annotations

from typing import Any

import httpx

from grampus.tools.library._base import err

_DDGO_URL = "https://api.duckduckgo.com/"
_MAX_RESULTS_CAP = 10


def _make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=10.0)


def _parse_topics(topics: list[Any], limit: int) -> list[dict[str, str]]:
    results = []
    for item in topics:
        if not isinstance(item, dict):
            continue
        # Skip category groupings (they have "Name" + "Topics", not "FirstURL")
        if "FirstURL" not in item or not item["FirstURL"]:
            continue
        text = item.get("Text", "")
        url = item["FirstURL"]
        # DuckDuckGo Text often starts with the page title followed by " - snippet"
        parts = text.split(" - ", 1)
        title = parts[0].strip() if parts else text
        snippet = parts[1].strip() if len(parts) > 1 else text
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


async def web_search(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
) -> dict[str, Any]:
    """Search the web via DuckDuckGo Instant Answer API (no API key required).

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (capped at 10).
        region: DuckDuckGo region code, e.g. "us-en", "wt-wt" (worldwide).

    Returns:
        ``{"ok": True, "query": str, "results": list, "count": int}`` or error dict.
    """
    limit = min(max_results, _MAX_RESULTS_CAP)

    params = {
        "q": query,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
        "kl": region,
    }

    try:
        async with _make_client() as client:
            response = await client.get(_DDGO_URL, params=params)
        data = response.json()
    except httpx.HTTPError as exc:
        return err(f"Search request failed: {exc}", code="HTTP_ERROR")

    topics: list[Any] = data.get("RelatedTopics", [])
    results = _parse_topics(topics, limit)

    if not results:
        return {
            "ok": True,
            "query": query,
            "results": [],
            "count": 0,
            "note": "No instant results. Try a more specific query.",
        }

    return {"ok": True, "query": query, "results": results, "count": len(results)}
