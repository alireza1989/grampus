"""Tests for TemplateRegistry."""

from __future__ import annotations

import json

from nexus.hub.registry import TemplateIndexEntry, TemplateRegistry


def _make_entry(**kwargs: object) -> TemplateIndexEntry:
    defaults: dict[str, object] = {
        "name": "test-template",
        "version": "1.0.0",
        "description": "A test template",
        "tags": ["test"],
        "category": "general",
        "author": "test",
        "download_url": "https://example.com/test.zip",
        "manifest_url": "https://example.com/nexus-template.yaml",
    }
    defaults.update(kwargs)
    return TemplateIndexEntry(**defaults)  # type: ignore[arg-type]


def _make_index_json(entries: list[dict[str, object]]) -> str:
    return json.dumps(
        {
            "version": "1",
            "updated_at": "2026-01-01T00:00:00Z",
            "templates": entries,
        }
    )


class TestBuiltinTemplates:
    def test_builtin_templates_always_available(self) -> None:
        registry = TemplateRegistry(_http_client=None)
        # Even with no network, built-ins are available
        templates = registry.list_templates()
        names = [t.name for t in templates]
        assert "simple-agent" in names
        assert "deep-research-crew" in names
        assert "customer-support-rag" in names
        assert "code-reviewer" in names

    def test_list_templates_returns_builtins(self) -> None:
        registry = TemplateRegistry(_http_client=None)
        templates = registry.list_templates()
        assert len(templates) >= 4

    def test_list_templates_filter_by_category(self) -> None:
        registry = TemplateRegistry(_http_client=None)
        research = registry.list_templates(category="research")
        assert all(t.category == "research" for t in research)
        names = [t.name for t in research]
        assert "deep-research-crew" in names

    def test_list_templates_filter_by_tag(self) -> None:
        registry = TemplateRegistry(_http_client=None)
        rag_templates = registry.list_templates(tag="rag")
        names = [t.name for t in rag_templates]
        assert "customer-support-rag" in names


class TestSearchAndGet:
    def test_search_matches_name(self) -> None:
        registry = TemplateRegistry(_http_client=None)
        results = registry.search("simple")
        names = [t.name for t in results]
        assert "simple-agent" in names

    def test_search_matches_description(self) -> None:
        registry = TemplateRegistry(_http_client=None)
        results = registry.search("research")
        names = [t.name for t in results]
        assert "deep-research-crew" in names

    def test_search_matches_tags(self) -> None:
        registry = TemplateRegistry(_http_client=None)
        results = registry.search("multi-agent")
        assert len(results) >= 1

    def test_search_case_insensitive(self) -> None:
        registry = TemplateRegistry(_http_client=None)
        lower = registry.search("SIMPLE")
        upper = registry.search("simple")
        assert {t.name for t in lower} == {t.name for t in upper}

    def test_get_exact_match(self) -> None:
        registry = TemplateRegistry(_http_client=None)
        entry = registry.get("simple-agent")
        assert entry is not None
        assert entry.name == "simple-agent"

    def test_get_missing_returns_none(self) -> None:
        registry = TemplateRegistry(_http_client=None)
        assert registry.get("nonexistent-template-xyz") is None


class TestNetworkHandling:
    def test_network_failure_falls_back_to_builtins(self) -> None:
        def failing_client(url: str) -> str:
            raise OSError("Network unavailable")

        registry = TemplateRegistry(_http_client=failing_client)
        # Should not raise — falls back to built-ins silently
        templates = registry.list_templates()
        assert len(templates) >= 4

    def test_registry_url_configurable(self) -> None:
        custom_url = "https://custom.example.com/registry.json"
        registry = TemplateRegistry(registry_url=custom_url, _http_client=None)
        assert registry._registry_url == custom_url

    def test_index_cached_within_ttl(self) -> None:
        call_count = 0

        def counting_client(url: str) -> str:
            nonlocal call_count
            call_count += 1
            return _make_index_json([])

        registry = TemplateRegistry(cache_ttl_seconds=60, _http_client=counting_client)
        registry.list_templates()
        registry.list_templates()
        # Should only fetch once within TTL
        assert call_count == 1

    def test_remote_templates_merged_with_builtins(self) -> None:
        remote_entry = {
            "name": "remote-template",
            "version": "1.0.0",
            "description": "A remote template",
            "tags": ["remote"],
            "category": "general",
            "author": "remote-author",
            "download_url": "https://example.com/remote.zip",
            "manifest_url": "https://example.com/remote-manifest.yaml",
        }

        def mock_client(url: str) -> str:
            return _make_index_json([remote_entry])

        registry = TemplateRegistry(_http_client=mock_client)
        names = [t.name for t in registry.list_templates()]
        assert "remote-template" in names
        assert "simple-agent" in names  # built-ins still present
