"""Template registry — index fetch, search, list, get."""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from nexus.hub.manifest import TemplateManifest

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_URL = (
    "https://raw.githubusercontent.com/nexus-ai/nexus-templates/main/registry.json"
)

# Path to bundled templates shipped with the package
_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


class TemplateIndexEntry(BaseModel):
    """Metadata for one template in the registry index."""

    name: str
    version: str
    description: str
    tags: list[str] = Field(default_factory=list)
    category: str = "general"
    author: str = "community"
    download_url: str = ""
    manifest_url: str = ""
    stars: int = 0
    downloads: int = 0


class TemplateIndex(BaseModel):
    """Full registry index fetched from the remote URL."""

    version: str
    updated_at: str
    templates: list[TemplateIndexEntry] = Field(default_factory=list)


class TemplateRegistry:
    """Provides template discovery: list, search, get, and manifest fetch."""

    def __init__(
        self,
        registry_url: str = DEFAULT_REGISTRY_URL,
        cache_ttl_seconds: int = 3600,
        _http_client: Callable[[str], str] | None = None,
    ) -> None:
        self._registry_url = registry_url
        self._cache_ttl = cache_ttl_seconds
        self._http_client = _http_client
        self._cached_index: TemplateIndex | None = None
        self._cache_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_templates(
        self,
        category: str | None = None,
        tag: str | None = None,
    ) -> list[TemplateIndexEntry]:
        """List available templates, optionally filtered by category or tag."""
        entries = self._load_index().templates
        if category:
            entries = [e for e in entries if e.category == category]
        if tag:
            entries = [e for e in entries if tag in e.tags]
        return entries

    def search(self, query: str) -> list[TemplateIndexEntry]:
        """Case-insensitive search across name, description, and tags."""
        q = query.lower()
        results = []
        for entry in self._load_index().templates:
            haystack = " ".join([entry.name, entry.description] + entry.tags).lower()
            if q in haystack:
                results.append(entry)
        return results

    def get(self, name: str) -> TemplateIndexEntry | None:
        """Exact name match; returns None if not found."""
        for entry in self._load_index().templates:
            if entry.name == name:
                return entry
        return None

    def fetch_manifest(self, name: str) -> TemplateManifest:
        """Fetch and parse the full nexus-template.yaml for a template.

        Built-in templates are loaded directly from the package; remote
        templates are fetched via HTTP.
        """
        builtin = _TEMPLATES_DIR / name / "nexus-template.yaml"
        if builtin.exists():
            return TemplateManifest.from_yaml(builtin)

        entry = self.get(name)
        if entry is None:
            raise KeyError(f"Template '{name}' not found in registry")
        if not entry.manifest_url:
            raise ValueError(f"Template '{name}' has no manifest_url")

        raw = self._fetch_url(entry.manifest_url)

        import yaml

        data: Any = yaml.safe_load(raw)
        return TemplateManifest.model_validate(data)

    def builtin_template_dir(self, name: str) -> Path | None:
        """Return the on-disk directory of a built-in template, or None."""
        path = _TEMPLATES_DIR / name
        return path if path.is_dir() else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> TemplateIndex:
        """Fetch registry.json, cache in memory; fall back to builtins on failure."""
        now = time.monotonic()
        if self._cached_index is not None and (now - self._cache_time) < self._cache_ttl:
            return self._cached_index

        builtins = self._load_builtin_templates()
        remote: list[TemplateIndexEntry] = []

        if self._http_client is not None:
            try:
                raw = self._fetch_url(self._registry_url)
                data = json.loads(raw)
                index = TemplateIndex.model_validate(data)
                # Remote entries that don't shadow built-in names
                builtin_names = {e.name for e in builtins}
                remote = [e for e in index.templates if e.name not in builtin_names]
            except Exception as exc:
                logger.warning("registry fetch failed, using built-in templates only: %s", exc)

        all_entries = builtins + remote
        self._cached_index = TemplateIndex(
            version="local",
            updated_at="",
            templates=all_entries,
        )
        self._cache_time = now
        return self._cached_index

    def _load_builtin_templates(self) -> list[TemplateIndexEntry]:
        """Scan the bundled templates/ directory for nexus-template.yaml files."""
        entries: list[TemplateIndexEntry] = []
        if not _TEMPLATES_DIR.is_dir():
            return entries
        for manifest_path in sorted(_TEMPLATES_DIR.glob("*/nexus-template.yaml")):
            try:
                manifest = TemplateManifest.from_yaml(manifest_path)
                entries.append(
                    TemplateIndexEntry(
                        name=manifest.name,
                        version=manifest.version,
                        description=manifest.description,
                        tags=manifest.tags,
                        category=manifest.category,
                        author=manifest.author,
                        download_url="",
                        manifest_url="",
                    )
                )
            except Exception as exc:
                logger.warning("failed to load built-in template %s: %s", manifest_path, exc)
        return entries

    def _fetch_url(self, url: str) -> str:
        """HTTP GET via injected client or stdlib urllib."""
        if self._http_client is not None:
            return self._http_client(url)
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            raw: bytes = resp.read()
            return raw.decode()
