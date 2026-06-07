"""Nexus Hub — template gallery for agent projects."""

from nexus.hub.installer import InstallResult, TemplateInstaller
from nexus.hub.manifest import TemplateManifest, TemplateParameter
from nexus.hub.registry import TemplateIndex, TemplateIndexEntry, TemplateRegistry
from nexus.hub.renderer import collect_variables, render_content, render_filename

__all__ = [
    "InstallResult",
    "TemplateInstaller",
    "TemplateManifest",
    "TemplateParameter",
    "TemplateIndex",
    "TemplateIndexEntry",
    "TemplateRegistry",
    "collect_variables",
    "render_content",
    "render_filename",
]
