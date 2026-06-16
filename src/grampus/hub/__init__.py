"""Nexus Hub — template gallery for agent projects."""

from grampus.hub.installer import InstallResult, TemplateInstaller
from grampus.hub.manifest import TemplateManifest, TemplateParameter
from grampus.hub.registry import TemplateIndex, TemplateIndexEntry, TemplateRegistry
from grampus.hub.renderer import collect_variables, render_content, render_filename

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
