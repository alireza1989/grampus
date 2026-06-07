"""Template manifest model — loaded from nexus-template.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class TemplateParameter(BaseModel):
    """A configurable parameter in a template."""

    name: str
    description: str
    default: str | None = None
    required: bool = False
    choices: list[str] | None = None


class TemplateManifest(BaseModel):
    """Full manifest for a Nexus agent template."""

    name: str
    version: str
    description: str
    long_description: str = ""
    tags: list[str] = Field(default_factory=list)
    category: str = "general"
    author: str = "community"
    min_nexus_version: str = "0.1.0"
    required_tools: list[str] = Field(default_factory=list)
    required_models: list[str] = Field(default_factory=list)
    parameters: list[TemplateParameter] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    entry_point: str = "agent.py"

    @classmethod
    def from_yaml(cls, path: Path) -> TemplateManifest:
        """Load manifest from nexus-template.yaml using yaml.safe_load()."""
        data: Any = yaml.safe_load(path.read_text())
        return cls.model_validate(data)

    def get_parameter(self, name: str) -> TemplateParameter | None:
        """Return the parameter with the given name, or None if not found."""
        for p in self.parameters:
            if p.name == name:
                return p
        return None
