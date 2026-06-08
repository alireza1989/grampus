"""Schema validator for artifact section content."""

from __future__ import annotations

import json
from typing import Any

from nexus.orchestration.artifact.types import (
    ArtifactContentType,
    ArtifactSection,
    ConflictType,
    SectionConflict,
    SectionSchema,
)


class SchemaValidator:
    """Validates section content against its SectionSchema before committing.

    Checks performed in order:
    1. Content type matches (str for text/markdown/code, dict for json)
    2. Required fields present (JSON sections only)
    3. Max tokens not exceeded (~len(str(content)) / 4)
    """

    def validate(
        self,
        section: ArtifactSection,
        schema: SectionSchema,
    ) -> SectionConflict | None:
        """Validate section content against its schema.

        Args:
            section: Section containing the content to validate.
            schema: Schema specification the content must satisfy.

        Returns:
            SectionConflict describing the first violation, or None if valid.
        """
        content = section.content
        if content is None:
            return None

        if not self._check_type(content, schema.content_type):
            expected = "str" if schema.content_type != ArtifactContentType.JSON else "dict"
            return SectionConflict(
                section_id=section.section_id,
                conflict_type=ConflictType.SCHEMA_VALIDATION,
                description=(
                    f"Section '{section.section_id}' expects {schema.content_type.value} "
                    f"(requires {expected}), got {type(content).__name__}"
                ),
                resolution="reject",
            )

        if schema.content_type == ArtifactContentType.JSON and isinstance(content, dict):
            missing = self._check_required_fields(content, schema.required_fields)
            if missing:
                return SectionConflict(
                    section_id=section.section_id,
                    conflict_type=ConflictType.SCHEMA_VALIDATION,
                    description=(
                        f"Section '{section.section_id}' missing required fields: "
                        f"{', '.join(missing)}"
                    ),
                    resolution="reject",
                )

        if schema.max_tokens is not None:
            estimated = self._estimate_tokens(content)
            if estimated > schema.max_tokens:
                return SectionConflict(
                    section_id=section.section_id,
                    conflict_type=ConflictType.SCHEMA_VALIDATION,
                    description=(
                        f"Section '{section.section_id}' estimated {estimated} tokens "
                        f"exceeds max_tokens={schema.max_tokens}"
                    ),
                    resolution="reject",
                )

        return None

    def _check_type(self, content: Any, expected: ArtifactContentType) -> bool:
        if expected == ArtifactContentType.JSON:
            return isinstance(content, (dict, list))
        return isinstance(content, str)

    def _check_required_fields(self, content: dict[str, Any], required: list[str]) -> list[str]:
        """Return list of required field names missing from content."""
        return [field for field in required if field not in content]

    def _estimate_tokens(self, content: Any) -> int:
        """Rough token estimate: len(str(content)) // 4."""
        if isinstance(content, str):
            return len(content) // 4
        return len(json.dumps(content)) // 4
