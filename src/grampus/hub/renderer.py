"""Safe {{variable}} substitution for template rendering."""

from __future__ import annotations

import re

from grampus.hub.manifest import TemplateManifest

# Only match valid Python identifiers — prevents code injection via expressions
_VAR_PATTERN = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")


def render_content(content: str, variables: dict[str, str]) -> str:
    """Replace {{variable_name}} placeholders in content.

    Unknown variables are left as-is. Variable names must match
    [a-zA-Z_][a-zA-Z0-9_]* — any other syntax is never substituted.
    """

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return variables.get(name, match.group(0))

    return _VAR_PATTERN.sub(_replace, content)


def render_filename(filename: str, variables: dict[str, str]) -> str:
    """Apply variable substitution to a file or directory name."""
    return render_content(filename, variables)


def collect_variables(
    manifest: TemplateManifest,
    overrides: dict[str, str],
) -> dict[str, str]:
    """Merge manifest parameter defaults with user-provided overrides.

    Raises ValueError listing any required parameters that have no value.
    Raises ValueError if a provided value is not in the parameter's choices list.
    """
    result: dict[str, str] = {}
    errors: list[str] = []

    for param in manifest.parameters:
        if param.name in overrides:
            value = overrides[param.name]
        elif param.default is not None:
            value = param.default
        elif param.required:
            errors.append(f"required parameter '{param.name}' not provided")
            continue
        else:
            continue

        if param.choices is not None and value not in param.choices:
            errors.append(
                f"parameter '{param.name}' value {value!r} not in choices {param.choices}"
            )
            continue

        result[param.name] = value

    if errors:
        raise ValueError("; ".join(errors))

    return result
