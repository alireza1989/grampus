"""CLI commands for the Nexus Hub template gallery."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import click

from nexus.hub.installer import TemplateInstaller
from nexus.hub.manifest import TemplateManifest
from nexus.hub.registry import TemplateRegistry

_VAR_PATTERN = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")


@click.group()
def hub() -> None:
    """Browse, pull, and validate agent templates from the Nexus Hub."""


# ---------------------------------------------------------------------------
# nexus hub list
# ---------------------------------------------------------------------------


@hub.command("list")
@click.option("--category", default=None, help="Filter by category")
@click.option("--tag", default=None, help="Filter by tag")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
)
def list_templates(category: str | None, tag: str | None, fmt: str) -> None:
    """List available templates."""
    registry = TemplateRegistry()
    entries = registry.list_templates(category=category, tag=tag)

    if fmt == "json":
        click.echo(json.dumps([e.model_dump() for e in entries], indent=2))
        return

    if not entries:
        click.echo("No templates found.")
        return

    col_name = max(len(e.name) for e in entries)
    col_cat = max(len(e.category) for e in entries)
    col_tags = max(len(", ".join(e.tags)) for e in entries) if entries else 10
    col_name = max(col_name, 4)
    col_cat = max(col_cat, 8)
    col_tags = max(col_tags, 4)

    header = f"{'NAME':<{col_name}}  {'CATEGORY':<{col_cat}}  {'TAGS':<{col_tags}}  DESCRIPTION"
    click.echo(header)
    click.echo("-" * (len(header) + 20))

    for e in entries:
        tags_str = ", ".join(e.tags)
        click.echo(
            f"{e.name:<{col_name}}  {e.category:<{col_cat}}  {tags_str:<{col_tags}}  {e.description}"
        )


# ---------------------------------------------------------------------------
# nexus hub search
# ---------------------------------------------------------------------------


@hub.command("search")
@click.argument("query")
def search_templates(query: str) -> None:
    """Search templates by name, description, or tags."""
    registry = TemplateRegistry()
    results = registry.search(query)

    if not results:
        click.echo(f"No templates found matching '{query}'.")
        return

    for e in results:
        tags_str = ", ".join(e.tags)
        click.echo(f"  {e.name}  [{e.category}]  {tags_str}")
        click.echo(f"    {e.description}")
        click.echo()


# ---------------------------------------------------------------------------
# nexus hub info
# ---------------------------------------------------------------------------


@hub.command("info")
@click.argument("template_name")
def template_info(template_name: str) -> None:
    """Show detailed information about a template."""
    registry = TemplateRegistry()
    entry = registry.get(template_name)
    if entry is None:
        click.echo(f"Template '{template_name}' not found.", err=True)
        sys.exit(1)

    try:
        manifest = registry.fetch_manifest(template_name)
    except Exception:
        manifest = None

    click.echo(f"Name:        {entry.name}")
    click.echo(f"Version:     {entry.version}")
    click.echo(f"Category:    {entry.category}")
    click.echo(f"Author:      {entry.author}")
    click.echo(f"Tags:        {', '.join(entry.tags)}")
    click.echo(f"Description: {entry.description}")

    if manifest and manifest.long_description:
        click.echo()
        click.echo(manifest.long_description.strip())

    if manifest and manifest.parameters:
        click.echo()
        click.echo("Parameters:")
        for p in manifest.parameters:
            req = " (required)" if p.required else ""
            default = f" [default: {p.default}]" if p.default else ""
            choices = f" choices: {p.choices}" if p.choices else ""
            click.echo(f"  {p.name}{req}{default}{choices}")
            click.echo(f"    {p.description}")

    if manifest and manifest.required_tools:
        click.echo()
        click.echo(f"Required tools: {', '.join(manifest.required_tools)}")


# ---------------------------------------------------------------------------
# nexus hub pull
# ---------------------------------------------------------------------------


@hub.command("pull")
@click.argument("template_name")
@click.option("--output", "-o", default=None, help="Output directory (default: ./<template-name>)")
@click.option("--var", "-v", "vars_list", multiple=True, help="key=value variable overrides")
@click.option("--dry-run", is_flag=True, default=False, help="Show what would be written")
@click.option("--force", is_flag=True, default=False, help="Overwrite existing directory")
def pull_template(
    template_name: str,
    output: str | None,
    vars_list: tuple[str, ...],
    dry_run: bool,
    force: bool,
) -> None:
    """Download and install a template."""
    registry = TemplateRegistry()
    entry = registry.get(template_name)
    if entry is None:
        click.echo(f"Template '{template_name}' not found.", err=True)
        sys.exit(1)

    output_dir = Path(output) if output else Path.cwd() / template_name

    # Parse --var key=value pairs
    variables: dict[str, str] = {}
    for var in vars_list:
        if "=" not in var:
            click.echo(f"Invalid --var format: '{var}' (expected key=value)", err=True)
            sys.exit(1)
        k, _, v = var.partition("=")
        variables[k.strip()] = v.strip()

    # Fetch manifest to know what parameters are needed
    try:
        manifest = registry.fetch_manifest(template_name)
    except Exception as exc:
        click.echo(f"Failed to load manifest: {exc}", err=True)
        sys.exit(1)

    # Prompt interactively for required params not provided
    for param in manifest.parameters:
        if param.name not in variables and param.required:
            value = click.prompt(
                f"{param.description} [{param.name}]",
                default=param.default or "",
            )
            variables[param.name] = value

    installer = TemplateInstaller(registry=registry, force=force)

    try:
        result = installer.install(template_name, output_dir, variables=variables, dry_run=dry_run)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if dry_run:
        click.echo(f"[dry-run] Would write {len(result.files_written)} files to {output_dir}/:")
        for f in result.files_written:
            click.echo(f"  {f}")
        return

    click.echo(f"\n✓ Template '{template_name}' installed to {output_dir}/\n")
    click.echo("Next steps:")
    click.echo(f"  cd {output_dir.name}")
    click.echo(f"  nexus run {result.entry_point.name}")


# ---------------------------------------------------------------------------
# nexus hub validate
# ---------------------------------------------------------------------------


@hub.command("validate")
@click.argument("template_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def validate_template(template_dir: Path) -> None:
    """Validate a local nexus-template.yaml and its referenced files."""
    errors: list[str] = []

    click.echo(f"Validating template at {template_dir}/")

    # 1. Parse manifest
    manifest_path = template_dir / "nexus-template.yaml"
    if not manifest_path.exists():
        click.echo("  ✗ nexus-template.yaml — not found")
        sys.exit(1)

    try:
        manifest = TemplateManifest.from_yaml(manifest_path)
        click.echo("  ✓ nexus-template.yaml — valid")
    except Exception as exc:
        click.echo(f"  ✗ nexus-template.yaml — invalid: {exc}")
        sys.exit(1)

    # 2. Check all declared files exist
    missing_files: list[str] = []
    for fname in manifest.files:
        fpath = template_dir / fname
        if not fpath.exists():
            missing_files.append(fname)

    if missing_files:
        for f in missing_files:
            click.echo(f"  ✗ Missing file: {f}")
            errors.append(f"Missing file: {f}")
    else:
        click.echo(f"  ✓ Required files present: {', '.join(manifest.files)}")

    # 3. Check all {{variables}} in files are declared
    declared = {p.name for p in manifest.parameters}
    undeclared_vars: list[str] = []
    for fname in manifest.files:
        fpath = template_dir / fname
        if not fpath.exists():
            continue
        if fpath.suffix not in (".py", ".yaml", ".yml", ".md", ".txt", ".toml", ".json"):
            continue
        content = fpath.read_text(encoding="utf-8", errors="replace")
        for match in _VAR_PATTERN.finditer(content):
            var_name = match.group(1)
            if var_name not in declared:
                placeholder = "{{" + var_name + "}}"
                undeclared_vars.append(
                    f"{placeholder} found in {fname} (not in manifest parameters)"
                )
                errors.append(f"Undeclared variable: {placeholder} found in {fname}")

    if undeclared_vars:
        for msg in undeclared_vars:
            click.echo(f"  ✗ Undeclared variable: {msg}")
    else:
        click.echo("  ✓ All {{variables}} in files are declared in manifest")

    # 4. Check for path traversal
    traversal_found = False
    for fname in manifest.files:
        if ".." in fname or fname.startswith(".git/"):
            traversal_found = True
            errors.append(f"Path traversal in filename: {fname}")
            click.echo(f"  ✗ Path traversal in filename: {fname}")
    if not traversal_found:
        click.echo("  ✓ No path traversal in filenames")

    click.echo()
    if errors:
        click.echo(f"{len(errors)} error(s) found.")
        sys.exit(1)
    else:
        click.echo("Template is valid.")


# ---------------------------------------------------------------------------
# nexus hub push
# ---------------------------------------------------------------------------


@hub.command("push")
@click.argument("template_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def push_template(template_dir: Path) -> None:
    """Submit a template to the Nexus Hub (opens a GitHub PR)."""
    click.echo("To share your template with the community:\n")
    click.echo("1. Fork https://github.com/nexus-ai/nexus-templates")
    click.echo("2. Copy your template directory into the fork")
    click.echo("3. Run: nexus hub validate <your-template-dir>")
    click.echo("4. Open a Pull Request at:")
    click.echo("   https://github.com/nexus-ai/nexus-templates/pulls")
    click.echo()
    click.echo("Guidelines:")
    click.echo("  - Template must pass `nexus hub validate`")
    click.echo("  - Include a README.md with usage instructions")
    click.echo("  - Templates are reviewed within 5 business days")
