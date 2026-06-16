"""Template installer — download, render variables, write to output_dir."""

from __future__ import annotations

import io
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from grampus.hub.manifest import TemplateManifest
from grampus.hub.registry import TemplateIndexEntry, TemplateRegistry
from grampus.hub.renderer import collect_variables, render_content, render_filename


class InstallResult(BaseModel):
    """Result of a successful template installation."""

    template_name: str
    output_dir: Path
    files_written: list[str]
    entry_point: Path
    variables_used: dict[str, str]


class TemplateInstaller:
    """Downloads (or locates built-in) a template and writes it to disk."""

    def __init__(
        self,
        registry: TemplateRegistry,
        _http_client: Callable[[str], bytes] | None = None,
        force: bool = False,
    ) -> None:
        self._registry = registry
        self._http_client = _http_client
        self._force = force

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def install(
        self,
        template_name: str,
        output_dir: Path,
        variables: dict[str, str] | None = None,
        dry_run: bool = False,
    ) -> InstallResult:
        """Install a template by name from the registry.

        Looks up built-in templates first; falls back to remote download.
        """
        builtin_dir = self._registry.builtin_template_dir(template_name)
        if builtin_dir is not None:
            return self.install_local(builtin_dir, output_dir, variables=variables, dry_run=dry_run)

        entry = self._registry.get(template_name)
        if entry is None:
            raise KeyError(f"Template '{template_name}' not found in registry")

        with tempfile.TemporaryDirectory() as tmp:
            template_dir = self._download_template(entry, Path(tmp))
            return self.install_local(
                template_dir, output_dir, variables=variables, dry_run=dry_run
            )

    def install_local(
        self,
        template_dir: Path,
        output_dir: Path,
        variables: dict[str, str] | None = None,
        dry_run: bool = False,
    ) -> InstallResult:
        """Install from a local directory containing grampus-template.yaml."""
        manifest_path = template_dir / "grampus-template.yaml"
        manifest = TemplateManifest.from_yaml(manifest_path)

        resolved = collect_variables(manifest, variables or {})

        if not dry_run:
            self._check_output_dir(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        files_written = self._render_and_write(template_dir, output_dir, resolved, dry_run)

        entry_point = output_dir / render_filename(manifest.entry_point, resolved)
        return InstallResult(
            template_name=manifest.name,
            output_dir=output_dir,
            files_written=files_written,
            entry_point=entry_point,
            variables_used=resolved,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_output_dir(self, output_dir: Path) -> None:
        if output_dir.exists() and any(output_dir.iterdir()) and not self._force:
            raise ValueError(
                f"Output directory '{output_dir}' already exists and is non-empty. "
                "Use force=True to overwrite."
            )

    def _download_template(self, entry: TemplateIndexEntry, dest: Path) -> Path:
        """Download zip archive, extract, return path to template directory."""
        if self._http_client is not None:
            raw = self._http_client(entry.download_url)
        else:
            with urllib.request.urlopen(entry.download_url, timeout=30) as resp:  # noqa: S310
                raw = resp.read()

        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            zf.extractall(dest)

        # Zip archives often have a single top-level directory
        subdirs = [p for p in dest.iterdir() if p.is_dir()]
        if len(subdirs) == 1 and (subdirs[0] / "grampus-template.yaml").exists():
            return subdirs[0]
        if (dest / "grampus-template.yaml").exists():
            return dest
        # Fallback: return the dest itself
        return dest

    def _render_and_write(
        self,
        template_dir: Path,
        output_dir: Path,
        variables: dict[str, str],
        dry_run: bool,
    ) -> list[str]:
        """Render all template files and (if not dry_run) write to output_dir."""
        manifest_path = template_dir / "grampus-template.yaml"
        manifest = TemplateManifest.from_yaml(manifest_path)

        # Use manifest.files as the allowlist if non-empty; else scan directory
        if manifest.files:
            file_list = manifest.files
        else:
            file_list = [
                str(p.relative_to(template_dir))
                for p in template_dir.rglob("*")
                if p.is_file() and p.name != "grampus-template.yaml"
            ]

        written: list[str] = []
        for rel_path in file_list:
            # Security: skip path traversal and .git/
            if ".." in rel_path or rel_path.startswith(".git/") or rel_path == ".git":
                continue

            src = template_dir / rel_path
            if not src.exists() or not src.is_file():
                continue

            rendered_rel = render_filename(rel_path, variables)
            dest = output_dir / rendered_rel

            content = src.read_text(encoding="utf-8", errors="replace")
            rendered = render_content(content, variables)

            if not dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(rendered, encoding="utf-8")

            written.append(rendered_rel)

        return written
