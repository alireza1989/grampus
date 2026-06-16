"""Output tools: save_report, format_markdown, word_count.

save_report   — persists final report to demos/deep_research/output/
format_markdown — builds structured markdown from section dict
word_count      — full text statistics including Flesch readability score
"""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from grampus.core.types import ToolParameter
from grampus.tools.registry import ToolRegistry

_registry = ToolRegistry()

_DEFAULT_OUTPUT_DIR = str(Path(__file__).parent.parent / "output")

# Markdown section order for format_markdown
_SECTION_ORDER = [
    "Executive Summary",
    "Key Findings",
    "Detailed Analysis",
    "Limitations & Caveats",
    "Citations",
    "Confidence Assessment",
]


def _slugify(text: str) -> str:
    """Convert title to filesystem-safe slug."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_-]+", "_", text).strip("_")[:60]


def _count_syllables(word: str) -> int:
    """Approximate syllable count using vowel-group heuristic."""
    word = word.lower().strip(".,;:!?\"'")
    if not word:
        return 0
    vowels = re.findall(r"[aeiouy]+", word)
    count = len(vowels)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


@_registry.tool(
    name="save_report",
    description="Save a research report to the output directory as a markdown or JSON file.",
    parameters=[
        ToolParameter(name="title", type="string", description="Report title", required=True),
        ToolParameter(name="content", type="string", description="Report content", required=True),
        ToolParameter(
            name="format",
            type="string",
            description="Output format: 'markdown' or 'json'",
            required=False,
            default="markdown",
            enum=["markdown", "json"],
        ),
        ToolParameter(
            name="output_dir",
            type="string",
            description="Directory to write the report",
            required=False,
        ),
    ],
)
async def save_report(
    title: str,
    content: str,
    format: str = "markdown",
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Write the report to disk and return file metadata."""
    out_dir = Path(output_dir) if output_dir else Path(_DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    slug = _slugify(title)
    ext = "md" if format == "markdown" else "json"
    filepath = out_dir / f"{ts}_{slug}.{ext}"

    if format == "json":
        import json

        data = {"title": title, "content": content, "created_at": ts}
        filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        filepath.write_text(content, encoding="utf-8")

    words = len(content.split())
    return {
        "filepath": str(filepath),
        "word_count": words,
        "created_at": datetime.now(UTC).isoformat(),
    }


@_registry.tool(
    name="format_markdown",
    description="Format a dict of sections into a structured markdown document.",
    parameters=[
        ToolParameter(
            name="sections",
            type="object",
            description="Mapping of section_name → content string",
            required=True,
        ),
    ],
)
async def format_markdown(sections: dict[str, Any]) -> dict[str, Any]:
    """Build a structured markdown document from a section dict."""
    lines: list[str] = []

    # Write sections in preferred order, then any remaining
    written: set[str] = set()
    for name in _SECTION_ORDER:
        if name in sections:
            lines.append(f"## {name}\n")
            lines.append(str(sections[name]))
            lines.append("")
            written.add(name)

    for name, body in sections.items():
        if name not in written:
            lines.append(f"## {name}\n")
            lines.append(str(body))
            lines.append("")

    markdown = "\n".join(lines).strip()
    words = len(markdown.split())
    return {
        "markdown": markdown,
        "word_count": words,
        "section_count": len(sections),
    }


@_registry.tool(
    name="word_count",
    description="Count words, sentences, paragraphs, and compute Flesch readability score.",
    parameters=[
        ToolParameter(name="text", type="string", description="Text to analyse", required=True),
    ],
)
async def word_count(text: str) -> dict[str, Any]:
    """Return detailed text statistics including Flesch reading ease."""
    words_list = re.findall(r"\b\w+\b", text)
    n_words = len(words_list)

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s.strip()]
    n_sentences = max(len(sentences), 1)

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    n_paragraphs = max(len(paragraphs), 1)

    n_syllables = sum(_count_syllables(w) for w in words_list)

    reading_time = round(n_words / 200, 1)  # 200 wpm average reader

    # Flesch Reading Ease formula
    if n_words > 0 and n_sentences > 0:
        flesch = round(
            206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / max(n_words, 1)),
            1,
        )
        flesch = min(max(flesch, 0), 100)
    else:
        flesch = 50.0

    return {
        "words": n_words,
        "sentences": n_sentences,
        "paragraphs": n_paragraphs,
        "reading_time_minutes": reading_time,
        "flesch_score": flesch,
    }
