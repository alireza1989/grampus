"""Prompt injection detection with regex, heuristic, and keyword layers."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel

from grampus.core.logging import get_logger

_log = get_logger(__name__)

_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
        "role_override",
    ),
    (
        re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
        "role_override",
    ),
    (
        re.compile(r"forget\s+(everything|all)\s+(you('ve| have)\s+)?been\s+told", re.IGNORECASE),
        "role_override",
    ),
    (
        re.compile(r"\[SYSTEM\]|\[INST\]|\[\/INST\]|<\|system\|>|<\|user\|>", re.IGNORECASE),
        "delimiter_injection",
    ),
    (
        re.compile(r"###\s*(instruction|system|prompt|context)s?", re.IGNORECASE),
        "delimiter_injection",
    ),
    (
        re.compile(r"you\s+are\s+now\s+(a\s+)?(?!an?\s+AI|an?\s+assistant)", re.IGNORECASE),
        "persona_hijack",
    ),
    (re.compile(r"act\s+as\s+(if\s+you\s+(are|were)|a\s+)", re.IGNORECASE), "persona_hijack"),
    (re.compile(r"pretend\s+(you\s+are|to\s+be)", re.IGNORECASE), "persona_hijack"),
    (
        re.compile(
            r"(always|from\s+now\s+on|in\s+(all\s+)?future)\s+.{0,50}(remember|recall|know)",
            re.IGNORECASE,
        ),
        "memory_poison",
    ),
    (re.compile(r"remember\s+that\s+(you|your|from)", re.IGNORECASE), "memory_poison"),
    (
        re.compile(
            r"from\s+now\s+on\s+.{0,60}(must|should|will|always|never|say|do|be)\b",
            re.IGNORECASE,
        ),
        "memory_poison",
    ),
]

_ROLE_MARKER_RE = re.compile(r"\b(system|user|assistant)\s*:", re.IGNORECASE)
_BOUNDARY_RE = re.compile(r"^(\-{3,}|={3,}|\*{3,})$", re.MULTILINE)
_IMPERATIVE_VERBS_RE = re.compile(
    r"\b(ignore|disregard|forget|override|bypass|reveal|expose|output|print|show|do not|stop)\b",
    re.IGNORECASE,
)


class DetectionLevel(StrEnum):
    """Controls the confidence threshold for flagging injection."""

    STRICT = "strict"
    BALANCED = "balanced"
    PERMISSIVE = "permissive"


_THRESHOLDS: dict[DetectionLevel, float] = {
    DetectionLevel.STRICT: 0.3,
    DetectionLevel.BALANCED: 0.5,
    DetectionLevel.PERMISSIVE: 0.8,
}


class InjectionResult(BaseModel):
    """Result of a prompt injection check."""

    detected: bool
    confidence: float
    matched_patterns: list[str]
    level: DetectionLevel
    input_preview: str


class PromptInjectionDetector:
    """Multi-layer prompt injection detector.

    Three detection layers applied in order:
    1. Regex — known attack signatures (fast, zero false-negatives on known patterns)
    2. Heuristic — structural signals (role override attempts, instruction boundaries)
    3. Keyword — semantic markers without full NLP

    Args:
        level: Detection strictness. Controls the confidence threshold
            above which ``detected=True`` is returned.
            STRICT >= 0.3, BALANCED >= 0.5, PERMISSIVE >= 0.8
    """

    def __init__(self, level: DetectionLevel = DetectionLevel.BALANCED) -> None:
        self.level = level

    def check(self, text: str) -> InjectionResult:
        """Synchronous check — no I/O. Returns InjectionResult.

        Args:
            text: The text to inspect for injection attempts.

        Returns:
            InjectionResult with confidence score and matched patterns.
        """
        confidence, patterns = self._check_regex(text)
        confidence += self._check_heuristics(text)
        confidence = min(confidence, 1.0)
        threshold = _THRESHOLDS[self.level]
        return InjectionResult(
            detected=confidence >= threshold,
            confidence=confidence,
            matched_patterns=patterns,
            level=self.level,
            input_preview=text[:100],
        )

    def _check_regex(self, text: str) -> tuple[float, list[str]]:
        """Return (confidence_delta, matched_pattern_names) from regex layer."""
        confidence = 0.0
        matched: list[str] = []
        for pattern, label in _INJECTION_PATTERNS:
            if pattern.search(text):
                confidence += 0.5
                if label not in matched:
                    matched.append(label)
        return confidence, matched

    def _check_heuristics(self, text: str) -> float:
        """Return additive confidence delta from heuristic signals."""
        delta = 0.0
        role_count = len(_ROLE_MARKER_RE.findall(text))
        if role_count >= 2:
            delta += 0.2
        if _BOUNDARY_RE.search(text):
            delta += 0.15
        words = text.split()
        if words:
            imperative_count = len(_IMPERATIVE_VERBS_RE.findall(text))
            ratio = imperative_count / len(words)
            if ratio > 0.15:
                delta += 0.1
        quote_count = text.count('"""') + text.count("'''")
        if quote_count >= 2:
            delta += 0.1
        return delta
