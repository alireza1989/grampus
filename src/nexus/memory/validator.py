"""Memory write validation: injection detection, size limits, rate limiting."""

from __future__ import annotations

import re
import time
from collections import defaultdict, deque

from pydantic import BaseModel

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore.*instructions", re.IGNORECASE),
    re.compile(r"forget everything", re.IGNORECASE),
    re.compile(r"in future conversations", re.IGNORECASE),
    re.compile(r"remember that.*always", re.IGNORECASE),
    re.compile(r"always respond with", re.IGNORECASE),
    re.compile(r"your new instructions", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"disregard", re.IGNORECASE),
    re.compile(r"override.*instructions", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you\s+(are|were)|a\s+)", re.IGNORECASE),
    re.compile(r"(no\s+restrictions|without\s+restrictions|unrestricted\s+ai)", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)", re.IGNORECASE),
]


class ValidationResult(BaseModel):
    """Result of a memory write validation check.

    Args:
        allowed: True when the write may proceed.
        reasons: Human-readable strings explaining each violation (empty when allowed).
    """

    allowed: bool
    reasons: list[str]


class MemoryValidator:
    """Pre-write validation pipeline for memory content.

    Checks in order:
    1. **Injection detection** — regex match against known prompt-injection patterns.
    2. **Size check** — content byte length vs. *max_content_bytes*.
    3. **Rate limit** — sliding-window counter per source_id vs. *max_writes_per_minute*.

    All failing checks are collected and returned together so callers see the full
    picture. A write is blocked if any check fails.

    Args:
        max_content_bytes: Maximum byte length of content (default 10 000).
        max_writes_per_minute: Maximum write attempts per source_id per 60 s (default 60).
    """

    def __init__(
        self,
        max_content_bytes: int = 10_000,
        max_writes_per_minute: int = 60,
    ) -> None:
        self._max_bytes = max_content_bytes
        self._max_writes = max_writes_per_minute
        self._rate_windows: dict[str, deque[float]] = defaultdict(deque)

    def validate(self, content: str, *, source_id: str) -> ValidationResult:
        """Run all validation checks against *content* from *source_id*.

        Args:
            content: Raw text about to be stored.
            source_id: Identifier of the source requesting the write.

        Returns:
            A :class:`ValidationResult` with ``allowed=True`` when all checks pass.
        """
        reasons: list[str] = []

        # 1. Injection detection
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(content):
                reasons.append("injection detected: suspicious instruction pattern in content")
                break

        # 2. Size check
        byte_len = len(content.encode())
        if byte_len > self._max_bytes:
            reasons.append(f"size exceeded: {byte_len} bytes > limit of {self._max_bytes} bytes")

        # 3. Sliding-window rate limit (always record the attempt)
        now = time.monotonic()
        window = self._rate_windows[source_id]
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()
        window.append(now)

        if len(window) > self._max_writes:
            reasons.append(
                f"rate limit exceeded: {len(window)} attempts in last 60s (max {self._max_writes})"
            )

        return ValidationResult(allowed=len(reasons) == 0, reasons=reasons)
