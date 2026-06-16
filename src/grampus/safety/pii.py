"""PII detection and redaction with configurable per-type actions."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel

from grampus.core.errors import SafetyError
from grampus.core.logging import get_logger

_log = get_logger(__name__)


class PIIType(StrEnum):
    """Supported PII categories."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"


class PIIAction(StrEnum):
    """Action to take when PII is detected."""

    LOG = "log"
    REDACT = "redact"
    BLOCK = "block"


_PII_PATTERNS: dict[PIIType, str] = {
    PIIType.EMAIL: r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    PIIType.PHONE: r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    PIIType.SSN: r"\b\d{3}-\d{2}-\d{4}\b",
    PIIType.CREDIT_CARD: (
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}"
        r"|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12})\b"
    ),
    PIIType.IP_ADDRESS: (
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    ),
    PIIType.DATE_OF_BIRTH: (
        r"\b(?:DOB|dob|date\s+of\s+birth)[:\s]+\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b"
    ),
}

_COMPILED: dict[PIIType, re.Pattern[str]] = {
    pii_type: re.compile(pattern) for pii_type, pattern in _PII_PATTERNS.items()
}


class PIIMatch(BaseModel):
    """A single PII match within the scanned text."""

    pii_type: PIIType
    start: int
    end: int
    redacted_value: str


class PIIResult(BaseModel):
    """Result of scanning text for PII."""

    original: str
    redacted: str
    matches: list[PIIMatch]
    action_taken: PIIAction
    blocked: bool = False


class PIIDetector:
    """Regex-based PII detector with configurable action per PII type.

    Args:
        actions: Map of PIIType -> PIIAction. Defaults to REDACT for all types.
            If a type is not in the map, defaults to LOG.
    """

    def __init__(self, actions: dict[PIIType, PIIAction] | None = None) -> None:
        if actions is None:
            self._actions: dict[PIIType, PIIAction] = {t: PIIAction.REDACT for t in PIIType}
        else:
            self._actions = actions

    def scan(self, text: str) -> PIIResult:
        """Scan text for PII. Returns PIIResult with redacted text if configured.

        Args:
            text: The text to scan.

        Returns:
            PIIResult with matches and possibly redacted text.

        Raises:
            SafetyError: code="PII_BLOCKED" if any matched type has action=BLOCK.
        """
        matches = self._find_matches(text)
        self._check_for_blocks(matches, text)
        overall_action, redacted = self._apply_actions(text, matches)
        return PIIResult(
            original=text,
            redacted=redacted,
            matches=matches,
            action_taken=overall_action,
            blocked=False,
        )

    def redact(self, text: str) -> str:
        """Scan and return redacted text.

        Args:
            text: The text to clean.

        Returns:
            Text with PII replaced by placeholders.

        Raises:
            SafetyError: code="PII_BLOCKED" if a BLOCK action is configured.
        """
        return self.scan(text).redacted

    def _find_matches(self, text: str) -> list[PIIMatch]:
        """Collect all PII matches across all types."""
        found: list[PIIMatch] = []
        for pii_type, pattern in _COMPILED.items():
            for m in pattern.finditer(text):
                found.append(
                    PIIMatch(
                        pii_type=pii_type,
                        start=m.start(),
                        end=m.end(),
                        redacted_value=f"[REDACTED:{pii_type.upper()}]",
                    )
                )
        return sorted(found, key=lambda x: x.start)

    def _check_for_blocks(self, matches: list[PIIMatch], text: str) -> None:
        """Raise SafetyError if any match has action=BLOCK."""
        for match in matches:
            action = self._actions.get(match.pii_type, PIIAction.LOG)
            if action == PIIAction.BLOCK:
                _log.warning(
                    "pii.blocked",
                    pii_type=match.pii_type,
                    preview=text[:50],
                )
                raise SafetyError(
                    f"PII type '{match.pii_type}' is blocked",
                    code="PII_BLOCKED",
                    details={"pii_type": match.pii_type},
                    hint="Remove personally identifiable information from the input or set pii_action to 'redact' in your safety policy.",
                )

    def _apply_actions(self, text: str, matches: list[PIIMatch]) -> tuple[PIIAction, str]:
        """Apply LOG or REDACT actions; return (dominant_action, result_text)."""
        if not matches:
            return PIIAction.LOG, text

        overall = PIIAction.LOG
        # Apply replacements from end to start to preserve offsets
        result = text
        for match in reversed(matches):
            action = self._actions.get(match.pii_type, PIIAction.LOG)
            if action == PIIAction.REDACT:
                overall = PIIAction.REDACT
                result = result[: match.start] + match.redacted_value + result[match.end :]
            else:
                _log.info("pii.logged", pii_type=match.pii_type)

        return overall, result
