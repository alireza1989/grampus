"""Memory auditor: periodic integrity scan of episodic records."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from grampus.core.logging import get_logger
from grampus.memory.provenance import Provenance, ProvenanceTracker

_log = get_logger(__name__)


class AuditReport(BaseModel):
    """Results of a single audit pass over all episodic records.

    Args:
        agent_id: The agent whose records were audited.
        audited_at: UTC timestamp when the audit ran.
        total_records: Total number of records inspected.
        tampered_ids: IDs of records whose content hash did not match provenance.
        missing_provenance_ids: IDs of records with no provenance metadata.
        integrity_score: Fraction of records that passed all checks (0–1).
    """

    agent_id: str
    audited_at: datetime
    total_records: int
    tampered_ids: list[str]
    missing_provenance_ids: list[str]
    integrity_score: float


class MemoryAuditor:
    """Scans all episodic records and verifies content-hash integrity.

    Args:
        episodic_memory: Episodic store to audit (must expose ``list_all()``).
        provenance_tracker: Used to re-derive expected hashes.
        agent_id: Agent namespace being audited.
    """

    def __init__(
        self,
        episodic_memory: Any,
        provenance_tracker: ProvenanceTracker,
        *,
        agent_id: str,
    ) -> None:
        self._episodic = episodic_memory
        self._tracker = provenance_tracker
        self._agent_id = agent_id

    async def audit(self) -> AuditReport:
        """Load all episodic records and check each for provenance integrity.

        A record is flagged as *tampered* when its provenance JSON is present but
        the stored content no longer matches the hash. It is flagged as
        *missing_provenance* when the ``provenance`` field is ``None`` or empty.

        Returns:
            An :class:`AuditReport` with counts and the fraction of clean records.
        """
        records = await self._episodic.list_all()
        tampered: list[str] = []
        missing: list[str] = []

        for record in records:
            if not record.provenance:
                missing.append(record.id)
                _log.warning("audit_missing_provenance", record_id=record.id)
                continue

            try:
                prov = Provenance.model_validate_json(record.provenance)
            except Exception:
                missing.append(record.id)
                _log.warning("audit_unparseable_provenance", record_id=record.id)
                continue

            if not self._tracker.verify(record.content, prov):
                tampered.append(record.id)
                _log.warning("audit_tampered_record", record_id=record.id)

        total = len(records)
        bad = len(tampered) + len(missing)
        integrity = 1.0 if total == 0 else (total - bad) / total

        _log.info(
            "audit_complete",
            agent=self._agent_id,
            total=total,
            tampered=len(tampered),
            missing=len(missing),
            integrity_score=integrity,
        )

        return AuditReport(
            agent_id=self._agent_id,
            audited_at=datetime.now(UTC),
            total_records=total,
            tampered_ids=tampered,
            missing_provenance_ids=missing,
            integrity_score=integrity,
        )
