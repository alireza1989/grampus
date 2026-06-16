"""Tests for grampus.memory.auditor — AuditReport, MemoryAuditor."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from grampus.memory.auditor import AuditReport, MemoryAuditor
from grampus.memory.provenance import ProvenanceTracker, SourceType
from grampus.memory.types import EpisodicRecord


def _record(content: str = "normal content", provenance: str | None = None) -> EpisodicRecord:
    return EpisodicRecord(
        id=str(uuid.uuid4()),
        agent_id="agent-1",
        session_id="sess-1",
        content=content,
        provenance=provenance,
    )


def _prov_json(content: str, source_type: SourceType = SourceType.USER_INPUT) -> str:
    tracker = ProvenanceTracker()
    return tracker.create(content, source_type, source_id="test").model_dump_json()


@pytest.fixture()
def mock_episodic() -> AsyncMock:
    em = AsyncMock()
    em.list_all = AsyncMock(return_value=[])
    return em


class TestAuditReport:
    def test_fields(self) -> None:
        report = AuditReport(
            agent_id="agent-1",
            audited_at=datetime.now(UTC),
            total_records=10,
            tampered_ids=[],
            missing_provenance_ids=[],
            integrity_score=1.0,
        )
        assert report.total_records == 10
        assert report.integrity_score == 1.0

    def test_json_round_trip(self) -> None:
        report = AuditReport(
            agent_id="a",
            audited_at=datetime.now(UTC),
            total_records=5,
            tampered_ids=["id-1"],
            missing_provenance_ids=["id-2"],
            integrity_score=0.6,
        )
        restored = AuditReport.model_validate_json(report.model_dump_json())
        assert restored.integrity_score == pytest.approx(0.6)
        assert "id-1" in restored.tampered_ids
        assert "id-2" in restored.missing_provenance_ids


class TestMemoryAuditor:
    async def test_empty_store_returns_perfect_score(self, mock_episodic: AsyncMock) -> None:
        auditor = MemoryAuditor(
            episodic_memory=mock_episodic,
            provenance_tracker=ProvenanceTracker(),
            agent_id="agent-1",
        )
        report = await auditor.audit()
        assert report.integrity_score == 1.0
        assert report.total_records == 0

    async def test_all_good_records_pass(self, mock_episodic: AsyncMock) -> None:
        content = "hello world"
        rec = _record(content, provenance=_prov_json(content))
        mock_episodic.list_all.return_value = [rec]
        auditor = MemoryAuditor(
            episodic_memory=mock_episodic,
            provenance_tracker=ProvenanceTracker(),
            agent_id="agent-1",
        )
        report = await auditor.audit()
        assert report.total_records == 1
        assert report.tampered_ids == []
        assert report.missing_provenance_ids == []
        assert report.integrity_score == pytest.approx(1.0)

    async def test_detects_missing_provenance(self, mock_episodic: AsyncMock) -> None:
        rec = _record("no provenance here", provenance=None)
        mock_episodic.list_all.return_value = [rec]
        auditor = MemoryAuditor(
            episodic_memory=mock_episodic,
            provenance_tracker=ProvenanceTracker(),
            agent_id="agent-1",
        )
        report = await auditor.audit()
        assert rec.id in report.missing_provenance_ids
        assert report.integrity_score < 1.0

    async def test_detects_tampered_content(self, mock_episodic: AsyncMock) -> None:
        original = "original content"
        tampered_rec = _record("this content was changed", provenance=_prov_json(original))
        mock_episodic.list_all.return_value = [tampered_rec]
        auditor = MemoryAuditor(
            episodic_memory=mock_episodic,
            provenance_tracker=ProvenanceTracker(),
            agent_id="agent-1",
        )
        report = await auditor.audit()
        assert tampered_rec.id in report.tampered_ids
        assert report.integrity_score < 1.0

    async def test_integrity_score_is_fraction_clean(self, mock_episodic: AsyncMock) -> None:
        good = "good content"
        good_rec = _record(good, provenance=_prov_json(good))
        bad_rec = _record("bad", provenance=None)
        mock_episodic.list_all.return_value = [good_rec, bad_rec]
        auditor = MemoryAuditor(
            episodic_memory=mock_episodic,
            provenance_tracker=ProvenanceTracker(),
            agent_id="agent-1",
        )
        report = await auditor.audit()
        assert report.total_records == 2
        assert report.integrity_score == pytest.approx(0.5)

    async def test_audit_reports_correct_agent_id(self, mock_episodic: AsyncMock) -> None:
        auditor = MemoryAuditor(
            episodic_memory=mock_episodic,
            provenance_tracker=ProvenanceTracker(),
            agent_id="my-agent",
        )
        report = await auditor.audit()
        assert report.agent_id == "my-agent"

    async def test_audit_sets_utc_timestamp(self, mock_episodic: AsyncMock) -> None:
        auditor = MemoryAuditor(
            episodic_memory=mock_episodic,
            provenance_tracker=ProvenanceTracker(),
            agent_id="a",
        )
        report = await auditor.audit()
        assert report.audited_at.tzinfo is not None

    async def test_multiple_tampered_records(self, mock_episodic: AsyncMock) -> None:
        original = "original"
        rec1 = _record("tampered 1", provenance=_prov_json(original))
        rec2 = _record("tampered 2", provenance=_prov_json(original))
        mock_episodic.list_all.return_value = [rec1, rec2]
        auditor = MemoryAuditor(
            episodic_memory=mock_episodic,
            provenance_tracker=ProvenanceTracker(),
            agent_id="a",
        )
        report = await auditor.audit()
        assert len(report.tampered_ids) == 2
        assert report.integrity_score == pytest.approx(0.0)
