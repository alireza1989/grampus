"""EvalRunStore — in-memory persistence for evaluation run records."""

from __future__ import annotations

import uuid
from collections import deque
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class EvalRunRecord(BaseModel):
    """Persisted record of one eval suite run.

    Attributes:
        run_id: Unique run identifier (UUID).
        suite_name: Name of the EvalSuite.
        run_at: UTC timestamp of the run start.
        pass_rate: Fraction of cases that passed (0–1).
        passed: Number of passing cases.
        failed: Number of failing cases.
        errors: Number of cases that raised exceptions.
        total_cases: Total cases executed.
        total_cost_usd: Aggregate cost for all cases.
        avg_duration_seconds: Mean per-case wall time.
        case_results: Serialised per-case results for export and detail view.
    """

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    suite_name: str
    run_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    pass_rate: float
    passed: int
    failed: int
    errors: int
    total_cases: int
    total_cost_usd: float
    avg_duration_seconds: float
    case_results: list[dict[str, Any]] = Field(default_factory=list)


class EvalRunStore:
    """In-memory store for EvalRunRecords.

    Records are stored in insertion order; queries return newest-first.

    Args:
        max_runs: Maximum records to retain (oldest evicted first).
    """

    def __init__(self, max_runs: int = 1000) -> None:
        self._records: deque[EvalRunRecord] = deque(maxlen=max_runs)

    def append(self, record: EvalRunRecord) -> None:
        """Append a record to the store.

        Args:
            record: The EvalRunRecord to persist.
        """
        self._records.append(record)

    def from_suite_result(self, suite_result: Any) -> EvalRunRecord:
        """Build an EvalRunRecord from a SuiteResult.

        Args:
            suite_result: A completed SuiteResult instance.

        Returns:
            An EvalRunRecord populated from the SuiteResult fields.
        """
        case_dicts = [
            cr.model_dump(mode="json") if hasattr(cr, "model_dump") else dict(cr)
            for cr in suite_result.case_results
        ]
        return EvalRunRecord(
            suite_name=suite_result.suite_name,
            run_at=suite_result.run_at,
            pass_rate=suite_result.pass_rate,
            passed=suite_result.passed,
            failed=suite_result.failed,
            errors=suite_result.errors,
            total_cases=suite_result.total_cases,
            total_cost_usd=suite_result.total_cost_usd,
            avg_duration_seconds=suite_result.avg_duration_seconds,
            case_results=case_dicts,
        )

    def list_runs(
        self,
        suite_name: str | None = None,
        limit: int = 50,
    ) -> list[EvalRunRecord]:
        """Return runs newest-first, optionally filtered by suite_name.

        Args:
            suite_name: If set, only return runs for this suite.
            limit: Maximum records to return.

        Returns:
            List of EvalRunRecords, newest first.
        """
        records: list[EvalRunRecord] = list(reversed(self._records))
        if suite_name:
            records = [r for r in records if r.suite_name == suite_name]
        return records[:limit]

    def get(self, run_id: str) -> EvalRunRecord | None:
        """Return the record with the given run_id, or None.

        Args:
            run_id: The UUID of the run to fetch.
        """
        for r in self._records:
            if r.run_id == run_id:
                return r
        return None

    def list_suite_names(self) -> list[str]:
        """Return sorted list of unique suite names seen in the store."""
        return sorted({r.suite_name for r in self._records})
