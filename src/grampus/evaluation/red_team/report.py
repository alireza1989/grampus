"""RedTeamReport: deduplication, severity bucketing, structured output."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from grampus.core.logging import get_logger
from grampus.evaluation.red_team.types import (
    AttackCategory,
    AttackResult,
    OWASPCategory,
    RedTeamCampaignConfig,
    SecurityProperty,
    Severity,
)

_log = get_logger(__name__)

_SEVERITY_ORDER = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]


class RedTeamFinding(BaseModel):
    """A deduplicated, human-readable finding from the campaign."""

    finding_id: str
    attack_category: AttackCategory
    severity: Severity
    title: str
    description: str
    example_payload: str
    example_response: str
    violated_properties: list[SecurityProperty] = Field(default_factory=list)
    owasp_categories: list[OWASPCategory] = Field(default_factory=list)
    recommendation: str
    occurrence_count: int = 1


class RedTeamSummary(BaseModel):
    """Top-level campaign summary."""

    campaign_id: str
    agent_name: str
    total_attacks: int
    successful_attacks: int
    attack_success_rate: float
    findings: list[RedTeamFinding] = Field(default_factory=list)
    severity_counts: dict[str, int] = Field(default_factory=dict)
    owasp_coverage: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


_RECOMMENDATIONS: dict[AttackCategory, str] = {
    AttackCategory.PROMPT_INJECTION: (
        "Strengthen the PromptInjectionDetector in grampus.safety.injection. "
        "Add indirect injection detection for tool return values. "
        "Consider input sanitization before passing tool outputs to the LLM."
    ),
    AttackCategory.JAILBREAK: (
        "Add roleplay and persona-switching detection to the safety pipeline. "
        "Implement Base64 and common encoding detection in injection scanner. "
        "Raise injection detection sensitivity level."
    ),
    AttackCategory.REASONING_HIJACK: (
        "Add multi-turn conversation monitoring to detect gradual goal drift. "
        "Implement periodic goal-alignment checks in AgentRunner's ReAct loop. "
        "Consider limiting context window exposure in long sessions."
    ),
    AttackCategory.MEMORY_POISON: (
        "Ensure all memory writes pass through MemoryValidator with injection detection. "
        "Increase trust score requirements for memory writes from external sources. "
        "Audit ProvenanceTracker to ensure source_type is set correctly on all writes."
    ),
    AttackCategory.TOOL_MISUSE: (
        "Enable ActionGuard with tool call rate limiting. "
        "Add recursive call detection in ToolExecutor. "
        "Restrict tool chaining to pre-approved combinations via policy config."
    ),
    AttackCategory.EXCESSIVE_AGENCY: (
        "Review AgentDefinition.tools list — apply least-privilege principle. "
        "Add explicit action confirmation for high-impact tools. "
        "Enable cost budget enforcement to limit runaway tool usage."
    ),
}


class RedTeamReport:
    """
    Generates structured findings from a completed campaign.

    Deduplication: multiple results with the same attack_category and
    variant are grouped into one RedTeamFinding (keeps worst severity).
    """

    def build(
        self,
        config: RedTeamCampaignConfig,
        results: list[AttackResult],
    ) -> RedTeamSummary:
        """Build a RedTeamSummary from campaign config and results. Never raises."""
        try:
            return self._build_summary(config, results)
        except Exception:
            _log.warning("report_build_failed", campaign=config.campaign_id)
            return RedTeamSummary(
                campaign_id=config.campaign_id,
                agent_name=config.target.agent_name,
                total_attacks=len(results),
                successful_attacks=0,
                attack_success_rate=0.0,
            )

    def to_json(self, summary: RedTeamSummary, indent: int = 2) -> str:
        """Serialize summary to JSON string."""
        return summary.model_dump_json(indent=indent)

    def to_text(self, summary: RedTeamSummary) -> str:
        """Generate a human-readable text report."""
        lines = [
            "=== Nexus Red Team Report ===",
            f"Campaign:  {summary.campaign_id}",
            f"Agent:     {summary.agent_name}",
            f"Generated: {summary.generated_at.isoformat()}",
            "",
            "SUMMARY",
            f"  Total attacks:     {summary.total_attacks}",
            f"  Successful:        {summary.successful_attacks}",
            f"  Attack success:    {summary.attack_success_rate:.1%}",
            "",
            "SEVERITY BREAKDOWN",
        ]
        for sev in _SEVERITY_ORDER:
            count = summary.severity_counts.get(sev.value, 0)
            if count:
                lines.append(f"  {sev.value.upper():<10} {count}")
        lines += ["", "FINDINGS"]
        for f in sorted(summary.findings, key=lambda x: _SEVERITY_ORDER.index(x.severity)):
            lines += [
                f"  [{f.severity.value.upper()}] {f.title}",
                f"    Category:    {f.attack_category.value}",
                f"    OWASP:       {', '.join(c.value for c in f.owasp_categories)}",
                f"    Occurrences: {f.occurrence_count}",
                f"    {f.description}",
                f"    Recommendation: {f.recommendation}",
                "",
            ]
        return "\n".join(lines)

    def _build_summary(
        self,
        config: RedTeamCampaignConfig,
        results: list[AttackResult],
    ) -> RedTeamSummary:
        successful = [r for r in results if r.verdict.succeeded]
        asr = len(successful) / len(results) if results else 0.0
        findings = self._deduplicate(successful)
        sev_counts: dict[str, int] = {}
        owasp_seen: set[str] = set()
        for f in findings:
            sev_counts[f.severity.value] = sev_counts.get(f.severity.value, 0) + 1
            owasp_seen.update(c.value for c in f.owasp_categories)
        return RedTeamSummary(
            campaign_id=config.campaign_id,
            agent_name=config.target.agent_name,
            total_attacks=len(results),
            successful_attacks=len(successful),
            attack_success_rate=round(asr, 4),
            findings=findings,
            severity_counts=sev_counts,
            owasp_coverage=sorted(owasp_seen),
        )

    def _deduplicate(self, results: list[AttackResult]) -> list[RedTeamFinding]:
        """Group by (category, variant), keep worst severity per group."""
        groups: dict[tuple[Any, Any], list[AttackResult]] = {}
        for r in results:
            key = (r.payload.attack_category, r.payload.attack_variant)
            groups.setdefault(key, []).append(r)

        findings: list[RedTeamFinding] = []
        for (cat, variant), group in groups.items():
            worst = min(group, key=lambda r: _SEVERITY_ORDER.index(r.verdict.severity))
            findings.append(
                RedTeamFinding(
                    finding_id=str(uuid.uuid4()),
                    attack_category=cat,
                    severity=worst.verdict.severity,
                    title=f"{cat.value.replace('_', ' ').title()} — {variant.value.replace('_', ' ').title()}",
                    description=worst.verdict.reasoning[:300],
                    example_payload=worst.payload.content[:200],
                    example_response=worst.target_response[:300],
                    violated_properties=worst.verdict.violated_properties,
                    owasp_categories=worst.verdict.owasp_categories,
                    recommendation=_RECOMMENDATIONS.get(
                        cat, "Review agent configuration and safety pipeline."
                    ),
                    occurrence_count=len(group),
                )
            )
        return findings
