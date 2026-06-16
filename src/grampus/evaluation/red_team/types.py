"""Red-team Pydantic types — no logic, only data models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AttackCategory(StrEnum):
    PROMPT_INJECTION = "prompt_injection"  # ASI01 direct/indirect
    JAILBREAK = "jailbreak"  # ASI01 roleplay/encoding/logic
    REASONING_HIJACK = "reasoning_hijack"  # ASI01 multi-turn trajectory
    MEMORY_POISON = "memory_poison"  # ASI06
    TOOL_MISUSE = "tool_misuse"  # ASI02
    EXCESSIVE_AGENCY = "excessive_agency"  # OWASP LLM 2025 #2


class Severity(StrEnum):
    CRITICAL = "critical"  # agent took clearly prohibited action
    HIGH = "high"  # security property violated, recoverable
    MEDIUM = "medium"  # partial bypass, degraded safety
    LOW = "low"  # minor deviation, unlikely exploitable
    INFO = "info"  # observation, no clear violation


class OWASPCategory(StrEnum):
    ASI01_GOAL_HIJACK = "ASI01:2026"
    ASI02_TOOL_MISUSE = "ASI02:2026"
    ASI03_IDENTITY_ABUSE = "ASI03:2026"
    ASI04_SUPPLY_CHAIN = "ASI04:2026"
    ASI05_CODE_EXEC = "ASI05:2026"
    ASI06_MEMORY_POISON = "ASI06:2026"
    ASI07_INTER_AGENT = "ASI07:2026"
    ASI08_CASCADE = "ASI08:2026"
    ASI09_TRUST_EXPLOIT = "ASI09:2026"
    ASI10_ROGUE = "ASI10:2026"


class SecurityProperty(StrEnum):
    """The four properties an agent must not violate (arXiv 2603.19469)."""

    TASK_ALIGNMENT = "task_alignment"
    ACTION_ALIGNMENT = "action_alignment"
    SOURCE_AUTHORIZATION = "source_authorization"
    DATA_ISOLATION = "data_isolation"


class AttackVariant(StrEnum):
    """Sub-type within a strategy for reporting granularity."""

    DIRECT_INJECTION = "direct_injection"
    INDIRECT_INJECTION = "indirect_injection"
    ROLEPLAY = "roleplay"
    ENCODING_TRICK = "encoding_trick"
    LOGIC_TRAP = "logic_trap"
    MULTI_TURN_DRIFT = "multi_turn_drift"
    MEMORY_WRITE_INJECT = "memory_write_inject"
    MEMORY_READ_POISON = "memory_read_poison"
    TOOL_LOOP = "tool_loop"
    TOOL_CHAIN_ESCAPE = "tool_chain_escape"
    SCOPE_ESCALATION = "scope_escalation"
    IMPLICIT_PERMISSION = "implicit_permission"


class AttackPayload(BaseModel):
    """A single adversarial input to present to the target agent."""

    content: str
    attack_category: AttackCategory
    attack_variant: AttackVariant
    strategy_name: str
    turn: int = Field(default=1, ge=1)
    prior_turns: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeVerdict(BaseModel):
    """RedTeamJudge's evaluation of one attack result."""

    succeeded: bool
    severity: Severity
    violated_properties: list[SecurityProperty] = Field(default_factory=list)
    owasp_categories: list[OWASPCategory] = Field(default_factory=list)
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


class AttackResult(BaseModel):
    """Full record of one attack attempt."""

    result_id: str
    payload: AttackPayload
    target_response: str
    verdict: JudgeVerdict
    duration_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RedTeamTargetConfig(BaseModel):
    """Describes the agent being red-teamed."""

    agent_name: str
    system_prompt: str
    available_tools: list[str] = Field(default_factory=list)
    memory_enabled: bool = False
    crew_enabled: bool = False
    max_turns: int = Field(default=3, ge=1, le=10)


class RedTeamCampaignConfig(BaseModel):
    """Configuration for a full red-team campaign."""

    campaign_id: str
    target: RedTeamTargetConfig
    enabled_categories: list[AttackCategory] = Field(default_factory=lambda: list(AttackCategory))
    payloads_per_strategy: int = Field(default=5, ge=1, le=50)
    max_concurrent: int = Field(default=3, ge=1, le=10)
    stop_on_critical: bool = False
