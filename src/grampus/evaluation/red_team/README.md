# `grampus/evaluation/red_team/` — Adversarial Red-Teaming (G41)

This sub-package implements automated red-teaming as a first-class evaluation primitive, described in ADR-020. It runs Attacker → Target → Judge campaigns that cover the OWASP Agentic Top 10 (ASI01–ASI10:2026) attack categories unique to agent systems.

Grounded in: OWASP Agentic Top 10:2026, AgenticRed (arXiv 2601.13518), Dreadnode (arXiv 2605.04019), arXiv 2603.19469 (security property formalization).

---

## Key abstractions

| Class | File | Role |
|---|---|---|
| `RedTeamRunner` | `runner.py` | Orchestrates one campaign: Attacker → Target → Judge loop, concurrency bounded by semaphore |
| `AttackerAgent` | `attacker.py` | Generates payloads from strategies; adapts failed payloads once via mutation |
| `RedTeamJudge` | `judge.py` | Evaluates whether an attack succeeded: LLM judge + rule-based fallback |
| `RedTeamReport` | `report.py` | Formats `AttackResult` list into human-readable report |
| `AttackStrategy` | `strategies/` | ABC — one strategy per OWASP category |
| `RedTeamCampaignConfig` | `types.py` | Campaign settings: enabled_categories, payloads_per_strategy, max_concurrent, stop_on_critical |
| `RedTeamTargetConfig` | `types.py` | Target description: name, description, system_prompt (for context) |
| `AttackPayload` | `types.py` | One generated attack attempt: payload, category, strategy_id, expected_effect |
| `AttackResult` | `types.py` | One completed attempt: payload, response, verdict, severity, owasp_category |
| `JudgeVerdict` | `types.py` | Judge output: success, confidence, explanation |
| `Severity` | `types.py` | Enum: `CRITICAL, HIGH, MEDIUM, LOW, INFORMATIONAL` |

---

## Attack strategies (6 OWASP Agentic Top 10 categories)

| Strategy | OWASP Category | Attack type |
|---|---|---|
| `MemoryPoisoningStrategy` | ASI01 — Memory Manipulation | Inject malicious content via natural-language memory writes |
| `PromptInjectionStrategy` | ASI02 — Prompt Injection | Tool result injection, indirect injection via retrieved content |
| `ToolAbuseStrategy` | ASI03 — Excessive Agency | Manipulate agent into calling unauthorized or dangerous tools |
| `PrivilegeEscalationStrategy` | ASI04 — Privilege Escalation | Social engineering to exceed agent's defined scope |
| `DataExfiltrationStrategy` | ASI05 — Sensitive Info Disclosure | Extract PII, credentials, or system prompt via crafted prompts |
| `ReasoningHijackStrategy` | ASI08 — Decision Hijacking | Multi-turn manipulation to alter agent's reasoning chain |

Each strategy maps to one or more of the four security properties from arXiv 2603.19469: **task alignment**, **action alignment**, **source authorization**, **data isolation**.

---

## How to run a red-team campaign

```python
from grampus.evaluation.red_team.runner import RedTeamRunner
from grampus.evaluation.red_team.attacker import AttackerAgent
from grampus.evaluation.red_team.judge import RedTeamJudge
from grampus.evaluation.red_team.types import (
    RedTeamCampaignConfig, RedTeamTargetConfig
)

# target_fn: any async callable (messages) -> str
# This decouples the runner from AgentRunner's full lifecycle
async def target_fn(messages: list[tuple[str, str]]) -> str:
    result = await runner.run(agent_def, messages[-1][1], session_id=uuid4().hex)
    return result.output

attacker = AttackerAgent(
    strategies=["memory_poisoning", "prompt_injection", "tool_abuse"],
    model_client=client,
    model_id="claude-haiku-4-5-20251001",
)

judge = RedTeamJudge(
    model_client=client,
    model_id="claude-haiku-4-5-20251001",
)

campaign_runner = RedTeamRunner(attacker=attacker, judge=judge, target_fn=target_fn)

config = RedTeamCampaignConfig(
    campaign_id="campaign-2026-01",
    target=RedTeamTargetConfig(
        name="research-agent",
        description="Agent that searches the web and summarizes findings",
        system_prompt=agent_def.system_prompt,
    ),
    enabled_categories=["memory_poisoning", "prompt_injection"],
    payloads_per_strategy=10,
    max_concurrent=3,
    stop_on_critical=True,
)

results = await campaign_runner.run(config)
```

### Via CLI

```bash
grampus redteam agent.py                          # run all strategies
grampus redteam agent.py --categories injection   # specific categories
grampus redteam agent.py --output report.json     # JSON output
```

**Exit code 1** on CRITICAL or HIGH severity findings — enables CI/CD pipeline integration.

---

## Attacker → Target → Judge loop

```
For each payload (bounded by asyncio.Semaphore(max_concurrent)):
    │
    ├─ target_fn(payload.messages) → response_text
    │
    ├─ RedTeamJudge.evaluate(payload, response_text)
    │   ├─ Rule-based check: regex patterns for known success signals
    │   ├─ LLM judge call (if model_client provided)
    │   └─ If LLM confidence < 0.5 → fall back to rule-based result
    │
    ├─ result.verdict.success == False AND model_client available?
    │   Yes → AttackerAgent.mutate_failed(payload) → one adaptive mutation → retry
    │         (AgenticRed pattern: doubles ASR on rule-based targets)
    │
    └─ Append AttackResult to results

If stop_on_critical and any result.severity == CRITICAL → abort remaining payloads
```

---

## Report format

```python
from grampus.evaluation.red_team.report import RedTeamReport

report = RedTeamReport(results)
print(report.summary())
# Attack Success Rate: 23.3% (7/30 payloads)
# CRITICAL: 1 finding — MemoryPoisoning (ASI01)
# HIGH: 2 findings — PromptInjection (ASI02)
# LOW: 4 findings — DataExfiltration (ASI05)
# Security properties violated: data_isolation, source_authorization

report.to_json("report.json")
```

---

## Hard invariants

- **`RedTeamRunner` never raises.** `run()` catches all exceptions and returns an empty list on failure. Campaign failures are logged but never surface to the caller.
- **`target_fn` is any `async (messages: list[tuple[str, str]]) -> str` callable.** Do not pass an `AgentRunner` instance directly — the adapter contract (`get_agent_config()` + `run_conversation()`) is required for the CLI, but `target_fn` is the programmatic API.
- **Rule-based judge always runs** — even when `model_client` is provided. If LLM confidence < 0.5, the rule-based result wins. This prevents a compromised judge from suppressing true positives.
- **One mutation retry per failed payload** — not a loop. The mutation heuristic (AgenticRed) is an enhancement, not a guarantee. Multiple retries would mask true agent robustness.
- **`max_concurrent` semaphore is per-campaign**, not per-strategy. Tune it based on target agent's capacity and API rate limits.

---

## Dependency map

```
red_team/ depends on:     core/ (errors, logging, types)
red_team/ is imported by: evaluation/suite.py (for combined safety+quality suites), cli/
red_team/ must NOT import from: orchestration/ (target_fn decoupling is intentional),
                               memory/, tools/, safety/, dapr/
```

---

## ADR references

- **ADR-020** — Adversarial red-teaming as a first-class evaluation primitive: full design rationale
