"""Tests for ModelRouter — selection, client lookup, and fallback logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.core.errors import ModelError
from nexus.core.models.base import ModelResponse
from nexus.core.types import Message, Role, TokenUsage
from nexus.orchestration.model_router import ModelRouter, ModelSpec, ModelTier, RoutingRule

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _usage() -> TokenUsage:
    return TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.001, model="m")


def _response(model_id: str = "fast-model") -> ModelResponse:
    return ModelResponse(
        content="hi",
        tool_calls=[],
        token_usage=_usage(),
        model=model_id,
        stop_reason="end_turn",
    )


def _spec(
    model_id: str,
    tier: ModelTier,
    input_cost: float,
    *,
    supports_tools: bool = True,
) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        tier=tier,
        provider="anthropic",
        input_cost_per_1k_tokens=input_cost,
        output_cost_per_1k_tokens=input_cost * 3,
        context_window=200_000,
        supports_tools=supports_tools,
    )


def _client(model_id: str, *, fail: bool = False) -> MagicMock:
    client = MagicMock()
    if fail:
        client.complete = AsyncMock(side_effect=ModelError("fail", code="MODEL_FAIL"))
    else:
        client.complete = AsyncMock(return_value=_response(model_id))
    return client


def _messages() -> list[Message]:
    return [Message(role=Role.USER, content="hello")]


# default model set used in most tests
FAST_CHEAP = _spec("fast-cheap", ModelTier.FAST, 0.0001)
FAST_EXP = _spec("fast-expensive", ModelTier.FAST, 0.001)
BALANCED = _spec("balanced-model", ModelTier.BALANCED, 0.003)
POWERFUL = _spec("powerful-model", ModelTier.POWERFUL, 0.015)
NO_TOOLS_FAST = _spec("no-tools-fast", ModelTier.FAST, 0.0002, supports_tools=False)


def _default_router(
    *,
    rules: list[RoutingRule] | None = None,
    fallback_chain: list[ModelTier] | None = None,
) -> ModelRouter:
    models = [FAST_CHEAP, FAST_EXP, BALANCED, POWERFUL, NO_TOOLS_FAST]
    clients = {
        "fast-cheap": _client("fast-cheap"),
        "fast-expensive": _client("fast-expensive"),
        "balanced-model": _client("balanced-model"),
        "powerful-model": _client("powerful-model"),
        "no-tools-fast": _client("no-tools-fast"),
    }
    return ModelRouter(models, clients, routing_rules=rules, fallback_chain=fallback_chain)


# ---------------------------------------------------------------------------
# TestModelRouterSelection
# ---------------------------------------------------------------------------


class TestModelRouterSelection:
    def test_select_returns_cheapest_model_in_tier(self) -> None:
        router = _default_router()
        # Both FAST_CHEAP (0.0001) and FAST_EXP (0.001) are in FAST tier
        spec = router.select("any_step")
        # default tier = BALANCED when no rule matches
        assert spec.model_id == "balanced-model"

    def test_select_applies_routing_rule_by_step_name_substring(self) -> None:
        rules = [RoutingRule(step_pattern="summarize", preferred_tier=ModelTier.FAST)]
        router = _default_router(rules=rules)
        spec = router.select("long_summarize_step")
        # cheapest FAST is fast-cheap (0.0001)
        assert spec.model_id == "fast-cheap"

    def test_select_defaults_to_balanced_when_no_rule_matches(self) -> None:
        rules = [RoutingRule(step_pattern="xyz_nonexistent", preferred_tier=ModelTier.FAST)]
        router = _default_router(rules=rules)
        spec = router.select("plan_step")
        assert spec.tier == ModelTier.BALANCED

    def test_select_filters_by_supports_tools_when_required(self) -> None:
        # In FAST tier: fast-cheap and fast-expensive support tools; no-tools-fast does not
        rules = [RoutingRule(step_pattern="tool_step", preferred_tier=ModelTier.FAST)]
        router = _default_router(rules=rules)
        spec = router.select("tool_step", require_tools=True)
        assert spec.supports_tools is True
        assert spec.model_id == "fast-cheap"

    def test_select_raises_model_error_when_no_model_in_tier(self) -> None:
        # Only POWERFUL registered, ask for FAST via rule
        models = [POWERFUL]
        clients = {"powerful-model": _client("powerful-model")}
        rules = [RoutingRule(step_pattern="fast_step", preferred_tier=ModelTier.FAST)]
        router = ModelRouter(models, clients, routing_rules=rules)
        with pytest.raises(ModelError) as exc_info:
            router.select("fast_step")
        assert exc_info.value.code == "NO_MODEL_AVAILABLE"

    def test_routing_rules_first_match_wins(self) -> None:
        rules = [
            RoutingRule(step_pattern="step", preferred_tier=ModelTier.POWERFUL),
            RoutingRule(step_pattern="analysis_step", preferred_tier=ModelTier.FAST),
        ]
        router = _default_router(rules=rules)
        # "analysis_step" matches first rule ("step" is substring) → POWERFUL wins
        spec = router.select("analysis_step")
        assert spec.tier == ModelTier.POWERFUL

    def test_select_case_insensitive_step_name_match(self) -> None:
        rules = [RoutingRule(step_pattern="SUMMARIZE", preferred_tier=ModelTier.FAST)]
        router = _default_router(rules=rules)
        spec = router.select("long_summarize_step")
        assert spec.tier == ModelTier.FAST


# ---------------------------------------------------------------------------
# TestModelRouterClient
# ---------------------------------------------------------------------------


class TestModelRouterClient:
    def test_get_client_returns_registered_client(self) -> None:
        router = _default_router()
        client = router.get_client(BALANCED)
        assert client is not None

    def test_get_client_raises_model_error_when_not_registered(self) -> None:
        router = _default_router()
        unknown_spec = _spec("unknown-model", ModelTier.POWERFUL, 0.02)
        with pytest.raises(ModelError) as exc_info:
            router.get_client(unknown_spec)
        assert exc_info.value.code == "NO_CLIENT_FOR_MODEL"


# ---------------------------------------------------------------------------
# TestModelRouterFallback
# ---------------------------------------------------------------------------


class TestModelRouterFallback:
    @pytest.mark.asyncio
    async def test_complete_with_fallback_returns_response_and_spec(self) -> None:
        router = _default_router()
        response, spec = await router.complete_with_fallback("my_step", _messages())
        assert response.content == "hi"
        assert spec.tier == ModelTier.BALANCED

    @pytest.mark.asyncio
    async def test_complete_with_fallback_tries_next_tier_on_model_error(self) -> None:
        # Make balanced fail, fast succeed
        models = [FAST_CHEAP, BALANCED]
        fast_client = _client("fast-cheap")
        balanced_client = _client("balanced-model", fail=True)
        clients = {"fast-cheap": fast_client, "balanced-model": balanced_client}
        # fallback_chain: balanced → fast
        router = ModelRouter(
            models,
            clients,
            fallback_chain=[ModelTier.BALANCED, ModelTier.FAST],
        )
        response, spec = await router.complete_with_fallback("step", _messages())
        assert spec.model_id == "fast-cheap"
        assert response.model == "fast-cheap"

    @pytest.mark.asyncio
    async def test_complete_with_fallback_raises_when_all_tiers_fail(self) -> None:
        models = [FAST_CHEAP, BALANCED]
        clients = {
            "fast-cheap": _client("fast-cheap", fail=True),
            "balanced-model": _client("balanced-model", fail=True),
        }
        router = ModelRouter(
            models,
            clients,
            fallback_chain=[ModelTier.BALANCED, ModelTier.FAST],
        )
        with pytest.raises(ModelError):
            await router.complete_with_fallback("step", _messages())

    @pytest.mark.asyncio
    async def test_complete_with_fallback_skips_tier_with_no_registered_model(self) -> None:
        # Only FAST registered; fallback_chain starts with BALANCED (missing) then FAST
        models = [FAST_CHEAP]
        clients = {"fast-cheap": _client("fast-cheap")}
        router = ModelRouter(
            models,
            clients,
            fallback_chain=[ModelTier.BALANCED, ModelTier.FAST],
        )
        response, spec = await router.complete_with_fallback("step", _messages())
        assert spec.model_id == "fast-cheap"

    @pytest.mark.asyncio
    async def test_complete_with_fallback_returns_spec_of_model_actually_used(self) -> None:
        # Primary = BALANCED fails, FAST succeeds
        models = [FAST_CHEAP, BALANCED, POWERFUL]
        clients = {
            "fast-cheap": _client("fast-cheap"),
            "balanced-model": _client("balanced-model", fail=True),
            "powerful-model": _client("powerful-model"),
        }
        router = ModelRouter(
            models,
            clients,
            fallback_chain=[ModelTier.BALANCED, ModelTier.FAST, ModelTier.POWERFUL],
        )
        _, spec = await router.complete_with_fallback("step", _messages())
        assert spec.model_id == "fast-cheap"
