"""Tests for A2A AgentRegistry."""

from __future__ import annotations

from unittest.mock import MagicMock

from a2a.types.a2a_pb2 import AgentCard, AgentSkill

from grampus.core.types import AgentDefinition


def _make_runner() -> MagicMock:
    runner = MagicMock()
    return runner


def _make_agent_def(name: str = "my-agent") -> AgentDefinition:
    return AgentDefinition(name=name, model="claude-3-5-haiku-20241022")


def test_register_local_creates_agent_card() -> None:
    from grampus.orchestration.a2a.registry import AgentRegistry

    registry = AgentRegistry()
    runner = _make_runner()
    registry.register_local(
        name="agent1",
        runner=runner,
        description="A helpful agent",
        base_url="http://localhost:8000",
    )

    entry = registry.get("agent1")
    assert entry is not None
    assert entry.runner is runner
    assert isinstance(entry.card, AgentCard)
    assert entry.card.name == "agent1"


def test_register_remote_stores_url() -> None:
    from grampus.orchestration.a2a.registry import AgentRegistry

    registry = AgentRegistry()
    registry.register_remote(name="remote-agent", url="http://remote.test")

    entry = registry.get("remote-agent")
    assert entry is not None
    assert entry.remote_url == "http://remote.test"
    assert entry.runner is None


def test_get_returns_entry() -> None:
    from grampus.orchestration.a2a.registry import AgentRegistry

    registry = AgentRegistry()
    registry.register_local(name="agent2", runner=_make_runner(), description="Agent 2")

    entry = registry.get("agent2")
    assert entry is not None
    assert entry.name == "agent2"


def test_get_missing_returns_none() -> None:
    from grampus.orchestration.a2a.registry import AgentRegistry

    registry = AgentRegistry()
    assert registry.get("does-not-exist") is None


def test_list_agents_returns_cards() -> None:
    from grampus.orchestration.a2a.registry import AgentRegistry

    registry = AgentRegistry()
    registry.register_local(name="a", runner=_make_runner(), description="A")
    registry.register_local(name="b", runner=_make_runner(), description="B")

    cards = registry.list_agents()
    names = {c.name for c in cards}
    assert "a" in names
    assert "b" in names


def test_generate_server_card_has_streaming_capability() -> None:
    from grampus.orchestration.a2a.registry import AgentRegistry

    registry = AgentRegistry()
    card = registry.generate_server_card(
        name="grampus-server",
        description="Main server",
        base_url="http://localhost:8000",
    )

    assert isinstance(card, AgentCard)
    assert card.capabilities.streaming is True


def test_generate_server_card_api_key_scheme_when_requested() -> None:
    from grampus.orchestration.a2a.registry import AgentRegistry

    registry = AgentRegistry()
    card = registry.generate_server_card(
        name="secure-server",
        description="Secure server",
        base_url="http://localhost:8000",
        api_key_scheme=True,
    )

    assert len(card.security_schemes) > 0


def test_generate_server_card_url_points_to_a2a_endpoint() -> None:
    from grampus.orchestration.a2a.registry import AgentRegistry

    registry = AgentRegistry()
    card = registry.generate_server_card(
        name="my-server",
        description="My server",
        base_url="http://example.com",
    )

    urls = [iface.url for iface in card.supported_interfaces]
    assert any("/a2a" in url for url in urls)


def test_register_local_with_skills() -> None:
    from grampus.orchestration.a2a.registry import AgentRegistry

    registry = AgentRegistry()
    skill = AgentSkill()
    skill.id = "summarize"
    skill.name = "Summarize"
    skill.description = "Summarizes documents"

    registry.register_local(
        name="agent-with-skills",
        runner=_make_runner(),
        description="A skilled agent",
        skills=[skill],
    )

    entry = registry.get("agent-with-skills")
    assert entry is not None
    assert len(entry.card.skills) == 1
    assert entry.card.skills[0].id == "summarize"


# ---------------------------------------------------------------------------
# Dapr service registration tests
# ---------------------------------------------------------------------------


def test_register_dapr_service_creates_entry() -> None:
    from grampus.orchestration.a2a.registry import AgentRegistry

    registry = AgentRegistry()
    registry.register_dapr_service(
        name="sibling-agent",
        dapr_app_id="grampus-worker",
        description="Worker service",
    )

    entry = registry.get("sibling-agent")
    assert entry is not None
    assert isinstance(entry.card, AgentCard)


def test_register_dapr_service_sets_app_id() -> None:
    from grampus.orchestration.a2a.registry import AgentRegistry

    registry = AgentRegistry()
    registry.register_dapr_service(
        name="worker",
        dapr_app_id="grampus-worker-svc",
        description="Worker",
    )

    entry = registry.get("worker")
    assert entry is not None
    assert entry.dapr_app_id == "grampus-worker-svc"
    assert entry.runner is None
    assert entry.remote_url is None
    assert entry.client is None


def test_register_dapr_service_default_method_is_a2a() -> None:
    from grampus.orchestration.a2a.registry import AgentRegistry

    registry = AgentRegistry()
    registry.register_dapr_service(name="svc", dapr_app_id="my-app", description="svc")

    entry = registry.get("svc")
    assert entry is not None
    assert entry.dapr_method == "a2a"


def test_register_dapr_service_custom_method() -> None:
    from grampus.orchestration.a2a.registry import AgentRegistry

    registry = AgentRegistry()
    registry.register_dapr_service(
        name="svc2",
        dapr_app_id="my-app",
        description="svc2",
        dapr_method="invoke/a2a",
    )

    entry = registry.get("svc2")
    assert entry is not None
    assert entry.dapr_method == "invoke/a2a"


def test_register_dapr_service_entry_has_no_runner_no_remote_url() -> None:
    from grampus.orchestration.a2a.registry import AgentRegistry

    registry = AgentRegistry()
    registry.register_dapr_service(
        name="pure-dapr",
        dapr_app_id="pure-dapr-app",
        description="pure dapr",
    )

    entry = registry.get("pure-dapr")
    assert entry is not None
    assert entry.runner is None
    assert entry.remote_url is None
    assert entry.client is None
