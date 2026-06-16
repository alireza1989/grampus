"""Dapr pub/sub integration tests — require a live Dapr sidecar."""

from __future__ import annotations

import pytest

from tests.integration.conftest import skip_if_no_dapr


@skip_if_no_dapr
@pytest.mark.dapr
class TestDaprPubSub:
    async def test_publish_does_not_raise(self) -> None:
        """Basic smoke test: publish to a topic without errors."""
        import os

        from grampus.dapr.client import DaprGateway
        from grampus.dapr.pubsub import DaprPubSub

        port = int(os.environ.get("DAPR_HTTP_PORT", "3500"))
        gw = DaprGateway(http_port=port)
        pubsub = DaprPubSub(gw, pubsub_name="pubsub")
        await pubsub.publish("grampus.test.topic", {"msg": "hello"})

    async def test_publish_and_subscribe_round_trip(self) -> None:
        """Publish a message and verify it is received within 2 seconds."""
        import asyncio
        import os

        from grampus.dapr.client import DaprGateway
        from grampus.dapr.pubsub import DaprPubSub

        port = int(os.environ.get("DAPR_HTTP_PORT", "3500"))
        gw = DaprGateway(http_port=port)
        pubsub = DaprPubSub(gw, pubsub_name="pubsub")

        received: list[dict] = []

        @pubsub.subscribe("grampus.test.roundtrip")
        async def handler(event: dict) -> None:
            received.append(event)

        await pubsub.publish("grampus.test.roundtrip", {"value": 42})
        await asyncio.sleep(0.5)
        assert any(e.get("value") == 42 for e in received), "Message not received within timeout"
