"""Tests for nexus.dapr.pubsub — DaprPubSub typed publish/subscribe."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from nexus.dapr.pubsub import DaprPubSub


class OrderEvent(BaseModel):
    order_id: str
    amount: float


class PaymentEvent(BaseModel):
    payment_id: str
    status: str


@pytest.fixture()
def mock_gw() -> AsyncMock:
    gw = AsyncMock()
    gw.publish_event = AsyncMock(return_value=None)
    return gw


@pytest.fixture()
def pubsub(mock_gw: AsyncMock) -> DaprPubSub:
    return DaprPubSub(gateway=mock_gw, pubsub_name="test-pubsub")


class TestDaprPubSubPublish:
    async def test_publish_calls_gateway_publish_event(
        self, pubsub: DaprPubSub, mock_gw: AsyncMock
    ) -> None:
        event = OrderEvent(order_id="ord-1", amount=99.9)
        await pubsub.publish("orders", event)
        mock_gw.publish_event.assert_called_once()

    async def test_publish_uses_pubsub_name(self, pubsub: DaprPubSub, mock_gw: AsyncMock) -> None:
        event = OrderEvent(order_id="ord-1", amount=99.9)
        await pubsub.publish("orders", event)
        call_kwargs = mock_gw.publish_event.call_args
        combined = str(call_kwargs)
        assert "test-pubsub" in combined

    async def test_publish_uses_topic_name(self, pubsub: DaprPubSub, mock_gw: AsyncMock) -> None:
        event = OrderEvent(order_id="ord-1", amount=99.9)
        await pubsub.publish("orders", event)
        call_kwargs = mock_gw.publish_event.call_args
        combined = str(call_kwargs)
        assert "orders" in combined

    async def test_publish_serializes_model_as_json(
        self, pubsub: DaprPubSub, mock_gw: AsyncMock
    ) -> None:
        event = OrderEvent(order_id="ord-42", amount=5.5)
        await pubsub.publish("orders", event)
        call_kwargs = mock_gw.publish_event.call_args
        args, kwargs = call_kwargs
        data_bytes = kwargs.get("data") or args[2]
        parsed = json.loads(data_bytes)
        assert parsed["order_id"] == "ord-42"
        assert parsed["amount"] == 5.5

    async def test_publish_returns_none(self, pubsub: DaprPubSub) -> None:
        event = OrderEvent(order_id="x", amount=0.0)
        result = await pubsub.publish("topic", event)
        assert result is None

    async def test_publish_with_metadata(self, pubsub: DaprPubSub, mock_gw: AsyncMock) -> None:
        event = OrderEvent(order_id="x", amount=0.0)
        await pubsub.publish("topic", event, metadata={"ttl": "60"})
        call_kwargs = mock_gw.publish_event.call_args
        combined = str(call_kwargs)
        assert "ttl" in combined


class TestDaprPubSubHandlerRegistry:
    def test_register_handler_stores_callback(self, pubsub: DaprPubSub) -> None:
        async def handler(event: OrderEvent) -> None:
            pass

        pubsub.register_handler("orders", OrderEvent, handler)
        assert pubsub.has_handler("orders")

    def test_no_handler_registered_initially(self, pubsub: DaprPubSub) -> None:
        assert not pubsub.has_handler("unregistered-topic")

    def test_register_multiple_topics(self, pubsub: DaprPubSub) -> None:
        async def h1(event: OrderEvent) -> None:
            pass

        async def h2(event: PaymentEvent) -> None:
            pass

        pubsub.register_handler("orders", OrderEvent, h1)
        pubsub.register_handler("payments", PaymentEvent, h2)
        assert pubsub.has_handler("orders")
        assert pubsub.has_handler("payments")

    def test_decorator_registers_handler(self, pubsub: DaprPubSub) -> None:
        @pubsub.handler("orders", OrderEvent)
        async def handle_order(event: OrderEvent) -> None:
            pass

        assert pubsub.has_handler("orders")

    def test_decorator_returns_original_function(self, pubsub: DaprPubSub) -> None:
        async def handle_order(event: OrderEvent) -> None:
            pass

        decorated = pubsub.handler("orders", OrderEvent)(handle_order)
        assert decorated is handle_order


class TestDaprPubSubHandleIncoming:
    async def test_handle_incoming_calls_registered_handler(self, pubsub: DaprPubSub) -> None:
        received: list[OrderEvent] = []

        async def handler(event: OrderEvent) -> None:
            received.append(event)

        pubsub.register_handler("orders", OrderEvent, handler)
        payload = json.dumps({"order_id": "x", "amount": 1.0}).encode()
        await pubsub.handle_incoming("orders", payload)
        assert len(received) == 1
        assert received[0].order_id == "x"

    async def test_handle_incoming_deserializes_event(self, pubsub: DaprPubSub) -> None:
        received: list[OrderEvent] = []

        async def handler(event: OrderEvent) -> None:
            received.append(event)

        pubsub.register_handler("orders", OrderEvent, handler)
        payload = json.dumps({"order_id": "ord-99", "amount": 50.0}).encode()
        await pubsub.handle_incoming("orders", payload)
        assert received[0].order_id == "ord-99"
        assert received[0].amount == 50.0

    async def test_handle_incoming_unknown_topic_is_noop(self, pubsub: DaprPubSub) -> None:
        payload = json.dumps({"order_id": "x", "amount": 1.0}).encode()
        await pubsub.handle_incoming("unknown-topic", payload)

    async def test_handle_incoming_multiple_handlers_called(self, pubsub: DaprPubSub) -> None:
        calls: list[str] = []

        async def h1(event: OrderEvent) -> None:
            calls.append("h1")

        async def h2(event: OrderEvent) -> None:
            calls.append("h2")

        pubsub.register_handler("orders", OrderEvent, h1)
        pubsub.register_handler("orders", OrderEvent, h2)
        payload = json.dumps({"order_id": "x", "amount": 1.0}).encode()
        await pubsub.handle_incoming("orders", payload)
        assert "h1" in calls
        assert "h2" in calls
