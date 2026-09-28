"""Events published in a transaction reach every matching subscription, signed, at least once."""

from __future__ import annotations

import json

import httpx
import pytest

from memory_service.adapters.models.credential_cipher import AesCredentialCipher
from memory_service.config.constants import WebhookTuning
from memory_service.config.settings import AgentCredentialSettings, WebhookSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.feedback import Feedback, FeedbackTargetKind, FeedbackVerdict
from memory_service.domain.webhooks import DeliveryStatus, Event, WebhookEvent
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.webhooks.service import WebhookService
from tests.integration.test_memory import _memories, _observe
from tests.unit.test_agent_credential_cipher import TEST_KEY
from trellis.memory.webhooks import verify_signature

pytestmark = pytest.mark.integration
ALICE = MemoryExecutionContext(tenant_id="acme", user_id="alice", workspace_id="fin")


class Receiver:
    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


def _install(container, receiver: Receiver, **tuning) -> WebhookService:
    service = WebhookService(
        container.services["uow_factory"],
        AesCredentialCipher(
            AgentCredentialSettings(active_key_id="test", encryption_keys={"test": TEST_KEY})
        ),
        WebhookSettings(allow_local_targets=True),
        tuning=WebhookTuning(**tuning),
        client_factory=receiver.client,
    )
    container.services["webhooks"] = service
    container.services["observation_pipeline"].events = service
    container.services["feedback"].events = service
    register_handlers(container)
    return service


async def _subscribe(
    service,
    uow_factory,
    *,
    tenant_id="acme",
    url="http://127.0.0.1:9/hook",
    events=None,
    workspace_id=None,
):
    async with uow_factory() as uow:
        subscription, secret = await service.subscribe(
            uow,
            tenant_id=tenant_id,
            url=url,
            events=events or list(WebhookEvent),
            created_by="svc:test",
            workspace_id=workspace_id,
        )
        await uow.commit()
    return subscription, secret


async def _deliveries(uow_factory, subscription):
    async with uow_factory() as uow:
        return await uow.webhook_deliveries.list(
            subscription.tenant_id, subscription.subscription_id
        )


async def test_memory_and_feedback_events_reach_matching_subscriptions_signed(
    container, uow_factory
) -> None:
    receiver = Receiver()
    service = _install(container, receiver)
    everything, secret = await _subscribe(service, uow_factory)
    ops_only, _ = await _subscribe(
        service, uow_factory, url="http://127.0.0.1:9/ops", workspace_id="ops"
    )
    foreign, _ = await _subscribe(
        service, uow_factory, tenant_id="globex", url="http://127.0.0.1:9/globex"
    )

    await _observe(
        container,
        uow_factory,
        ALICE,
        "My name is Amit and my timezone is Europe/Berlin. I prefer concise answers with code.",
    )
    await container.tasks.drain()  # webhooks.fanout -> webhooks.deliver
    memories = await _memories(uow_factory, ALICE, container)
    created = [r for r in receiver.requests if r.headers["X-Trellis-Event"] == "memory.created"]
    assert {json.loads(r.content)["data"]["memory_id"] for r in created} == {
        m.memory_id for m in memories
    }
    assert all(str(r.url) == "http://127.0.0.1:9/hook" for r in created)  # never /ops or /globex
    for request in created:
        assert verify_signature(secret, request.headers["X-Trellis-Signature"], request.content)
        body = json.loads(request.content)
        assert body["tenant_id"] == "acme" and body["workspace_id"] == "fin"
        assert body["event_id"].startswith("evt_") and body["data"]["scope_level"]
    assert await _deliveries(uow_factory, ops_only) == []
    assert await _deliveries(uow_factory, foreign) == []
    rows = await _deliveries(uow_factory, everything)
    assert rows and all(r.status is DeliveryStatus.DELIVERED and r.attempts == 1 for r in rows)

    async with uow_factory() as uow:
        await container.services["feedback"].submit(
            uow,
            ALICE,
            Feedback(
                tenant_id="acme",
                target_kind=FeedbackTargetKind.MEMORY,
                target_id=memories[0].memory_id,
                verdict=FeedbackVerdict.REJECT,
            ),
        )
        await uow.commit()
    for _ in range(3):
        await container.tasks.drain()
    kinds = [r.headers["X-Trellis-Event"] for r in receiver.requests]
    assert (
        "feedback.received" in kinds
        and "feedback.projected" in kinds
        and "memory.retracted" in kinds
    )
    projected = next(
        r for r in receiver.requests if r.headers["X-Trellis-Event"] == "feedback.projected"
    )
    assert json.loads(projected.content)["data"]["projection"]["action"] == "memory_retracted"
    assert "correction" not in json.loads(projected.content)["data"]


async def test_publishing_costs_nothing_without_subscriptions_and_dead_endpoints_are_disabled(
    container, uow_factory
) -> None:
    receiver = Receiver(status=503)
    service = _install(container, receiver, max_attempts=1, disable_after_failures=2)
    async with uow_factory() as uow:
        await service.publish(uow, Event(type=WebhookEvent.TEST, tenant_id="acme"))
        assert not uow._pending_outbox  # no subscription: no outbox row
        await uow.commit()
    subscription, _ = await _subscribe(service, uow_factory, events=[WebhookEvent.TEST])
    for _ in range(2):
        async with uow_factory() as uow:
            await service.publish(uow, Event(type=WebhookEvent.TEST, tenant_id="acme"))
            await uow.commit()
        await container.tasks.drain()
    rows = await _deliveries(uow_factory, subscription)
    assert len(rows) == 2 and all(
        r.status is DeliveryStatus.DEAD and r.status_code == 503 for r in rows
    )
    async with uow_factory() as uow:
        current = await uow.webhooks.get("acme", subscription.subscription_id)
    assert current is not None and current.failures == 2 and current.enabled is False
    # a disabled subscription is skipped by fan-out
    async with uow_factory() as uow:
        await service.publish(uow, Event(type=WebhookEvent.TEST, tenant_id="acme"))
        await uow.commit()
    await container.tasks.drain()
    assert len(await _deliveries(uow_factory, subscription)) == 2
    assert len(receiver.requests) == 2


async def test_fanout_only_reaches_subscriptions_that_want_the_event(
    container, uow_factory
) -> None:
    receiver = Receiver()
    service = _install(container, receiver)
    tests_only, _ = await _subscribe(service, uow_factory, events=[WebhookEvent.TEST])
    async with uow_factory() as uow:
        await service.publish(uow, Event(type=WebhookEvent.MEMORY_CREATED, tenant_id="acme"))
        await uow.commit()
    await container.tasks.drain()
    assert await _deliveries(uow_factory, tests_only) == [] and receiver.requests == []
    async with uow_factory() as uow:
        await service.publish(uow, Event(type=WebhookEvent.TEST, tenant_id="acme"))
        await uow.commit()
    await container.tasks.drain()
    assert len(await _deliveries(uow_factory, tests_only)) == 1 and len(receiver.requests) == 1


async def test_old_deliveries_are_purged(container, uow_factory) -> None:
    from datetime import UTC, datetime, timedelta

    receiver = Receiver()
    service = _install(container, receiver, delivery_retention_days=1)
    subscription, _ = await _subscribe(service, uow_factory, events=[WebhookEvent.TEST])
    async with uow_factory() as uow:
        await service.publish(uow, Event(type=WebhookEvent.TEST, tenant_id="acme"))
        await uow.commit()
    await container.tasks.drain()
    assert len(await _deliveries(uow_factory, subscription)) == 1
    assert await service.purge() == 0  # younger than the retention
    assert await service.purge(now=datetime.now(UTC) + timedelta(days=2)) == 1
    assert await _deliveries(uow_factory, subscription) == []
