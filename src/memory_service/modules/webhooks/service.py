"""Subscriptions, fan-out and signed delivery.

``publish`` is called inside the transaction that made the event true: it writes one
``webhooks.fanout`` job to the outbox when the tenant has any enabled subscription, so
a request pays one indexed existence check and nothing else. Fan-out (a job) writes one
delivery row and one ``webhooks.deliver`` job per matching subscription; delivery (a job
with retries) signs and POSTs, records the outcome, and disables a subscription whose
endpoint keeps failing.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Final
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import SecretStr

from memory_service.config.constants import HEADERS, WEBHOOK_HEADERS, WEBHOOKS, WebhookTuning
from memory_service.config.settings import WebhookSettings
from memory_service.domain.errors import (
    Conflict,
    DependencyUnavailable,
    NotFound,
    ValidationFailed,
)
from memory_service.domain.ids import is_valid_id, new_id
from memory_service.domain.webhooks import (
    ERROR_MAX_CHARS,
    DeliveryStatus,
    Event,
    WebhookDelivery,
    WebhookEvent,
    WebhookSubscription,
)
from memory_service.modules.webhooks.signing import sign
from memory_service.modules.webhooks.targets import (
    TargetPolicy,
    TargetRefused,
    resolved_addresses,
    validate_url,
)
from memory_service.observability.logging import get_logger
from memory_service.observability.tracing import (
    current_span_id,
    current_trace_flags,
    format_traceparent,
    new_span_id,
    new_trace_id,
)
from memory_service.ports.credentials import CredentialCipher, ModelIdentity, StoredCredential
from memory_service.ports.tasks import JobSpec, Queue
from memory_service.ports.uow import UnitOfWork, UnitOfWorkFactory

log = get_logger(__name__)

TASK_WEBHOOK_FANOUT = "webhooks.fanout"
TASK_WEBHOOK_DELIVER = "webhooks.deliver"
TASK_WEBHOOK_PURGE = "webhooks.purge_deliveries"
USER_AGENT: Final = "trellis-memory-webhooks"
SECRET_BYTES: Final = 32
#: The principal a subscription's signing key is bound to in the credential envelope.
ENVELOPE_PRINCIPAL_PREFIX: Final = "webhook:"
#: Terminal delivery states: a job run that finds one has nothing left to do.
SETTLED: Final = frozenset({DeliveryStatus.DELIVERED, DeliveryStatus.DEAD})


class DeliveryError(Exception):
    """One attempt failed. The queue retries it unless ``permanent`` (a target that now
    resolves to a private address, a secret the envelope can no longer open)."""

    def __init__(self, status_code: int | None, error: str, *, permanent: bool = False) -> None:
        super().__init__(error)
        self.status_code = status_code
        self.error = error[:ERROR_MAX_CHARS]
        self.permanent = permanent


def _now() -> datetime:
    return datetime.now(UTC)


def _secret_identity(subscription: WebhookSubscription) -> ModelIdentity:
    return ModelIdentity(
        subscription.tenant_id, f"{ENVELOPE_PRINCIPAL_PREFIX}{subscription.subscription_id}"
    )


class WebhookService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        cipher: CredentialCipher,
        settings: WebhookSettings,
        *,
        tuning: WebhookTuning = WEBHOOKS,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.cipher = cipher
        self.tuning = tuning
        self.policy = TargetPolicy(allow_local_targets=settings.allow_local_targets)
        self._client_factory = client_factory or self._default_client

    def _default_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self.tuning.timeout_seconds,
            follow_redirects=False,
            headers={"User-Agent": USER_AGENT},
        )

    # ------------------------------------------------------------------ subscriptions
    async def subscribe(
        self,
        uow: UnitOfWork,
        *,
        tenant_id: str,
        url: str,
        events: list[WebhookEvent],
        created_by: str,
        workspace_id: str | None = None,
        description: str | None = None,
        subscription_id: str | None = None,
    ) -> tuple[WebhookSubscription, str]:
        """Create a subscription; the returned secret is shown once and never stored."""
        if subscription_id is not None and not is_valid_id(subscription_id):
            raise ValidationFailed(f"invalid subscription_id: {subscription_id!r}")
        target = validate_url(url, self.policy)
        await resolved_addresses(target, self.policy)
        if await uow.webhooks.count(tenant_id) >= self.tuning.max_subscriptions_per_tenant:
            raise Conflict(
                "the tenant has reached its webhook subscription limit",
                details={"limit": self.tuning.max_subscriptions_per_tenant},
            )
        secret = secrets.token_hex(SECRET_BYTES)
        draft = WebhookSubscription(
            subscription_id=subscription_id or new_id("webhook"),
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            url=target,
            events=sorted(set(events), key=lambda e: e.value),
            description=description,
            created_by=created_by,
            secret_key_id="",
            secret_ciphertext=b"",
        )
        key_id, ciphertext = self.cipher.encrypt(_secret_identity(draft), SecretStr(secret))
        subscription = draft.model_copy(
            update={"secret_key_id": key_id, "secret_ciphertext": ciphertext}
        )
        if await uow.webhooks.get(tenant_id, subscription.subscription_id) is not None:
            raise Conflict(
                "a webhook subscription with this id exists",
                details={"subscription_id": subscription.subscription_id},
            )
        await uow.webhooks.add(subscription)
        return subscription, secret

    async def get(
        self, uow: UnitOfWork, tenant_id: str, subscription_id: str
    ) -> WebhookSubscription:
        subscription = await uow.webhooks.get(tenant_id, subscription_id)
        if subscription is None:
            raise NotFound(f"webhook subscription {subscription_id} not found")
        return subscription

    async def update(
        self, uow: UnitOfWork, tenant_id: str, subscription_id: str, changes: Mapping[str, Any]
    ) -> WebhookSubscription:
        values = dict(changes)
        if "url" in values:
            values["url"] = validate_url(str(values["url"]), self.policy)
            await resolved_addresses(values["url"], self.policy)
        if "events" in values:
            values["events"] = sorted({WebhookEvent(e).value for e in values["events"]})
        if values.get("enabled") is True:
            values["failures"] = 0  # re-enabling forgives the endpoint's history
        updated = await uow.webhooks.update(tenant_id, subscription_id, values)
        if updated is None:
            raise NotFound(f"webhook subscription {subscription_id} not found")
        return updated

    async def delete(self, uow: UnitOfWork, tenant_id: str, subscription_id: str) -> bool:
        return await uow.webhooks.delete(tenant_id, subscription_id)

    async def test(self, uow: UnitOfWork, subscription: WebhookSubscription) -> WebhookDelivery:
        """Queue one ``webhook.test`` delivery to this subscription, whatever it subscribes to."""
        event = Event(
            type=WebhookEvent.TEST,
            tenant_id=subscription.tenant_id,
            workspace_id=subscription.workspace_id,
            data={"subscription_id": subscription.subscription_id},
        )
        return await self._schedule(uow, subscription, event)

    # ------------------------------------------------------------------ publishing
    async def publish(self, uow: UnitOfWork, event: Event) -> None:
        if not await uow.webhooks.any_enabled(event.tenant_id):
            return
        await uow.enqueue(
            JobSpec(
                task_name=TASK_WEBHOOK_FANOUT,
                queue=Queue.RECONCILE,
                payload=event.model_dump(mode="json"),
                idempotency_key=f"webhook:fanout:{event.event_id}",
                tenant_id=event.tenant_id,
            )
        )

    async def fanout(self, payload: dict[str, Any]) -> int:
        """The ``webhooks.fanout`` job: one delivery per subscription that wants the event."""
        event = Event.model_validate(payload)
        async with self.uow_factory() as uow:
            subscriptions = await uow.webhooks.matching(event)
            for subscription in subscriptions:
                await self._schedule(uow, subscription, event)
            await uow.commit()
        return len(subscriptions)

    async def _schedule(
        self, uow: UnitOfWork, subscription: WebhookSubscription, event: Event
    ) -> WebhookDelivery:
        delivery = WebhookDelivery(
            tenant_id=subscription.tenant_id,
            subscription_id=subscription.subscription_id,
            event_id=event.event_id,
            event_type=event.type,
            payload=event.model_dump(mode="json"),
        )
        await uow.webhook_deliveries.add(delivery)
        await uow.enqueue(
            JobSpec(
                task_name=TASK_WEBHOOK_DELIVER,
                queue=Queue.RECONCILE,
                payload={"tenant_id": delivery.tenant_id, "delivery_id": delivery.delivery_id},
                idempotency_key=f"webhook:deliver:{delivery.delivery_id}",
                tenant_id=delivery.tenant_id,
            )
        )
        return delivery

    # ------------------------------------------------------------------ delivery
    async def deliver(self, payload: dict[str, Any]) -> bool:
        """The ``webhooks.deliver`` job: one attempt. Raises ``DeliveryError`` for the queue
        to retry; the last permitted attempt (or a permanent failure) records DEAD and counts
        against the subscription instead."""
        tenant_id, delivery_id = payload["tenant_id"], payload["delivery_id"]
        async with self.uow_factory() as uow:
            delivery = await uow.webhook_deliveries.get(tenant_id, delivery_id)
            subscription = (
                await uow.webhooks.get(tenant_id, delivery.subscription_id) if delivery else None
            )
        if delivery is None or delivery.status in SETTLED:
            return delivery is not None and delivery.status is DeliveryStatus.DELIVERED
        if subscription is None or not subscription.enabled:
            await self._record(delivery, DeliveryStatus.DEAD, None, "subscription gone or disabled")
            return False
        final = delivery.attempts + 1 >= self.tuning.max_attempts
        try:
            status_code = await self._post(subscription, delivery)
        except DeliveryError as failed:
            dead = final or failed.permanent
            status = DeliveryStatus.DEAD if dead else DeliveryStatus.FAILED
            await self._record(delivery, status, failed.status_code, failed.error, attempted=True)
            if dead:
                await self._count_failure(subscription)
                return False
            raise
        await self._record(delivery, DeliveryStatus.DELIVERED, status_code, None, attempted=True)
        async with self.uow_factory() as uow:
            await uow.webhooks.record_outcome(
                subscription.tenant_id, subscription.subscription_id, delivered=True
            )
            await uow.commit()
        return True

    async def _post(self, subscription: WebhookSubscription, delivery: WebhookDelivery) -> int:
        """One signed POST, pinned to an address vetted *now*: the name is resolved and
        checked again, and the connection goes to that address with the original host in
        ``Host`` and SNI, so a record that changes between check and connect cannot steer
        the request onto the local network."""
        try:
            addresses = await resolved_addresses(subscription.url, self.policy)
            secret = self.cipher.decrypt(
                StoredCredential(
                    _secret_identity(subscription),
                    subscription.secret_key_id,
                    subscription.secret_ciphertext,
                    0,
                    subscription.updated_at,
                )
            ).get_secret_value()
        except (TargetRefused, DependencyUnavailable) as exc:
            raise DeliveryError(None, f"{type(exc).__name__}: {exc}", permanent=True) from exc
        parts = urlsplit(subscription.url)
        host = parts.hostname or ""
        pinned_host = f"[{addresses[0]}]" if ":" in addresses[0] else addresses[0]
        netloc = f"{pinned_host}:{parts.port}" if parts.port else pinned_host
        pinned = urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))
        body = Event.model_validate(delivery.payload).model_dump_json().encode()
        headers = {
            "Host": parts.netloc,
            "Content-Type": "application/json",
            WEBHOOK_HEADERS.event: delivery.event_type.value,
            WEBHOOK_HEADERS.delivery: delivery.delivery_id,
            WEBHOOK_HEADERS.signature: sign(secret, int(time.time()), body),
            HEADERS.request_id: new_id("request"),
            HEADERS.traceparent: format_traceparent(
                new_trace_id(), current_span_id() or new_span_id(), current_trace_flags()
            ),
        }
        extensions = {"sni_hostname": host} if parts.scheme == "https" else {}
        try:
            async with (
                self._client_factory() as client,
                client.stream(
                    "POST", pinned, content=body, headers=headers, extensions=extensions
                ) as response,
            ):
                await self._drain(response)
        except httpx.HTTPError as exc:
            raise DeliveryError(None, f"{type(exc).__name__}: {exc}") from exc
        if response.status_code >= 300:
            raise DeliveryError(response.status_code, f"HTTP {response.status_code}")
        return response.status_code

    async def _drain(self, response: httpx.Response) -> None:
        """Read at most the cap; a receiver's answer is an acknowledgement, not content."""
        consumed = 0
        async for chunk in response.aiter_bytes():
            consumed += len(chunk)
            if consumed > self.tuning.response_read_cap_bytes:
                break

    async def _record(
        self,
        delivery: WebhookDelivery,
        status: DeliveryStatus,
        status_code: int | None,
        error: str | None,
        *,
        attempted: bool = False,
    ) -> None:
        async with self.uow_factory() as uow:
            await uow.webhook_deliveries.mark(
                delivery.tenant_id,
                delivery.delivery_id,
                status=status,
                attempted=attempted,
                status_code=status_code,
                error=error[:ERROR_MAX_CHARS] if error else None,
                delivered_at=_now() if status is DeliveryStatus.DELIVERED else None,
            )
            await uow.commit()

    async def _count_failure(self, subscription: WebhookSubscription) -> None:
        async with self.uow_factory() as uow:
            failures = await uow.webhooks.record_outcome(
                subscription.tenant_id, subscription.subscription_id, delivered=False
            )
            if failures >= self.tuning.disable_after_failures:
                await uow.webhooks.update(
                    subscription.tenant_id, subscription.subscription_id, {"enabled": False}
                )
                log.warning(
                    "webhook.disabled",
                    tenant_id=subscription.tenant_id,
                    subscription_id=subscription.subscription_id,
                    failures=failures,
                )
            await uow.commit()

    async def purge(self, *, now: datetime | None = None) -> int:
        """The ``webhooks.purge_deliveries`` job: drop delivery rows past their retention."""
        cutoff = (now or _now()) - timedelta(days=self.tuning.delivery_retention_days)
        async with self.uow_factory() as uow:
            purged = await uow.webhook_deliveries.purge_before(cutoff)
            await uow.commit()
        if purged:
            log.info("webhooks.deliveries_purged", count=purged)
        return purged
