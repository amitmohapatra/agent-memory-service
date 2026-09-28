"""Webhook subscription and delivery repository ports."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from memory_service.domain.webhooks import (
    DeliveryStatus,
    Event,
    WebhookDelivery,
    WebhookSubscription,
)

if TYPE_CHECKING:
    from memory_service.ports.uow import UnitOfWork


@runtime_checkable
class EventPublisher(Protocol):
    async def publish(self, uow: UnitOfWork, event: Event) -> None:
        """Record that ``event`` happened, in the transaction that made it true."""
        ...


@runtime_checkable
class WebhookRepository(Protocol):
    async def add(self, subscription: WebhookSubscription) -> None: ...

    async def get(self, tenant_id: str, subscription_id: str) -> WebhookSubscription | None: ...

    async def list(
        self, tenant_id: str, *, after: str = "", limit: int = 100
    ) -> list[WebhookSubscription]:
        """By subscription id; ``after`` is the last id of the previous page."""
        ...

    async def count(self, tenant_id: str) -> int: ...

    async def update(
        self, tenant_id: str, subscription_id: str, changes: Mapping[str, object]
    ) -> WebhookSubscription | None: ...

    async def delete(self, tenant_id: str, subscription_id: str) -> bool: ...

    async def any_enabled(self, tenant_id: str) -> bool:
        """Whether publishing an event for this tenant can reach anyone (the cheap gate)."""
        ...

    async def matching(self, event: Event) -> list[WebhookSubscription]:
        """The enabled subscriptions of the event's tenant that want it."""
        ...

    async def record_outcome(self, tenant_id: str, subscription_id: str, *, delivered: bool) -> int:
        """Reset (delivered) or increment (dead) the consecutive-failure count; returns it."""
        ...


@runtime_checkable
class WebhookDeliveryRepository(Protocol):
    async def add(self, delivery: WebhookDelivery) -> None: ...

    async def get(self, tenant_id: str, delivery_id: str) -> WebhookDelivery | None: ...

    async def list(
        self,
        tenant_id: str,
        subscription_id: str,
        *,
        before: tuple[datetime, str] | None = None,
        limit: int = 100,
    ) -> list[WebhookDelivery]:
        """Newest first; ``before`` is the (created_at, delivery_id) keyset of the next page."""
        ...

    async def mark(
        self,
        tenant_id: str,
        delivery_id: str,
        *,
        status: DeliveryStatus,
        attempted: bool,
        status_code: int | None,
        error: str | None,
        delivered_at: datetime | None,
    ) -> None:
        """Record an outcome; ``attempted`` counts one more attempt. A DELIVERED row is never
        changed again (a duplicate run of the job cannot regress it)."""
        ...

    async def purge_before(self, before: datetime, *, limit: int = 5000) -> int:
        """Delete delivery rows created before ``before``; returns how many."""
        ...
