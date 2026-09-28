"""Webhook subscriptions and deliveries."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import delete, exists, func, literal, or_, select, tuple_, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from memory_service.adapters.db.orm import WebhookDeliveryRow, WebhookSubscriptionRow
from memory_service.domain.webhooks import (
    DeliveryStatus,
    Event,
    WebhookDelivery,
    WebhookSubscription,
)


def _subscription(r: WebhookSubscriptionRow) -> WebhookSubscription:
    return WebhookSubscription.model_validate(
        {
            "subscription_id": r.subscription_id,
            "tenant_id": r.tenant_id,
            "workspace_id": r.workspace_id,
            "url": r.url,
            "events": list(r.events),
            "description": r.description,
            "enabled": r.enabled,
            "failures": r.failures,
            "created_by": r.created_by,
            "created_at": r.created_at,
            "updated_at": r.updated_at,
            "secret_key_id": r.secret_key_id,
            "secret_ciphertext": r.secret_ciphertext,
        }
    )


def _delivery(r: WebhookDeliveryRow) -> WebhookDelivery:
    return WebhookDelivery.model_validate(
        {
            "delivery_id": r.delivery_id,
            "tenant_id": r.tenant_id,
            "subscription_id": r.subscription_id,
            "event_id": r.event_id,
            "event_type": r.event_type,
            "payload": dict(r.payload),
            "status": r.status,
            "attempts": r.attempts,
            "status_code": r.status_code,
            "last_error": r.last_error,
            "created_at": r.created_at,
            "delivered_at": r.delivered_at,
        }
    )


class SqlWebhookRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def add(self, subscription: WebhookSubscription) -> None:
        values: dict[str, Any] = subscription.model_dump(
            mode="json", exclude={"secret_ciphertext", "created_at", "updated_at"}
        )
        self.s.add(
            WebhookSubscriptionRow(
                **values,
                secret_ciphertext=subscription.secret_ciphertext,
                created_at=subscription.created_at,
                updated_at=subscription.updated_at,
            )
        )
        await self.s.flush()

    async def get(self, tenant_id: str, subscription_id: str) -> WebhookSubscription | None:
        row = await self.s.get(WebhookSubscriptionRow, (tenant_id, subscription_id))
        return _subscription(row) if row is not None else None

    async def list(
        self, tenant_id: str, *, after: str = "", limit: int = 100
    ) -> list[WebhookSubscription]:
        stmt = (
            select(WebhookSubscriptionRow)
            .where(
                WebhookSubscriptionRow.tenant_id == tenant_id,
                WebhookSubscriptionRow.subscription_id > after,
            )
            .order_by(WebhookSubscriptionRow.subscription_id)
            .limit(limit)
        )
        return [_subscription(r) for r in (await self.s.scalars(stmt)).all()]

    async def count(self, tenant_id: str) -> int:
        stmt = select(func.count()).where(WebhookSubscriptionRow.tenant_id == tenant_id)
        return int((await self.s.execute(stmt)).scalar_one())

    async def update(
        self, tenant_id: str, subscription_id: str, changes: Mapping[str, object]
    ) -> WebhookSubscription | None:
        stmt = (
            update(WebhookSubscriptionRow)
            .where(
                WebhookSubscriptionRow.tenant_id == tenant_id,
                WebhookSubscriptionRow.subscription_id == subscription_id,
            )
            .values(**dict(changes), updated_at=datetime.now(UTC))
            .returning(WebhookSubscriptionRow)
        )
        row = (
            await self.s.scalars(stmt, execution_options={"populate_existing": True})
        ).one_or_none()
        return _subscription(row) if row is not None else None

    async def delete(self, tenant_id: str, subscription_id: str) -> bool:
        result = cast(
            CursorResult[Any],
            await self.s.execute(
                delete(WebhookSubscriptionRow).where(
                    WebhookSubscriptionRow.tenant_id == tenant_id,
                    WebhookSubscriptionRow.subscription_id == subscription_id,
                )
            ),
        )
        return (result.rowcount or 0) > 0

    async def any_enabled(self, tenant_id: str) -> bool:
        stmt = select(
            exists().where(
                WebhookSubscriptionRow.tenant_id == tenant_id,
                WebhookSubscriptionRow.enabled.is_(True),
            )
        )
        return bool((await self.s.execute(stmt)).scalar_one())

    async def matching(self, event: Event) -> list[WebhookSubscription]:
        stmt = (
            select(WebhookSubscriptionRow)
            .where(
                WebhookSubscriptionRow.tenant_id == event.tenant_id,
                WebhookSubscriptionRow.enabled.is_(True),
                WebhookSubscriptionRow.events.contains([event.type.value]),
                or_(
                    WebhookSubscriptionRow.workspace_id.is_(None),
                    WebhookSubscriptionRow.workspace_id == event.workspace_id,
                ),
            )
            .order_by(WebhookSubscriptionRow.subscription_id)
        )
        return [_subscription(r) for r in (await self.s.scalars(stmt)).all()]

    async def record_outcome(self, tenant_id: str, subscription_id: str, *, delivered: bool) -> int:
        failures = 0 if delivered else WebhookSubscriptionRow.failures + 1
        stmt = (
            update(WebhookSubscriptionRow)
            .where(
                WebhookSubscriptionRow.tenant_id == tenant_id,
                WebhookSubscriptionRow.subscription_id == subscription_id,
            )
            .values(failures=failures)
            .returning(WebhookSubscriptionRow.failures)
        )
        value = (await self.s.execute(stmt)).scalar_one_or_none()
        return int(value or 0)


class SqlWebhookDeliveryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def add(self, delivery: WebhookDelivery) -> None:
        values: dict[str, Any] = delivery.model_dump(
            mode="json", exclude={"created_at", "delivered_at"}
        )
        self.s.add(
            WebhookDeliveryRow(
                **values, created_at=delivery.created_at, delivered_at=delivery.delivered_at
            )
        )
        await self.s.flush()

    async def get(self, tenant_id: str, delivery_id: str) -> WebhookDelivery | None:
        row = await self.s.get(WebhookDeliveryRow, (tenant_id, delivery_id))
        return _delivery(row) if row is not None else None

    async def list(
        self,
        tenant_id: str,
        subscription_id: str,
        *,
        before: tuple[datetime, str] | None = None,
        limit: int = 100,
    ) -> list[WebhookDelivery]:
        stmt = select(WebhookDeliveryRow).where(
            WebhookDeliveryRow.tenant_id == tenant_id,
            WebhookDeliveryRow.subscription_id == subscription_id,
        )
        if before is not None:
            stmt = stmt.where(
                tuple_(WebhookDeliveryRow.created_at, WebhookDeliveryRow.delivery_id)
                < tuple_(literal(before[0]), literal(before[1]))
            )
        stmt = stmt.order_by(
            WebhookDeliveryRow.created_at.desc(), WebhookDeliveryRow.delivery_id.desc()
        )
        return [_delivery(r) for r in (await self.s.scalars(stmt.limit(limit))).all()]

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
        attempts = WebhookDeliveryRow.attempts + 1 if attempted else WebhookDeliveryRow.attempts
        await self.s.execute(
            update(WebhookDeliveryRow)
            .where(
                WebhookDeliveryRow.tenant_id == tenant_id,
                WebhookDeliveryRow.delivery_id == delivery_id,
                WebhookDeliveryRow.status != DeliveryStatus.DELIVERED.value,
            )
            .values(
                status=status.value,
                attempts=attempts,
                status_code=status_code,
                last_error=error,
                delivered_at=delivered_at,
            )
        )

    async def purge_before(self, before: datetime, *, limit: int = 5000) -> int:
        doomed = (
            select(WebhookDeliveryRow.tenant_id, WebhookDeliveryRow.delivery_id)
            .where(WebhookDeliveryRow.created_at < before)
            .limit(limit)
            .subquery()
        )
        result = cast(
            CursorResult[Any],
            await self.s.execute(
                delete(WebhookDeliveryRow).where(
                    tuple_(WebhookDeliveryRow.tenant_id, WebhookDeliveryRow.delivery_id).in_(
                        select(doomed.c.tenant_id, doomed.c.delivery_id)
                    )
                )
            ),
        )
        return int(result.rowcount or 0)
