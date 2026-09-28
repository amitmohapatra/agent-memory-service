"""Outbound webhooks (ADR 0023): subscriptions, the events they receive, deliveries.

An event is published in the transaction that made it true; fan-out and delivery run from
the outbox afterwards, so a subscriber sees every event at least once and a slow endpoint
never slows a request. The receiver verifies ``X-Trellis-Signature`` with the secret it
was shown once at subscription time.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from memory_service.domain.ids import is_valid_id, new_id

URL_MAX_CHARS: Final = 2048
DESCRIPTION_MAX_CHARS: Final = 500
ERROR_MAX_CHARS: Final = 500


class WebhookEvent(StrEnum):
    MEMORY_CREATED = "memory.created"
    MEMORY_SUPERSEDED = "memory.superseded"
    MEMORY_RETRACTED = "memory.retracted"
    FEEDBACK_RECEIVED = "feedback.received"
    FEEDBACK_PROJECTED = "feedback.projected"
    TEST = "webhook.test"


class DeliveryStatus(StrEnum):
    PENDING = "PENDING"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"  # an attempt failed and another is scheduled
    DEAD = "DEAD"  # every attempt failed; nothing more will be tried


class Event(BaseModel):
    """What happened, as the receiver sees it (``data`` is the event's own payload)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(default_factory=lambda: new_id("event"))
    type: WebhookEvent
    tenant_id: str
    workspace_id: str | None = None
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data: dict[str, Any] = Field(default_factory=dict)


class WebhookSubscription(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    subscription_id: str = Field(default_factory=lambda: new_id("webhook"))
    tenant_id: str
    workspace_id: str | None = Field(
        default=None, description="only this workspace's events; None means every event"
    )
    url: str = Field(min_length=1, max_length=URL_MAX_CHARS)
    events: list[WebhookEvent] = Field(min_length=1)
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX_CHARS)
    enabled: bool = True
    failures: int = Field(default=0, ge=0, description="consecutive dead deliveries")
    created_by: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    secret_key_id: str = Field(repr=False)
    secret_ciphertext: bytes = Field(repr=False)

    @field_validator("subscription_id")
    @classmethod
    def _identifier(cls, value: str) -> str:
        if not is_valid_id(value):
            raise ValueError(f"invalid identifier {value!r}")
        return value

    def wants(self, event: Event) -> bool:
        if not self.enabled or event.type not in self.events:
            return False
        return self.workspace_id is None or self.workspace_id == event.workspace_id


class WebhookDelivery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    delivery_id: str = Field(default_factory=lambda: new_id("delivery"))
    tenant_id: str
    subscription_id: str
    event_id: str
    event_type: WebhookEvent
    payload: dict[str, Any]
    status: DeliveryStatus = DeliveryStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    status_code: int | None = None
    last_error: str | None = Field(default=None, max_length=ERROR_MAX_CHARS)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    delivered_at: datetime | None = None
