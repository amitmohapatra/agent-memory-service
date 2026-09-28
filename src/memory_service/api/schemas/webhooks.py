"""Webhook wire shapes."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_service.domain.webhooks import (
    DESCRIPTION_MAX_CHARS,
    URL_MAX_CHARS,
    DeliveryStatus,
    WebhookDelivery,
    WebhookEvent,
    WebhookSubscription,
)


class WebhookCreateRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "url": "https://<your receiver>/trellis",
                    "events": ["memory.created", "feedback.projected"],
                    "description": "the team's inbox",
                }
            ]
        },
    )

    url: str = Field(min_length=1, max_length=URL_MAX_CHARS)
    events: list[WebhookEvent] = Field(min_length=1)
    workspace_id: str | None = Field(
        default=None, description="only this workspace's events; omit for every event"
    )
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX_CHARS)
    subscription_id: str | None = Field(
        default=None, max_length=200, description="your own id; generated when omitted"
    )


class WebhookUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str | None = Field(default=None, min_length=1, max_length=URL_MAX_CHARS)
    events: list[WebhookEvent] | None = Field(default=None, min_length=1)
    enabled: bool | None = None
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX_CHARS)

    @model_validator(mode="after")
    def _something_to_change(self) -> WebhookUpdateRequest:
        if not self.model_fields_set:
            raise ValueError("nothing to change: name url, events, enabled or description")
        for name in ("url", "events", "enabled"):
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(f"{name} cannot be null; omit it to leave it unchanged")
        return self


class WebhookResponse(BaseModel):
    subscription_id: str
    tenant_id: str
    workspace_id: str | None
    url: str
    events: list[WebhookEvent]
    description: str | None
    enabled: bool
    failures: int = Field(description="consecutive dead deliveries; the endpoint's health")
    created_by: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, subscription: WebhookSubscription) -> WebhookResponse:
        return cls.model_validate(
            subscription.model_dump(exclude={"secret_key_id", "secret_ciphertext"})
        )


class WebhookCreatedResponse(WebhookResponse):
    secret: str | None = Field(
        description="the HMAC secret, shown once; null on an idempotent replay"
    )

    @classmethod
    def created(cls, subscription: WebhookSubscription, secret: str) -> WebhookCreatedResponse:
        return cls.model_validate(
            {**WebhookResponse.of(subscription).model_dump(), "secret": secret}
        )


class WebhookListResponse(BaseModel):
    webhooks: list[WebhookResponse]
    next_cursor: str | None = None


class DeliveryResponse(BaseModel):
    delivery_id: str
    subscription_id: str
    event_id: str
    event_type: WebhookEvent
    status: DeliveryStatus
    attempts: int
    status_code: int | None
    last_error: str | None
    created_at: datetime
    delivered_at: datetime | None

    @classmethod
    def of(cls, delivery: WebhookDelivery) -> DeliveryResponse:
        return cls.model_validate(delivery.model_dump(exclude={"payload", "tenant_id"}))


class DeliveryListResponse(BaseModel):
    deliveries: list[DeliveryResponse]
    next_cursor: str | None = None
