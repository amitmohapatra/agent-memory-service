"""Outbound webhook subscriptions, administered per tenant."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse

from memory_service.api.deps import (
    AdministeredTenantDep,
    ContainerDep,
    ServicePrincipalDep,
    request_context,
)
from memory_service.api.errors import error_responses
from memory_service.api.idempotent import run_idempotent
from memory_service.api.pagination import CursorQuery, decode_cursor, link_next, page
from memory_service.api.schemas.webhooks import (
    DeliveryListResponse,
    DeliveryResponse,
    WebhookCreatedResponse,
    WebhookCreateRequest,
    WebhookListResponse,
    WebhookResponse,
    WebhookUpdateRequest,
)
from memory_service.modules.webhooks.service import WebhookService

router = APIRouter(tags=["webhooks"])
_ERRORS = error_responses(401, 403, 404, 409, 422, 503)
_DELIVERY_CURSOR = {"created_at": datetime, "delivery_id": str}


def _service(container) -> WebhookService:  # type: ignore[no-untyped-def]
    return container.services["webhooks"]


def _without_secret(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "secret": None}


@router.post(
    "/webhooks",
    response_model=WebhookCreatedResponse,
    status_code=201,
    responses=_ERRORS,
    summary="Subscribe a URL to events (the signing secret is shown once)",
    description=(
        "The URL must be https and public: loopback, link-local, private and metadata "
        "addresses are refused, at creation and again at every delivery. Deliveries carry "
        "`X-Trellis-Event`, `X-Trellis-Delivery`, `X-Request-ID`, `traceparent` and "
        '`X-Trellis-Signature: t=<unix seconds>,v1=<hex hmac-sha256 of "t.body">`. Send '
        "`Idempotency-Key` to make a retry safe; the replay carries `secret: null`."
    ),
)
async def create_webhook(
    request: Request,
    body: WebhookCreateRequest,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
    principal: ServicePrincipalDep,
) -> JSONResponse:
    ctx = request_context(request, tenant_id)

    async def handler(uow):  # type: ignore[no-untyped-def]
        subscription, secret = await _service(container).subscribe(
            uow,
            tenant_id=tenant_id,
            url=body.url,
            events=body.events,
            created_by=principal.service_id,
            workspace_id=body.workspace_id,
            description=body.description,
            subscription_id=body.subscription_id,
        )
        created = WebhookCreatedResponse.created(subscription, secret)
        return 201, created.model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key,
        payload=body.model_dump(mode="json"),
        handler=handler,
        stored_body=_without_secret,
    )


@router.get(
    "/webhooks",
    response_model=WebhookListResponse,
    responses=_ERRORS,
    summary="List subscriptions (cursor paged)",
)
async def list_webhooks(
    request: Request,
    response: Response,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
    cursor: CursorQuery = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> WebhookListResponse:
    position = decode_cursor(cursor, fields=("subscription_id",))
    async with container.services["uow_factory"]() as uow:
        rows = await uow.webhooks.list(
            tenant_id, after=position["subscription_id"] if position else "", limit=limit + 1
        )
    items, next_cursor = page(
        rows, limit=limit, position=lambda s: {"subscription_id": s.subscription_id}
    )
    link_next(request, response, next_cursor)
    return WebhookListResponse(
        webhooks=[WebhookResponse.of(s) for s in items], next_cursor=next_cursor
    )


@router.get(
    "/webhooks/{subscription_id}",
    response_model=WebhookResponse,
    responses=_ERRORS,
    summary="Read a subscription",
)
async def get_webhook(
    subscription_id: str, container: ContainerDep, tenant_id: AdministeredTenantDep
) -> WebhookResponse:
    async with container.services["uow_factory"]() as uow:
        return WebhookResponse.of(await _service(container).get(uow, tenant_id, subscription_id))


@router.patch(
    "/webhooks/{subscription_id}",
    response_model=WebhookResponse,
    responses=_ERRORS,
    summary="Change a subscription's url, events, description or enabled flag",
)
async def update_webhook(
    subscription_id: str,
    body: WebhookUpdateRequest,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
) -> WebhookResponse:
    changes = body.model_dump(exclude_unset=True)
    async with container.services["uow_factory"]() as uow:
        updated = await _service(container).update(uow, tenant_id, subscription_id, changes)
        await uow.commit()
    return WebhookResponse.of(updated)


@router.delete(
    "/webhooks/{subscription_id}",
    status_code=204,
    responses=_ERRORS,
    summary="Delete a subscription (its delivery history stays readable by id)",
)
async def delete_webhook(
    subscription_id: str, container: ContainerDep, tenant_id: AdministeredTenantDep
) -> Response:
    async with container.services["uow_factory"]() as uow:
        await _service(container).delete(uow, tenant_id, subscription_id)
        await uow.commit()
    return Response(status_code=204)


@router.get(
    "/webhooks/{subscription_id}/deliveries",
    response_model=DeliveryListResponse,
    responses=_ERRORS,
    summary="The subscription's deliveries, newest first (cursor paged)",
)
async def list_deliveries(
    request: Request,
    response: Response,
    subscription_id: str,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
    cursor: CursorQuery = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> DeliveryListResponse:
    position = decode_cursor(cursor, fields=_DELIVERY_CURSOR)
    before = (position["created_at"], position["delivery_id"]) if position else None
    async with container.services["uow_factory"]() as uow:
        await _service(container).get(uow, tenant_id, subscription_id)
        rows = await uow.webhook_deliveries.list(
            tenant_id, subscription_id, before=before, limit=limit + 1
        )
    items, next_cursor = page(
        rows,
        limit=limit,
        position=lambda d: {"created_at": d.created_at.isoformat(), "delivery_id": d.delivery_id},
    )
    link_next(request, response, next_cursor)
    return DeliveryListResponse(
        deliveries=[DeliveryResponse.of(d) for d in items], next_cursor=next_cursor
    )


@router.post(
    "/webhooks/{subscription_id}/test",
    response_model=DeliveryResponse,
    status_code=202,
    responses=_ERRORS,
    summary="Queue a `webhook.test` delivery to the subscription's URL",
)
async def test_webhook(
    subscription_id: str, container: ContainerDep, tenant_id: AdministeredTenantDep
) -> DeliveryResponse:
    async with container.services["uow_factory"]() as uow:
        subscription = await _service(container).get(uow, tenant_id, subscription_id)
        delivery = await _service(container).test(uow, subscription)
        await uow.commit()
    return DeliveryResponse.of(delivery)
