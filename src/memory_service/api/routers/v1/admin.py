"""Platform administration: the bootstrap key onboards tenants and nothing else."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Request, Response

from memory_service.api.deps import ContainerDep, PlatformDep, request_context
from memory_service.api.errors import error_responses
from memory_service.api.idempotent import run_idempotent
from memory_service.api.pagination import CursorQuery, decode_cursor, link_next, page
from memory_service.api.params import TenantIdPath, limit_query
from memory_service.api.schemas.tenancy import (
    CreatedTenantResponse,
    CreateTenantRequest,
    IssuedKeyResponse,
    TenantResponse,
    UpdateTenantRequest,
)
from memory_service.domain.tenancy import PLATFORM_SCOPE
from memory_service.observability.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])
_ERRORS = error_responses(401, 403, 404, 409, 422, 503)
#: reads never conflict
_READ_ERRORS = error_responses(401, 403, 404, 422, 503)


def _service(container):  # type: ignore[no-untyped-def]
    return container.services["tenancy"]


def _without_token(body: dict[str, Any]) -> dict[str, Any]:
    """What a replay may return: the record, never the secret shown on the first response."""
    return {**body, "admin_key": {**body["admin_key"], "token": None}}


@router.post(
    "/tenants",
    response_model=CreatedTenantResponse,
    status_code=201,
    responses=_ERRORS,
    summary="Onboard a tenant and receive its first admin key (shown once)",
    description="Send `Idempotency-Key` to make a retry safe: it returns the same tenant and "
    "key record with `Idempotent-Replayed: true` and `admin_key.token: null`; the secret is "
    "never shown twice. Without the header, a repeated call with a generated id onboards "
    "another tenant, and one with a named id is a 409.",
)
async def create_tenant(
    request: Request, body: CreateTenantRequest, container: ContainerDep, principal: PlatformDep
) -> Response:
    payload = body.model_dump(mode="json")
    ctx = request_context(request, PLATFORM_SCOPE)

    async def handler(uow):  # type: ignore[no-untyped-def]
        tenant, admin = await _service(container).create_tenant(
            uow,
            name=body.name,
            tenant_id=body.tenant_id,
            retention_days=body.retention_days,
            rate_limit_per_minute=body.rate_limit_per_minute,
            created_by=principal.service_id,
        )
        response = CreatedTenantResponse(
            tenant=TenantResponse.of(tenant), admin_key=IssuedKeyResponse.issued(admin)
        )

        async def after_commit() -> None:
            await container.services["api_keys"].forget_missing(admin.key.key_id)
            try:
                await container.services["tenant_registry"].observe(tenant)
            except Exception as exc:  # the refresh loop repairs this within a minute
                log.warning(
                    "tenant_registry.observe_failed", tenant_id=tenant.tenant_id, error=str(exc)
                )

        return 201, response.model_dump(mode="json"), after_commit

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key,
        payload=payload,
        handler=handler,
        stored_body=_without_token,
        location=lambda body: f"/v1/admin/tenants/{body['tenant']['tenant_id']}",
    )


@router.get(
    "/tenants",
    response_model=list[TenantResponse],
    responses=_READ_ERRORS,
    summary="List tenants by id (cursor paged)",
)
async def list_tenants(
    request: Request,
    response: Response,
    container: ContainerDep,
    _: PlatformDep,
    cursor: CursorQuery = None,
    limit: Annotated[int, limit_query(500, "tenants")] = 100,
) -> list[TenantResponse]:
    position = decode_cursor(cursor, fields=("tenant_id",))
    start = position["tenant_id"] if position else ""
    async with container.services["uow_factory"]() as uow:
        tenants = await _service(container).list_tenants(uow, after=start, limit=limit + 1)
    items, next_cursor = page(tenants, limit=limit, position=lambda t: {"tenant_id": t.tenant_id})
    link_next(request, response, next_cursor)
    return [TenantResponse.of(t) for t in items]


@router.get(
    "/tenants/{tenant_id}",
    response_model=TenantResponse,
    responses=_READ_ERRORS,
    summary="Get a tenant",
)
async def get_tenant(
    tenant_id: TenantIdPath, container: ContainerDep, _: PlatformDep
) -> TenantResponse:
    async with container.services["uow_factory"]() as uow:
        return TenantResponse.of(await _service(container).get_tenant(uow, tenant_id))


@router.patch(
    "/tenants/{tenant_id}",
    response_model=TenantResponse,
    responses=_ERRORS,
    summary="Rename, suspend or resume a tenant; set its retention and request quota",
    description="Send `status` only to change it: any request naming a status makes every "
    "key of the tenant re-read the store for the next two minutes (that is what makes a "
    "suspension bite at once, and a retry after a failure safe).",
)
async def update_tenant(
    request: Request,
    tenant_id: TenantIdPath,
    body: UpdateTenantRequest,
    container: ContainerDep,
    _: PlatformDep,
) -> Response:
    verifier = container.services["api_keys"]

    async def handler(uow):  # type: ignore[no-untyped-def]
        tenant, _changed = await _service(container).update_tenant(
            uow,
            tenant_id,
            name=body.name,
            status=body.status,
            retention_days=body.retention_days,
            rate_limit_per_minute=body.rate_limit_per_minute,
            clear_retention=body.clear_retention,
            clear_rate_limit=body.clear_rate_limit,
        )
        if body.status is not None:
            # Keyed on the requested status, not on a change, so a retry after a failure
            # still tombstones; strictly and before the commit, so a cache that is away is a
            # 503 to retry rather than a suspension that keys keep serving through.
            await verifier.invalidate_tenant(tenant_id, strict=True)

        async def after_commit() -> None:
            if body.status is not None:
                await verifier.invalidate_tenant(tenant_id)
            try:
                await container.services["tenant_registry"].observe(tenant)
            except Exception as exc:  # the refresh loop repairs this within a minute
                log.warning("tenant_registry.observe_failed", tenant_id=tenant_id, error=str(exc))

        return 200, TenantResponse.of(tenant).model_dump(mode="json"), after_commit

    return await run_idempotent(
        request,
        container,
        request_context(request, PLATFORM_SCOPE),
        key=request.state.idempotency_key,
        payload={"tenant_id": tenant_id, **body.model_dump(mode="json")},
        handler=handler,
    )
