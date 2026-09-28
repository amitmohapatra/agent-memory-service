"""Register/rotate/revoke model keys at agent, workspace and tenant level; never return
secret material. A call resolves the most specific level that has a key (ADR 0023)."""

import hashlib
from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from memory_service.api.deps import (
    AdministeredTenantDep,
    ContainerDep,
    HeaderContextDep,
    ScopeBody,
    ServicePrincipalDep,
    build_context,
    request_context,
)
from memory_service.api.errors import error_responses
from memory_service.api.idempotent import default_idempotency_key, run_idempotent
from memory_service.domain.errors import NotFound
from memory_service.modules.llm.credentials import ModelCredentials
from memory_service.ports.credentials import (
    ModelIdentity,
    StoredCredential,
    tenant_identity,
    workspace_identity,
)

router = APIRouter()
_ERRORS = error_responses(401, 403, 404, 409, 422, 503)


class AgentKeyRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"scope": {"agent_id": "research"}, "virtual_key": "vk-example"}]
        },
    )

    scope: ScopeBody
    virtual_key: SecretStr = Field(min_length=1, max_length=4096)


class ModelKeyRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"examples": [{"virtual_key": "vk-example"}]}
    )

    virtual_key: SecretStr = Field(min_length=1, max_length=4096)


class AgentKeyStatus(BaseModel):
    registered: bool
    revoked: bool
    revision: int
    updated_at: datetime | None = None


def _status(record: StoredCredential | None) -> AgentKeyStatus:
    return AgentKeyStatus(
        registered=record is not None,
        revoked=record is not None and record.ciphertext is None,
        revision=record.revision if record else 0,
        updated_at=record.updated_at if record else None,
    )


def _service(container) -> ModelCredentials:  # type: ignore[no-untyped-def]
    return container.services["model_credentials"]


def _digest(key: SecretStr) -> str:
    return hashlib.sha256(key.get_secret_value().encode()).hexdigest()


# --------------------------------------------------------------------------- the agent's own


@router.get(
    "/agents/model-key",
    response_model=AgentKeyStatus,
    tags=["agents"],
    responses=_ERRORS,
    summary="Read the acting agent's model-key status, without its secret",
)
async def key_status(container: ContainerDep, ctx: HeaderContextDep) -> AgentKeyStatus:
    return _status(await _service(container).metadata(ctx))


@router.put(
    "/agents/model-key",
    response_model=AgentKeyStatus,
    tags=["agents"],
    responses=_ERRORS,
    summary="Register or rotate the acting agent's encrypted Bifrost virtual key",
)
async def set_key(
    request: Request, body: AgentKeyRequest, container: ContainerDep, _: ServicePrincipalDep
):
    ctx = build_context(request, container, body.scope)

    async def write(uow):  # type: ignore[no-untyped-def]
        record = await _service(container).set(uow, ctx, body.virtual_key)
        return 200, _status(record).model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key
        or default_idempotency_key(ctx, "agent-key-put", ctx.request_id),
        payload={
            "principal": ctx.principal_id,
            "action": "rotate",
            "digest": _digest(body.virtual_key),
        },
        handler=write,
    )


@router.delete(
    "/agents/model-key",
    response_model=AgentKeyStatus,
    tags=["agents"],
    responses=_ERRORS,
    summary="Revoke the acting agent's model key and invalidate assisted read caches",
)
async def revoke_key(request: Request, container: ContainerDep, ctx: HeaderContextDep):
    async def write(uow):  # type: ignore[no-untyped-def]
        record = await _service(container).set(uow, ctx, None)
        return 200, _status(record).model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key
        or default_idempotency_key(ctx, "agent-key-delete", ctx.request_id),
        payload={"principal": ctx.principal_id, "action": "revoke"},
        handler=write,
    )


# --------------------------------------------------------------------------- the team's


async def _workspace_identity(container, tenant_id: str, workspace_id: str) -> ModelIdentity:  # type: ignore[no-untyped-def]
    async with container.services["uow_factory"]() as uow:
        workspace = await uow.workspaces.get(tenant_id, workspace_id)
    if workspace is None or workspace.deleted_at is not None:
        raise NotFound(f"workspace {workspace_id} not found")
    return workspace_identity(tenant_id, workspace_id)


async def _put_level(
    request: Request,
    container,
    tenant_id: str,
    identity: ModelIdentity,
    key: SecretStr | None,
    action: str,
):  # type: ignore[no-untyped-def]
    ctx = request_context(request, tenant_id)

    async def write(uow):  # type: ignore[no-untyped-def]
        record = await _service(container).set_for(uow, identity, key)
        return 200, _status(record).model_dump(mode="json"), None

    payload = {"principal": identity.principal_id, "action": action}
    if key is not None:
        payload["digest"] = _digest(key)
    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key
        or default_idempotency_key(ctx, f"model-key-{action}", ctx.request_id),
        payload=payload,
        handler=write,
    )


@router.get(
    "/workspaces/{workspace_id}/model-key",
    response_model=AgentKeyStatus,
    tags=["tenancy"],
    responses=_ERRORS,
    summary="Read the workspace's model-key status (used by its agents without a key of their own)",
)
async def workspace_key_status(
    workspace_id: str, container: ContainerDep, tenant_id: AdministeredTenantDep
) -> AgentKeyStatus:
    identity = await _workspace_identity(container, tenant_id, workspace_id)
    return _status(await _service(container).metadata_for(identity))


@router.put(
    "/workspaces/{workspace_id}/model-key",
    response_model=AgentKeyStatus,
    tags=["tenancy"],
    responses=_ERRORS,
    summary="Register or rotate the workspace's encrypted Bifrost virtual key",
)
async def set_workspace_key(
    request: Request,
    workspace_id: str,
    body: ModelKeyRequest,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
):
    identity = await _workspace_identity(container, tenant_id, workspace_id)
    return await _put_level(request, container, tenant_id, identity, body.virtual_key, "rotate")


@router.delete(
    "/workspaces/{workspace_id}/model-key",
    response_model=AgentKeyStatus,
    tags=["tenancy"],
    responses=_ERRORS,
    summary="Revoke the workspace's model key (its agents fall back to nothing, not the operator)",
)
async def revoke_workspace_key(
    request: Request, workspace_id: str, container: ContainerDep, tenant_id: AdministeredTenantDep
):
    identity = await _workspace_identity(container, tenant_id, workspace_id)
    return await _put_level(request, container, tenant_id, identity, None, "revoke")


@router.get(
    "/model-key",
    response_model=AgentKeyStatus,
    tags=["tenancy"],
    responses=_ERRORS,
    summary="Read the tenant's model-key status (the last level before the operator key)",
)
async def tenant_key_status(
    container: ContainerDep, tenant_id: AdministeredTenantDep
) -> AgentKeyStatus:
    return _status(await _service(container).metadata_for(tenant_identity(tenant_id)))


@router.put(
    "/model-key",
    response_model=AgentKeyStatus,
    tags=["tenancy"],
    responses=_ERRORS,
    summary="Register or rotate the tenant's encrypted Bifrost virtual key",
)
async def set_tenant_key(
    request: Request,
    body: ModelKeyRequest,
    container: ContainerDep,
    tenant_id: AdministeredTenantDep,
):
    return await _put_level(
        request, container, tenant_id, tenant_identity(tenant_id), body.virtual_key, "rotate"
    )


@router.delete(
    "/model-key",
    response_model=AgentKeyStatus,
    tags=["tenancy"],
    responses=_ERRORS,
    summary="Revoke the tenant's model key",
)
async def revoke_tenant_key(
    request: Request, container: ContainerDep, tenant_id: AdministeredTenantDep
):
    return await _put_level(
        request, container, tenant_id, tenant_identity(tenant_id), None, "revoke"
    )
