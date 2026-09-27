"""Register/rotate/revoke the acting agent's model key; never return secret material."""

import hashlib
from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from memory_service.api.deps import (
    ContainerDep,
    HeaderContextDep,
    ScopeBody,
    ServicePrincipalDep,
    build_context,
)
from memory_service.api.errors import error_responses
from memory_service.api.idempotent import default_idempotency_key, run_idempotent
from memory_service.ports.credentials import StoredCredential

router = APIRouter()
_ERRORS = error_responses(401, 403, 409, 422, 503)


class AgentKeyRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"scope": {"agent_id": "research"}, "virtual_key": "vk-example"}]
        },
    )

    scope: ScopeBody
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


@router.get(
    "/agents/model-key",
    response_model=AgentKeyStatus,
    tags=["agents"],
    responses=_ERRORS,
    summary="Read the acting agent's model-key status, without its secret",
)
async def key_status(container: ContainerDep, ctx: HeaderContextDep) -> AgentKeyStatus:
    return _status(await container.services["agent_credentials"].metadata(ctx))


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
    digest = hashlib.sha256(body.virtual_key.get_secret_value().encode()).hexdigest()

    async def write(uow):
        record = await container.services["agent_credentials"].set(uow, ctx, body.virtual_key)
        return 200, _status(record).model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.headers.get("Idempotency-Key")
        or default_idempotency_key(ctx, "agent-key-put", ctx.request_id),
        payload={"principal": ctx.principal_id, "action": "rotate", "digest": digest},
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
    async def write(uow):
        record = await container.services["agent_credentials"].set(uow, ctx, None)
        return 200, _status(record).model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.headers.get("Idempotency-Key")
        or default_idempotency_key(ctx, "agent-key-delete", ctx.request_id),
        payload={"principal": ctx.principal_id, "action": "revoke"},
        handler=write,
    )
