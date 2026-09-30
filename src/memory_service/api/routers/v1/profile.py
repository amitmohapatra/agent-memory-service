"""Public /v1/profile routes: the pinned profile blocks of the caller's user, agent and
workspace - read, replaced, edited in place."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from memory_service.api.deps import (
    ContainerDep,
    HeaderContextDep,
    ScopeBody,
    ServicePrincipalDep,
    build_context,
)
from memory_service.api.errors import error_responses
from memory_service.domain.profile import PROFILE_BLOCK_MAX_CHARS, ProfileBlock

router = APIRouter()
_ERRORS = error_responses(401, 403, 404, 422, 503)

_BLOCK_DESCRIPTION = (
    "user (the person the agent acts for), agent (this agent, for this user), workspace (the "
    "team), or one of them followed by .<name> (user.preferences)"
)


class ProfileBlockBody(BaseModel):
    block: str = Field(description=_BLOCK_DESCRIPTION)
    text: str
    version: int
    updated_at: datetime

    @classmethod
    def of(cls, block: ProfileBlock) -> ProfileBlockBody:
        return cls(
            block=block.block, text=block.text, version=block.version, updated_at=block.updated_at
        )


class ProfileResponse(BaseModel):
    blocks: list[ProfileBlockBody]


class PutBlockRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"text": "name: Ann\ndelivery address: Hauptstr. 1"}]},
    )

    scope: ScopeBody = Field(default_factory=ScopeBody)
    text: str = Field(..., max_length=PROFILE_BLOCK_MAX_CHARS)


class EditBlockRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"old": "Hauptstr. 1", "new": "Ringstr. 9"}]},
    )

    scope: ScopeBody = Field(default_factory=ScopeBody)
    old: str = Field(..., min_length=1, max_length=PROFILE_BLOCK_MAX_CHARS)
    new: str = Field(..., max_length=PROFILE_BLOCK_MAX_CHARS)


@router.get(
    "/profile",
    response_model=ProfileResponse,
    tags=["profile"],
    summary="The pinned profile blocks of this user, agent and workspace",
    responses=_ERRORS,
)
async def get_profile(ctx: HeaderContextDep, container: ContainerDep) -> ProfileResponse:
    async with container.services["uow_factory"]() as uow:
        blocks = await container.services["profile"].blocks(uow, ctx)
    return ProfileResponse(blocks=[ProfileBlockBody.of(b) for b in blocks])


@router.put(
    "/profile/{block}",
    response_model=ProfileBlockBody,
    tags=["profile"],
    summary="Replace a profile block's text",
    responses=_ERRORS,
)
async def put_profile_block(
    block: str,
    request: Request,
    body: PutBlockRequest,
    container: ContainerDep,
    _: ServicePrincipalDep,
) -> ProfileBlockBody:
    ctx = build_context(request, container, body.scope)
    async with container.services["uow_factory"]() as uow:
        stored = await container.services["profile"].put(uow, ctx, block, body.text)
        await uow.commit()
    return ProfileBlockBody.of(stored)


@router.patch(
    "/profile/{block}",
    response_model=ProfileBlockBody,
    tags=["profile"],
    summary="Replace one occurrence of old with new in a profile block (409 when old is gone)",
    responses=error_responses(401, 403, 404, 409, 422, 503),
)
async def edit_profile_block(
    block: str,
    request: Request,
    body: EditBlockRequest,
    container: ContainerDep,
    _: ServicePrincipalDep,
) -> ProfileBlockBody:
    ctx = build_context(request, container, body.scope)
    async with container.services["uow_factory"]() as uow:
        stored = await container.services["profile"].edit(uow, ctx, block, body.old, body.new)
        await uow.commit()
    return ProfileBlockBody.of(stored)
