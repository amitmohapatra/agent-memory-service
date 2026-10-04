"""Public /v1/profile routes: the pinned profile blocks of the caller's user, agent and
workspace - read, and edited in place (text, or a standing question the service answers)."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_service.api.deps import (
    ContainerDep,
    HeaderContextDep,
    ScopeBody,
    ServicePrincipalDep,
    build_context,
)
from memory_service.api.errors import error_responses
from memory_service.api.idempotent import run_idempotent
from memory_service.api.params import ProfileBlockPath
from memory_service.domain.profile import (
    PROFILE_BLOCK_MAX_CHARS,
    SOURCE_QUERY_MAX_CHARS,
    ProfileBlock,
)

router = APIRouter()
_ERRORS = error_responses(401, 403, 404, 422, 503)

_BLOCK_DESCRIPTION = (
    "user (the person the agent acts for), agent (this agent, for this user), workspace (the "
    "team), or one of them followed by .<name> (user.preferences)"
)


class ProfileBlockBody(BaseModel):
    block: str = Field(description=_BLOCK_DESCRIPTION)
    text: str = Field(description="The block's text, as the context renders it.")
    version: int = Field(description="How many times the block was edited (1 for the first write).")
    updated_at: datetime = Field(description="When the record last changed (ISO 8601, UTC).")
    source_query: str | None = Field(
        default=None, description="the standing question the service keeps this block answering"
    )

    @classmethod
    def of(cls, block: ProfileBlock) -> ProfileBlockBody:
        return cls(
            block=block.block,
            text=block.text,
            version=block.version,
            updated_at=block.updated_at,
            source_query=block.source_query,
        )


class ProfileResponse(BaseModel):
    blocks: list[ProfileBlockBody] = Field(
        description="The user's, the agent's and the workspace's blocks."
    )


class EditBlockRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"old": "Hauptstr. 1", "new": "Ringstr. 9"},
                {"new": "name: Ann\ndelivery address: Hauptstr. 1"},
                {"source_query": "Which suppliers does this team buy from, and on what terms?"},
            ]
        },
    )

    scope: ScopeBody = Field(
        default_factory=ScopeBody,
        description="The lineage the call acts in (thread, session, turn, work, agent, "
        "run). Tenant, workspace and user come from the trusted headers; a "
        "value here must agree with them.",
    )
    old: str = Field(
        default="",
        max_length=PROFILE_BLOCK_MAX_CHARS,
        description="the exact text to replace; empty: new replaces the whole block",
    )
    new: str | None = Field(
        default=None, max_length=PROFILE_BLOCK_MAX_CHARS, description="omitted: text unchanged"
    )
    source_query: str | None = Field(
        default=None,
        max_length=SOURCE_QUERY_MAX_CHARS,
        description="a standing question the service answers into this block now and every "
        "hour; null removes it, omitted keeps it",
    )

    @model_validator(mode="after")
    def _changes_something(self) -> EditBlockRequest:
        if self.new is None and "source_query" not in self.model_fields_set:
            raise ValueError("send new, source_query, or both")
        if self.old and self.new is None:
            raise ValueError("old needs new")
        return self


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


@router.patch(
    "/profile/{block}",
    response_model=ProfileBlockBody,
    tags=["profile"],
    summary="Edit a profile block: replace old with new (the whole text when old is empty; "
    "409 when old is gone), and set or clear its standing question",
    responses=error_responses(401, 403, 404, 409, 422, 503),
)
async def edit_profile_block(
    block: ProfileBlockPath,
    request: Request,
    body: EditBlockRequest,
    container: ContainerDep,
    _: ServicePrincipalDep,
) -> Response:
    """With ``Idempotency-Key``, a retry of an edit that succeeded gets the edited block
    again rather than the 409 its ``old`` text, now replaced, would earn."""
    ctx = build_context(request, container, body.scope)

    async def handler(uow):  # type: ignore[no-untyped-def]
        stored = await container.services["profile"].edit(
            uow,
            ctx,
            block,
            body.old,
            body.new,
            **(
                {"source_query": body.source_query}
                if "source_query" in body.model_fields_set
                else {}
            ),
        )
        return 200, ProfileBlockBody.of(stored).model_dump(mode="json"), None

    return await run_idempotent(
        request,
        container,
        ctx,
        key=request.state.idempotency_key,
        payload={"block": block, **body.model_dump(mode="json", exclude_unset=True)},
        handler=handler,
    )
