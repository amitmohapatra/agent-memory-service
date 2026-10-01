"""Public /v1 routes for memory: stated memories in (stored now), canonical memories out,
superseded or forgotten on request. A memory is named by its id, or by the handle a context
cited it by (``m3``) together with that context's ``bundle_id`` - or, in an agent run, the
run's latest context."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_service.api.deps import (
    ContainerDep,
    HeaderContextDep,
    ScopeBody,
    ServicePrincipalDep,
    build_context,
)
from memory_service.api.errors import error_responses
from memory_service.api.idempotent import (
    default_idempotency_key,
    derived_or_body,
    run_idempotent,
)
from memory_service.api.pagination import CursorQuery, decode_cursor, link_next, page
from memory_service.api.validation import CustomMetadata
from memory_service.domain.enums import (
    Lifetime,
    MemoryType,
    ScopeLevel,
    TemporalStatus,
    Visibility,
)
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.memory import CanonicalMemory
from memory_service.modules.memory.service import MemoryService

router = APIRouter()
#: 404: WORKSPACE visibility naming a workspace that is not a team (modules/tenancy/gate.py)
_WRITE_ERRORS = error_responses(401, 403, 404, 409, 422, 503)
_READ_ERRORS = error_responses(401, 403, 404, 422, 503)

_SCOPE: dict[str, Any] = {"thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH"}
_REMEMBER_EXAMPLE: dict[str, Any] = {
    "scope": {},
    "content": "Prefers metric units.",
    "memory_type": "PREFERENCE",
    "visibility": "USER",
}


class RememberRequest(BaseModel):
    """'This is true': stored verbatim as one memory, now. Nothing is extracted from it."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [_REMEMBER_EXAMPLE]})

    scope: ScopeBody = Field(default_factory=ScopeBody)
    content: str = Field(..., min_length=1, max_length=8_000)
    memory_type: MemoryType = Field(
        default=MemoryType.SEMANTIC,
        description=(
            "The kind of memory: SEMANTIC (a fact), PREFERENCE, EPISODIC (something that "
            "happened), PROCEDURAL (how to do something), TASK, USER (a profile attribute), "
            "TOOL or OUTCOME are the ones a caller normally states."
        ),
    )
    lifetime: Lifetime = Field(
        default=Lifetime.LONG_TERM,
        description="SHORT_TERM (the current thread or session) or LONG_TERM (durable); "
        "EPHEMERAL and ARCHIVAL are not stored through this route.",
    )
    visibility: Visibility | None = Field(
        default=None,
        description="Who may retrieve it, narrowest first: PRIVATE, RUN, THREAD, WORK, "
        "AGENT_GROUP, GROUP, USER, WORKSPACE, TENANT, GLOBAL. Omit for the scope's own.",
    )
    subject: str | None = Field(
        default=None,
        max_length=300,
        description="The entity the memory is about; defaults to the user for USER and "
        "PREFERENCE memories",
    )
    entities: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Entities the memory names: linked in the graph and anchored in search",
    )
    valid_from: datetime | None = Field(default=None, description="When it became true")
    valid_to: datetime | None = Field(default=None, description="When it stops being true")
    custom_metadata: CustomMetadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def _storable(self) -> RememberRequest:
        if self.lifetime in (Lifetime.EPHEMERAL, Lifetime.ARCHIVAL):
            raise ValueError("lifetime must be SHORT_TERM or LONG_TERM")
        if self.memory_type is MemoryType.CUSTOM:
            raise ValueError("CUSTOM is not a type a caller states: use a typed memory_type")
        if self.valid_from and self.valid_to and self.valid_to <= self.valid_from:
            raise ValueError("valid_to must be after valid_from")
        return self


class RememberResponse(BaseModel):
    memory_id: str
    deduplicated: bool = Field(
        description="true: the same content was already a current memory in this scope, "
        "and that memory's id is returned"
    )
    job_ids: list[str] = Field(
        default_factory=list, description="the index and graph job queued for it"
    )


class SupersedeRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"content": "Prefers imperial units.", "reason": "user corrected it"}]
        },
    )

    scope: ScopeBody = Field(default_factory=ScopeBody)
    content: str = Field(..., min_length=1, max_length=8_000)
    reason: str = Field(default="", max_length=500, description="why it changed")
    bundle_id: str | None = Field(
        default=None, max_length=64, description="the context whose handle the path names"
    )


class SupersedeResponse(BaseModel):
    memory_id: str = Field(description="the new, current version")
    supersedes: str = Field(description="the version it replaced, now SUPERSEDED")


class MemoryResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    memory_id: str
    content: str
    memory_type: MemoryType = Field(
        ...,
        description=(
            "What kind of intelligence the memory carries. Extracted memories are SEMANTIC "
            "(a fact about the world), PREFERENCE, EPISODIC, DECISION or PROCEDURAL; the "
            "pipeline itself writes ENTITY_SUMMARY, OBSERVATION, BELIEF, TOOL, AGENT, TASK "
            "and USER; the remaining values come from imports that declared their type."
        ),
    )
    lifetime: Lifetime = Field(
        ...,
        description="EPHEMERAL (within the turn), SHORT_TERM (the current thread or session), "
        "LONG_TERM (durable) or ARCHIVAL (kept for audit, out of retrieval).",
    )
    visibility: Visibility = Field(
        ...,
        description="Who may retrieve it, narrowest first: PRIVATE, RUN, THREAD, WORK, "
        "AGENT_GROUP, GROUP, USER, WORKSPACE, TENANT, GLOBAL.",
    )
    scope_level: ScopeLevel = Field(
        ...,
        description="Where the memory is anchored (distinct from visibility): AGENT, "
        "AGENT_GROUP, WORK, THREAD, USER, GROUP, WORKSPACE, TENANT or GLOBAL.",
    )
    owner_principal: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    temporal_status: TemporalStatus = Field(
        ...,
        description="CURRENT is the live value; SUPERSEDED was replaced by a newer memory "
        "(see superseded_by); CONTRADICTED conflicts with a current one; EXPIRED passed its "
        "valid_to; RETRACTED was withdrawn; ARCHIVED was forgotten by policy but kept.",
    )
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    observed_at: datetime
    supersedes: str | None = None
    superseded_by: str | None = None
    confidence: float
    importance: float
    reinforcement_count: int
    contributors: list[str] = Field(
        default_factory=list, description="Other principals that corroborated this memory"
    )
    echoes: int = Field(
        default=0,
        description=(
            "How much of reinforcement_count is the agent restating its own output. Echoes "
            "are counted but never raise confidence — without this the count cannot be read: "
            "seven repeats and no contributors is either corroboration or a recall loop."
        ),
    )
    contradicts: list[str] = Field(
        default_factory=list, description="CURRENT memories this one conflicts with"
    )
    evidence: list[EvidenceRef]
    category: str | None = None
    created_at: datetime
    updated_at: datetime
    indexed_at: datetime | None = Field(
        default=None,
        description=(
            "When this memory was last written to the search index. A memory is listed as "
            "soon as it is stored, but it is only retrievable once the index job has run, "
            "and the two are different moments: null means stored-but-not-yet-searchable. "
            "Null is also a normal steady state after an edit - changing a memory clears "
            "this until the re-index lands - so it reads as 'not yet', never as 'broken'."
        ),
    )


class MemoryListResponse(BaseModel):
    memories: list[MemoryResponse]
    next_cursor: str | None = Field(
        default=None, description="pass as `cursor` for the next page; null on the last"
    )


def memory_to_api(m: CanonicalMemory) -> dict[str, Any]:
    return {
        "memory_id": m.memory_id,
        "content": m.content,
        "memory_type": m.memory_type.value,
        "lifetime": m.lifetime.value,
        "visibility": m.visibility.value,
        "scope_level": m.scope.level.value,
        "owner_principal": m.owner_principal,
        "subject": m.subject,
        "predicate": m.predicate,
        "object": m.object,
        "temporal_status": m.temporal.status.value,
        "valid_from": m.temporal.valid_from,
        "valid_to": m.temporal.valid_to,
        "observed_at": m.temporal.observed_at,
        "supersedes": m.temporal.supersedes,
        "superseded_by": m.temporal.superseded_by,
        "confidence": m.confidence,
        "importance": m.importance,
        "reinforcement_count": m.reinforcement_count,
        "contributors": list(m.system_metadata.get("contributors") or []),
        "echoes": int(m.system_metadata.get("echoes") or 0),
        "contradicts": list(m.temporal.contradicts),
        "evidence": [e.model_dump(mode="json", exclude_none=True) for e in m.evidence],
        "category": m.system_metadata.get("category"),
        "created_at": m.created_at,
        "updated_at": m.updated_at,
        "indexed_at": m.system_metadata.get("indexed_at"),
    }


def _service(container: Any) -> MemoryService:
    return container.services["memory"]


@router.post(
    "/memories",
    response_model=RememberResponse,
    status_code=201,
    tags=["memory"],
    summary="Remember a statement verbatim as one memory (stored now, indexed asynchronously)",
    responses={
        **_WRITE_ERRORS,
        200: {"model": RememberResponse, "description": "Replayed (Idempotency-Key seen before)"},
    },
)
async def remember(
    request: Request, body: RememberRequest, container: ContainerDep, _: ServicePrincipalDep
) -> JSONResponse:
    ctx = build_context(request, container, body.scope)
    identity = ("remember", body.memory_type.value, body.content)
    key = request.state.idempotency_key or default_idempotency_key(ctx, *identity)
    payload = derived_or_body(request, body, identity)

    async def handler(uow):  # type: ignore[no-untyped-def]
        if ctx.thread_id:
            # A memory naming a thread implies the thread, the same way a message does:
            # without it the thread is never granted, and a THREAD-scoped memory is readable
            # only through its author's own key - and from every other thread too.
            # create_thread is get-or-create and requires access to an existing one.
            await container.services["conversation"].create_thread(uow, ctx)
        ack = await _service(container).remember(
            uow,
            ctx,
            content=body.content,
            memory_type=body.memory_type,
            lifetime=body.lifetime,
            visibility=body.visibility,
            subject=body.subject,
            entities=body.entities,
            valid_from=body.valid_from,
            valid_to=body.valid_to,
            custom_metadata=body.custom_metadata,
        )
        return 201, RememberResponse(**ack.__dict__).model_dump(mode="json"), None

    return await run_idempotent(request, container, ctx, key=key, payload=payload, handler=handler)


@router.post(
    "/memories/{memory_id}/supersede",
    response_model=SupersedeResponse,
    tags=["memory"],
    summary="Replace a memory with a new version; the old one is closed, not deleted",
    responses=_WRITE_ERRORS,
)
async def supersede_memory(
    request: Request,
    memory_id: str,
    body: SupersedeRequest,
    container: ContainerDep,
    _: ServicePrincipalDep,
) -> JSONResponse:
    ctx = build_context(request, container, body.scope)
    memory_id = await container.services["bundle_records"].resolve(
        ctx, memory_id, bundle_id=body.bundle_id
    )
    identity = ("supersede", memory_id, body.content)
    key = request.state.idempotency_key or default_idempotency_key(ctx, *identity)
    payload = derived_or_body(request, body, identity)

    async def handler(uow):  # type: ignore[no-untyped-def]
        new = await _service(container).supersede(
            uow, ctx, memory_id, content=body.content, reason=body.reason or "updated"
        )
        out = SupersedeResponse(memory_id=new.memory_id, supersedes=memory_id)
        return 200, out.model_dump(mode="json"), None

    return await run_idempotent(request, container, ctx, key=key, payload=payload, handler=handler)


@router.get(
    "/memories",
    response_model=MemoryListResponse,
    tags=["memory"],
    summary="List current memories anchored to the caller's scopes, newest created first",
    responses=_READ_ERRORS,
)
async def list_memories(
    request: Request,
    response: Response,
    container: ContainerDep,
    _: ServicePrincipalDep,
    memory_type: Annotated[
        list[MemoryType] | None,
        Query(
            description=(
                "Keep only these memory types (repeat the parameter for several). The ones "
                "a caller usually wants: SEMANTIC, PREFERENCE, EPISODIC, DECISION, "
                "PROCEDURAL; omit for every type."
            )
        ),
    ] = None,
    include_superseded: bool = False,
    cursor: CursorQuery = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    thread_id: str | None = None,
    work_id: str | None = None,
    agent_id: str | None = None,
    agent_group_id: str | None = None,
) -> MemoryListResponse:
    """Security fields come from trusted headers; lineage anchors (thread, work, agent) are
    query parameters so a caller can list the memories of a specific thread or agent.
    Newest created first; the page is `limit` visible memories and `next_cursor` (also the
    `Link` header) names the next one."""
    position = decode_cursor(cursor, fields={"created_at": datetime, "memory_id": str})
    before = (position["created_at"], position["memory_id"]) if position else None
    ctx = build_context(
        request,
        container,
        ScopeBody(
            thread_id=thread_id, work_id=work_id, agent_id=agent_id, agent_group_id=agent_group_id
        ),
    )
    async with container.services["uow_factory"]() as uow:
        rows = await _service(container).list_memories(
            uow,
            ctx,
            memory_types=[m.value for m in memory_type] if memory_type else None,
            include_superseded=include_superseded,
            before=before,
            limit=limit + 1,
        )
    items, next_cursor = page(
        rows,
        limit=limit,
        position=lambda m: {"created_at": m.created_at.isoformat(), "memory_id": m.memory_id},
    )
    link_next(request, response, next_cursor)
    return MemoryListResponse(
        memories=[MemoryResponse(**memory_to_api(m)) for m in items], next_cursor=next_cursor
    )


@router.get(
    "/memories/{memory_id}",
    response_model=MemoryResponse,
    tags=["memory"],
    summary="Get a memory (with evidence and temporal state)",
    responses=_READ_ERRORS,
)
async def get_memory(
    memory_id: str, ctx: HeaderContextDep, container: ContainerDep
) -> MemoryResponse:
    async with container.services["uow_factory"]() as uow:
        memory = await _service(container).get_memory(uow, ctx, memory_id)
    return MemoryResponse(**memory_to_api(memory))


@router.delete(
    "/memories/{memory_id}",
    status_code=204,
    tags=["memory"],
    summary="Forget a memory (soft delete + index removal)",
    responses=_READ_ERRORS,
)
async def forget_memory(
    memory_id: str,
    ctx: HeaderContextDep,
    container: ContainerDep,
    bundle_id: Annotated[
        str | None, Query(max_length=64, description="the context whose handle the path names")
    ] = None,
) -> None:
    memory_id = await container.services["bundle_records"].resolve(
        ctx, memory_id, bundle_id=bundle_id
    )
    async with container.services["uow_factory"]() as uow:
        await _service(container).forget(uow, ctx, memory_id)
        await uow.commit()


@router.post(
    "/memories/{memory_id}/restore",
    response_model=MemoryResponse,
    tags=["memory"],
    summary="Restore a memory automatic forgetting archived (CURRENT and searchable again)",
    responses=_READ_ERRORS,
)
async def restore_memory(
    memory_id: str, ctx: HeaderContextDep, container: ContainerDep
) -> MemoryResponse:
    async with container.services["uow_factory"]() as uow:
        memory = await _service(container).restore(
            uow, ctx, memory_id, container.services["forgetting"]
        )
        await uow.commit()
    return MemoryResponse(**memory_to_api(memory))
