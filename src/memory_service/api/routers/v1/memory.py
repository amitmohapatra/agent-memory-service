"""Public /v1 routes for memory intelligence: observations in, canonical memories out."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from memory_service.api.deps import (
    ContainerDep,
    HeaderContextDep,
    ScopeBody,
    ServicePrincipalDep,
    build_context,
)
from memory_service.api.errors import error_responses
from memory_service.api.idempotent import default_idempotency_key, run_idempotent
from memory_service.api.schemas.conversation import ProcessingHintsIn
from memory_service.domain.enums import ObservationKind
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.memory.service import MemoryService

router = APIRouter()
_WRITE_ERRORS = error_responses(401, 403, 409, 422, 503)
_READ_ERRORS = error_responses(401, 403, 404, 422, 503)

_SCOPE: dict[str, Any] = {"thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH"}
_OBS_EXAMPLE: dict[str, Any] = {
    "scope": _SCOPE,
    "kind": "EVENT",
    "content": "My timezone is Europe/Berlin and I prefer concise answers.",
    "hints": {},
}


class ObservationRequest(BaseModel):
    """'This happened or was learned.' The service decides what (if anything) to remember."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [_OBS_EXAMPLE]})

    scope: ScopeBody = Field(default_factory=ScopeBody)
    kind: ObservationKind = Field(default=ObservationKind.EVENT, examples=["EVENT"])
    content: str = Field(..., min_length=1, max_length=100_000)
    hints: ProcessingHintsIn = Field(default_factory=ProcessingHintsIn)
    custom_metadata: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime | None = None
    source_system: str | None = Field(default=None, max_length=100)
    source_id: str | None = Field(default=None, max_length=400)
    tool_run_id: str | None = Field(default=None, max_length=200)


class ObservationAckResponse(BaseModel):
    observation_id: str
    job_ids: list[str] = Field(default_factory=list)


class MemoryResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    memory_id: str
    content: str
    memory_type: str
    lifetime: str
    visibility: str
    scope_level: str
    owner_principal: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    temporal_status: str
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
    evidence: list[dict[str, Any]]
    category: str | None = None
    created_at: datetime
    updated_at: datetime


class MemoryListResponse(BaseModel):
    memories: list[MemoryResponse]


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
    }


def _service(container: Any) -> MemoryService:
    return container.services["memory"]


@router.post(
    "/observations",
    response_model=ObservationAckResponse,
    status_code=202,
    tags=["memory"],
    summary="Submit an observation (durably acknowledged, processed asynchronously)",
    responses={
        **_WRITE_ERRORS,
        202: {"model": ObservationAckResponse, "description": "Durably acknowledged"},
    },
)
async def submit_observation(
    request: Request, body: ObservationRequest, container: ContainerDep, _: ServicePrincipalDep
) -> JSONResponse:
    ctx = build_context(request, container, body.scope)
    key = request.state.idempotency_key or default_idempotency_key(
        ctx, "observation", body.kind.value, body.content
    )
    payload = body.model_dump(mode="json")

    async def handler(uow):  # type: ignore[no-untyped-def]
        ack = await _service(container).submit_observation(
            uow,
            ctx,
            kind=body.kind,
            content=body.content,
            hints=ProcessingHints(**body.hints.model_dump()),
            custom_metadata=body.custom_metadata,
            occurred_at=body.occurred_at,
            source_system=body.source_system,
            source_id=body.source_id,
            tool_run_id=body.tool_run_id,
        )
        return 202, {"observation_id": ack.observation_id, "job_ids": ack.job_ids}, None

    return await run_idempotent(request, container, ctx, key=key, payload=payload, handler=handler)


@router.get(
    "/memories",
    response_model=MemoryListResponse,
    tags=["memory"],
    summary="List current memories anchored to the caller's scopes",
    responses=_READ_ERRORS,
)
async def list_memories(
    request: Request,
    container: ContainerDep,
    _: ServicePrincipalDep,
    memory_type: Annotated[list[str] | None, Query()] = None,
    include_superseded: bool = False,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    thread_id: str | None = None,
    work_id: str | None = None,
    agent_id: str | None = None,
    agent_group_id: str | None = None,
) -> MemoryListResponse:
    """Security fields come from trusted headers; lineage anchors (thread, work, agent) are
    query parameters so a caller can list the memories of a specific thread or agent."""
    ctx = build_context(
        request,
        container,
        ScopeBody(
            thread_id=thread_id, work_id=work_id, agent_id=agent_id, agent_group_id=agent_group_id
        ),
    )
    async with container.services["uow_factory"]() as uow:
        rows = await _service(container).list_memories(
            uow, ctx, memory_types=memory_type, include_superseded=include_superseded, limit=limit
        )
    return MemoryListResponse(memories=[MemoryResponse(**memory_to_api(m)) for m in rows])


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
async def forget_memory(memory_id: str, ctx: HeaderContextDep, container: ContainerDep) -> None:
    async with container.services["uow_factory"]() as uow:
        await _service(container).forget(uow, ctx, memory_id)
        await uow.commit()
