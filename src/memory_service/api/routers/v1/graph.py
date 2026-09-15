"""Public /v1/graph routes: entity resolution and bounded, visibility-filtered traversal."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from memory_service.api.deps import ContainerDep, ScopeBody, ServicePrincipalDep, build_context
from memory_service.api.errors import error_responses

router = APIRouter()
_ERRORS = error_responses(401, 403, 422, 503)

_EXAMPLE: dict[str, Any] = {
    "scope": {"thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH"},
    "query": "Why did Adjusted EBITDA increase despite lower revenue?",
    "hops": 2,
}


class GraphQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [_EXAMPLE]})

    scope: ScopeBody = Field(default_factory=ScopeBody)
    query: str | None = Field(
        default=None, max_length=4000, description="free text; entities are resolved from it"
    )
    entities: list[str] = Field(default_factory=list, description="explicit entity names")
    hops: int = Field(default=1, ge=1, le=3)
    as_of: datetime | None = Field(
        default=None, description="temporal view: facts valid at this instant"
    )
    max_visited: int | None = Field(default=None, ge=1, le=2000)


class EntityOut(BaseModel):
    entity_id: str
    name: str
    canonical_name: str
    entity_type: str
    mention_count: int


class FactOut(BaseModel):
    relation_id: str
    subject: str
    predicate: str
    object: str
    fact_text: str
    status: str
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    observed_at: datetime
    confidence: float
    memory_id: str | None = None
    document_id: str | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class GraphQueryResponse(BaseModel):
    matched: list[EntityOut]
    entities: list[EntityOut]
    facts: list[FactOut]
    visited: int


@router.post(
    "/graph/query",
    response_model=GraphQueryResponse,
    tags=["graph"],
    summary="Resolve entities and traverse the knowledge graph (bounded, visibility-filtered)",
    responses=_ERRORS,
)
async def graph_query(
    request: Request, body: GraphQueryRequest, container: ContainerDep, _: ServicePrincipalDep
) -> GraphQueryResponse:
    ctx = build_context(request, container, body.scope)
    graph = container.services["graph"]
    answer = await graph.query(
        ctx,
        query=body.query,
        entities=body.entities,
        hops=body.hops,
        as_of=body.as_of,
        max_visited=body.max_visited,
    )
    names = {e.entity_id: e.name for e in answer.entities}

    def ent(e: Any) -> EntityOut:
        return EntityOut(
            entity_id=e.entity_id,
            name=e.name,
            canonical_name=e.canonical_name,
            entity_type=e.entity_type,
            mention_count=e.mention_count,
        )

    return GraphQueryResponse(
        matched=[ent(e) for e in answer.matched],
        entities=[ent(e) for e in answer.entities],
        facts=[
            FactOut(
                relation_id=r.relation_id,
                subject=names.get(r.subject_id, r.subject_id),
                predicate=r.predicate,
                object=names.get(r.object_id, r.object_id),
                fact_text=r.fact_text,
                status=r.status,
                valid_from=r.valid_from,
                valid_to=r.valid_to,
                observed_at=r.observed_at,
                confidence=r.confidence,
                memory_id=r.memory_id,
                document_id=r.document_id,
                evidence=[e.model_dump(mode="json", exclude_none=True) for e in r.evidence[:3]],
            )
            for r in answer.relations
        ],
        visited=answer.visited,
    )
