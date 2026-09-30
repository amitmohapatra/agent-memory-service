"""Public /v1/graph routes: entity resolution, bounded visibility-filtered traversal, and
entity search and profiles."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, cast

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from memory_service.api.deps import (
    ContainerDep,
    HeaderContextDep,
    ScopeBody,
    ServicePrincipalDep,
    build_context,
)
from memory_service.api.errors import error_responses
from memory_service.api.validation import UseLLM
from memory_service.config.constants import GRAPH
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.graph import GraphLayer, RelationStatus
from memory_service.modules.graph.service import GraphService
from memory_service.ports.intelligence import Entity, Relation

router = APIRouter()
_ERRORS = error_responses(401, 403, 422, 503)
_READ_ERRORS = error_responses(401, 403, 404, 422, 503)

_EXAMPLE: dict[str, Any] = {
    "scope": {"thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH"},
    "query": "Why did Adjusted EBITDA increase despite lower revenue?",
    "hops": 2,
}


class GraphQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [_EXAMPLE]})

    scope: ScopeBody = Field(default_factory=ScopeBody)
    use_llm: UseLLM = None
    query: str | None = Field(
        default=None, max_length=4000, description="free text; entities are resolved from it"
    )
    entities: list[str] = Field(default_factory=list, description="explicit entity names")
    hops: int = Field(default=1, ge=1, le=3)
    as_of: datetime | None = Field(
        default=None,
        description="valid time: facts that were true at this instant (including superseded "
        "facts that held then)",
    )
    valid_at: datetime | None = Field(
        default=None,
        description="knowledge time: facts that had been asserted by this instant and not yet "
        "invalidated",
    )
    layers: list[GraphLayer] | None = Field(
        default=None,
        min_length=1,
        description="restrict the traversal to these layers: entity (typed facts), temporal, "
        "causal, structural (where things appear in the corpus); omit for every layer",
    )
    max_visited: int | None = Field(
        default=None,
        ge=1,
        le=500,
        description="Cap on entities the traversal may visit; omit for the server default",
    )


class EntityOut(BaseModel):
    entity_id: str
    name: str
    canonical_name: str
    entity_type: str
    mention_count: int
    aliases: list[str] = Field(default_factory=list)
    summary: str = Field(
        default="",
        description="one paragraph over the entity's current facts; empty until "
        "the enrichment job has written it",
    )


class FactOut(BaseModel):
    relation_id: str
    subject: str
    predicate: str
    object: str
    fact_text: str
    status: RelationStatus = Field(
        ...,
        description="CURRENT unless the query carries as_of or valid_at, which also return "
        "SUPERSEDED facts that held (or were asserted) at that instant. /v1/graph/query never "
        "returns facts that were never right; an entity profile's history does.",
    )
    layer: GraphLayer = Field(
        ...,
        description="entity (a typed fact), temporal (when it held, which fact replaced "
        "which), causal (why), structural (where it appears in the corpus)",
    )
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    observed_at: datetime
    confidence: float
    memory_id: str | None = None
    document_id: str | None = None
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured fact data: period, currency, amount, change, table, page ...",
    )
    evidence: list[EvidenceRef] = Field(default_factory=list)


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
    graph: GraphService = container.services["graph"]
    async with container.services["llm_assist"].reading(ctx, use_llm=body.use_llm):
        answer = await graph.query(
            ctx,
            query=body.query,
            entities=body.entities,
            hops=body.hops,
            as_of=body.as_of,
            valid_at=body.valid_at,
            layers=body.layers,
            max_visited=body.max_visited,
        )
    names = {e.entity_id: e.name for e in answer.entities}
    return GraphQueryResponse(
        matched=[_entity(e) for e in answer.matched],
        entities=[_entity(e) for e in answer.entities],
        facts=[_fact(r, names) for r in answer.relations],
        visited=answer.visited,
    )


class EntityListResponse(BaseModel):
    entities: list[EntityOut]


class CurrentValueOut(BaseModel):
    predicate: str
    value: str
    relation_id: str
    valid_from: datetime | None = None
    observed_at: datetime


class EntityProfileResponse(BaseModel):
    entity: EntityOut
    current: list[CurrentValueOut] = Field(
        description="the newest current value of each predicate the entity is the subject of"
    )
    relations: list[FactOut] = Field(description="current facts with the entity at either end")
    history: list[FactOut] = Field(
        description="facts that stopped holding: superseded, retracted or invalidated"
    )
    evidence: list[EvidenceRef] = Field(description="where the entity was seen")


@router.get(
    "/graph/entities",
    response_model=EntityListResponse,
    tags=["graph"],
    summary="Search the entities visible in this scope (name prefix, type), most mentioned first",
    responses=_ERRORS,
)
async def search_entities(
    ctx: HeaderContextDep,
    container: ContainerDep,
    q: Annotated[
        str | None,
        Query(max_length=300, description="start of the entity name (case-insensitive)"),
    ] = None,
    type: Annotated[
        str | None, Query(max_length=40, description="entity type, e.g. ORG or PERSON")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=GRAPH.entity_search_max)] = 20,
) -> EntityListResponse:
    graph: GraphService = container.services["graph"]
    found = await graph.search_entities(ctx, query=q, entity_type=type, limit=limit)
    return EntityListResponse(entities=[_entity(e) for e in found])


@router.get(
    "/graph/entities/{entity_id}",
    response_model=EntityProfileResponse,
    tags=["graph"],
    summary="An entity's profile: current value per predicate, relations, history, evidence",
    responses=_READ_ERRORS,
)
async def entity_profile(
    entity_id: str, ctx: HeaderContextDep, container: ContainerDep
) -> EntityProfileResponse:
    graph: GraphService = container.services["graph"]
    profile = await graph.profile(ctx, entity_id)
    names = {**profile.names, profile.entity.entity_id: profile.entity.name}
    return EntityProfileResponse(
        entity=_entity(profile.entity),
        current=[
            CurrentValueOut(
                predicate=r.predicate,
                value=names.get(r.object_id, r.object_id),
                relation_id=r.relation_id,
                valid_from=r.valid_from,
                observed_at=r.observed_at,
            )
            for r in profile.current
        ],
        relations=[_fact(r, names) for r in profile.relations],
        history=[_fact(r, names) for r in profile.history],
        evidence=list(profile.entity.evidence),
    )


def _entity(e: Entity) -> EntityOut:
    return EntityOut(
        entity_id=e.entity_id,
        name=e.name,
        canonical_name=e.canonical_name,
        entity_type=e.entity_type,
        mention_count=e.mention_count,
        aliases=list(e.aliases),
        summary=e.summary,
    )


def _fact(r: Relation, names: dict[str, str]) -> FactOut:
    return FactOut(
        relation_id=r.relation_id,
        subject=names.get(r.subject_id, r.subject_id),
        predicate=r.predicate,
        object=names.get(r.object_id, r.object_id),
        fact_text=r.fact_text,
        status=cast(RelationStatus, r.status),
        layer=r.layer,
        valid_from=r.valid_from,
        valid_to=r.valid_to,
        observed_at=r.observed_at,
        confidence=r.confidence,
        memory_id=r.memory_id,
        document_id=r.document_id,
        attributes=dict(r.attributes),
        evidence=list(r.evidence[:3]),
    )
