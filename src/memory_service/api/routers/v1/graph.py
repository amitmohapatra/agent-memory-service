"""Public /v1/graph routes: find entities (resolved from free text, or by name), and an
entity's profile with a bounded, visibility-filtered traversal from it."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, cast

from fastapi import APIRouter, Query, Request, Response
from pydantic import BaseModel, Field

from memory_service.api.deps import ContainerDep, HeaderContextDep
from memory_service.api.errors import error_responses
from memory_service.api.pagination import CursorQuery, decode_cursor, encode_cursor, link_next
from memory_service.api.params import limit_query
from memory_service.config.constants import GRAPH
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.graph import GraphLayer, RelationStatus
from memory_service.modules.graph.service import GraphService
from memory_service.ports.intelligence import Entity, Relation

router = APIRouter()
_ERRORS = error_responses(401, 403, 422, 503)
_READ_ERRORS = error_responses(401, 403, 404, 422, 503)


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
        description="CURRENT unless the traversal carries as_of or valid_at, which also return "
        "SUPERSEDED facts that held (or were asserted) at that instant. A traversal never "
        "returns facts that were never right; the profile's history does.",
    )
    layer: GraphLayer = Field(
        ...,
        description="entity (a typed fact), temporal (when it held, which fact replaced "
        "which), causal (why), structural (where it appears in the corpus), procedural (a "
        "tool call used the entity, or returned the id that identifies it)",
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


class NeighborhoodOut(BaseModel):
    entities: list[EntityOut]
    facts: list[FactOut]
    visited: int


class EntityListResponse(BaseModel):
    entities: list[EntityOut]
    next_cursor: str | None = Field(
        default=None,
        description="Pass as `cursor` for the next page (also in `Link`); null on the last.",
    )


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
    neighborhood: NeighborhoodOut | None = Field(
        default=None, description="the traversal from the entity, with depth"
    )


@router.get(
    "/graph/entities",
    response_model=EntityListResponse,
    tags=["graph"],
    summary="Find the entities visible in this scope: those a text names, and by name prefix",
    responses=_ERRORS,
)
async def search_entities(
    request: Request,
    response: Response,
    ctx: HeaderContextDep,
    container: ContainerDep,
    q: Annotated[
        str | None,
        Query(
            max_length=4000,
            description="free text: the entities it names come first, then entities whose "
            "name starts with it (case-insensitive)",
        ),
    ] = None,
    type: Annotated[
        str | None, Query(max_length=40, description="entity type, e.g. ORG or PERSON")
    ] = None,
    cursor: CursorQuery = None,
    limit: Annotated[int, limit_query(GRAPH.entity_search_max, "entities")] = 20,
) -> EntityListResponse:
    """The answer is a ranking (named first, then by prefix, most mentioned first), so the
    cursor is a position in it; the ranking holds at most the service's search bound
    (100 entities) in all - narrow ``q`` or ``type`` to reach past it."""
    position = decode_cursor(cursor, fields={"offset": int})
    offset = position["offset"] if position else 0
    graph: GraphService = container.services["graph"]
    async with container.services["llm_assist"].reading(ctx):
        found = await graph.search_entities(
            ctx, query=q, entity_type=type, limit=offset + limit + 1
        )
    items = found[offset : offset + limit]
    more = len(found) > offset + limit and len(items) == limit
    next_cursor = encode_cursor({"offset": offset + limit}) if more else None
    link_next(request, response, next_cursor)
    return EntityListResponse(entities=[_entity(e) for e in items], next_cursor=next_cursor)


@router.get(
    "/graph/entities/{entity_id}",
    response_model=EntityProfileResponse,
    tags=["graph"],
    summary="An entity's profile (current value per predicate, relations, history, evidence) "
    "and, with depth, the graph around it",
    responses=_READ_ERRORS,
)
async def entity_profile(
    entity_id: str,
    ctx: HeaderContextDep,
    container: ContainerDep,
    depth: Annotated[
        int, Query(ge=0, le=3, description="hops of traversal from the entity; 0: none")
    ] = 0,
    as_of: Annotated[
        datetime | None,
        Query(description="valid time: facts that were true at this instant"),
    ] = None,
    valid_at: Annotated[
        datetime | None,
        Query(description="knowledge time: facts asserted by this instant, not yet withdrawn"),
    ] = None,
    layers: Annotated[
        list[GraphLayer] | None,
        Query(
            description="traverse only these layers: entity, temporal, causal, structural, "
            "procedural; omit for every layer"
        ),
    ] = None,
) -> EntityProfileResponse:
    graph: GraphService = container.services["graph"]
    profile = await graph.profile(
        ctx, entity_id, depth=depth, as_of=as_of, valid_at=valid_at, layers=layers
    )
    hood = profile.neighborhood
    names = {
        **profile.names,
        **({e.entity_id: e.name for e in hood.entities} if hood else {}),
        profile.entity.entity_id: profile.entity.name,
    }
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
        neighborhood=NeighborhoodOut(
            entities=[_entity(e) for e in hood.entities],
            facts=[_fact(r, names) for r in hood.relations],
            visited=hood.visited,
        )
        if hood is not None
        else None,
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
