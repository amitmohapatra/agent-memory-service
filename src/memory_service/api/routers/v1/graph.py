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
from memory_service.api.params import EntityIdPath, limit_query
from memory_service.config.constants import GRAPH
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.graph import GraphLayer, RelationStatus
from memory_service.domain.instants import UTC_RULE, UtcDateTime
from memory_service.modules.graph.service import GraphService
from memory_service.ports.intelligence import Entity, Relation

router = APIRouter()
_ERRORS = error_responses(401, 403, 422, 503)
_READ_ERRORS = error_responses(401, 403, 404, 422, 503)


class EntityOut(BaseModel):
    entity_id: str = Field(description="The entity's id (ent_...).")
    name: str = Field(description="The entity's display name, as most often written.")
    canonical_name: str = Field(
        description="The normalised name the entity is matched by (case and punctuation folded)."
    )
    entity_type: str = Field(
        description="The entity's type, e.g. ORG, PERSON, PRODUCT, MONEY, DATE."
    )
    mention_count: int = Field(
        description="How many times it was mentioned in what the caller may read."
    )
    aliases: list[str] = Field(default_factory=list, description="Other names it was written as.")
    summary: str = Field(
        default="",
        description="one paragraph over the entity's current facts; empty until "
        "the enrichment job has written it",
    )


class FactOut(BaseModel):
    relation_id: str = Field(description="The fact's id (rel_...).")
    subject: str = Field(description="The entity the fact is about (its name).")
    predicate: str = Field(description="The relation, e.g. employs, reported, replaced_by.")
    object: str = Field(description="The entity or value it relates to (a name, or a literal).")
    fact_text: str = Field(description="The fact as one sentence.")
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
    valid_from: datetime | None = Field(
        default=None, description="When it became true (valid time), when known."
    )
    valid_to: datetime | None = Field(
        default=None, description="When it stopped being true (valid time), when known."
    )
    observed_at: datetime = Field(description="When the service learned it (knowledge time).")
    confidence: float = Field(description="0..1, how far the extraction trusts it.")
    memory_id: str | None = Field(
        default=None, description="The memory it was extracted from, if any."
    )
    document_id: str | None = Field(
        default=None, description="The document it was extracted from, if any."
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured fact data: period, currency, amount, change, table, page ...",
    )
    evidence: list[EvidenceRef] = Field(
        default_factory=list, description="Where it came from (at most three references)."
    )


class NeighborhoodOut(BaseModel):
    entities: list[EntityOut] = Field(description="The entities reached, the start included.")
    facts: list[FactOut] = Field(description="The facts traversed between them.")
    visited: int = Field(
        description="How many entities the traversal visited (a bound on its cost)."
    )


class EntityListResponse(BaseModel):
    entities: list[EntityOut] = Field(
        description="The page: entities the text names first, then those whose name starts "
        "with it, most mentioned first."
    )
    next_cursor: str | None = Field(
        default=None,
        description="Pass as `cursor` for the next page (also in `Link`); null on the last.",
    )


class CurrentValueOut(BaseModel):
    predicate: str = Field(description="The relation, e.g. ceo or headquarters.")
    value: str = Field(description="Its newest current value (an entity's name, or a literal).")
    relation_id: str = Field(description="The fact that holds the value (rel_...).")
    valid_from: datetime | None = Field(
        default=None, description="When the value became true, when known."
    )
    observed_at: datetime = Field(description="When the service learned it.")


class EntityProfileResponse(BaseModel):
    entity: EntityOut = Field(description="The entity itself.")
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
    entity_id: EntityIdPath,
    ctx: HeaderContextDep,
    container: ContainerDep,
    depth: Annotated[
        int, Query(ge=0, le=3, description="hops of traversal from the entity; 0: none")
    ] = 0,
    as_of: Annotated[
        UtcDateTime | None,
        Query(description=f"Valid time: facts that were true at this instant. {UTC_RULE}"),
    ] = None,
    valid_at: Annotated[
        UtcDateTime | None,
        Query(
            description="Knowledge time: facts asserted by this instant, not yet withdrawn. "
            + UTC_RULE
        ),
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
