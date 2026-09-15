"""PostgreSQL GraphStore: entities + temporal relations with audience-key filtering and a
bounded, hop-by-hop traversal (one indexed query per hop, capped by ``max_visited``).

The graph is derived state (rebuildable from memories and chunks) but it lives next to the
canonical rows so that a single database backup restores everything. It uses its own
sessions rather than the caller's unit of work: enrichment runs inside index jobs and is
idempotent, so a partial write is repaired by the next run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import Text, delete, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import array, insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from memory_service.adapters.db.orm import GraphEntityRow, GraphRelationRow
from memory_service.domain.evidence import EvidenceRef
from memory_service.observability.metrics import stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.intelligence import Entity, GraphNeighborhood, Relation


def _keys_clause(column: Any, keys: Sequence[str]) -> Any:
    """``visibility_keys ?| ARRAY[...]`` — any-of match on the JSONB string array."""
    return column.op("?|")(array([str(k) for k in keys], type_=Text))


def _entity(r: GraphEntityRow) -> Entity:
    return Entity(
        entity_id=r.entity_id,
        tenant_id=r.tenant_id,
        name=r.name,
        canonical_name=r.canonical_name,
        entity_type=r.entity_type,
        aliases=list(r.aliases or []),
        scope_key=r.scope_key,
        visibility_keys=list(r.visibility_keys or []),
        evidence=[EvidenceRef.model_validate(e) for e in (r.evidence or [])],
        mention_count=r.mention_count,
        revision=r.revision,
    )


def _relation(r: GraphRelationRow) -> Relation:
    return Relation(
        relation_id=r.relation_id,
        tenant_id=r.tenant_id,
        subject_id=r.subject_id,
        predicate=r.predicate,
        object_id=r.object_id,
        scope_key=r.scope_key,
        visibility_keys=list(r.visibility_keys or []),
        valid_from=r.valid_from,
        valid_to=r.valid_to,
        observed_at=r.observed_at,
        status=r.status,
        superseded_by=r.superseded_by,
        confidence=r.confidence,
        evidence=[EvidenceRef.model_validate(e) for e in (r.evidence or [])],
        memory_id=r.memory_id,
        document_id=r.document_id,
        fact_text=r.fact_text or "",
        attributes=dict(r.attributes or {}),
    )


def _ev(evidence: Sequence[EvidenceRef]) -> list[dict[str, Any]]:
    return [json.loads(e.model_dump_json(exclude_none=True)) for e in evidence]


class PostgresGraphStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)

    def session(self) -> AsyncSession:
        return self._sessions()

    # -- writes ---------------------------------------------------------------------
    async def upsert_entities(self, entities: Sequence[Entity]) -> None:
        if not entities:
            return
        async with self.session() as s, s.begin():
            # merge audiences/aliases in Python: an entity seen in a wider scope becomes wider
            names = {(e.tenant_id, e.scope_key, e.canonical_name) for e in entities}
            existing = (
                await s.scalars(
                    select(GraphEntityRow).where(
                        GraphEntityRow.tenant_id.in_({t for t, _, _ in names}),
                        GraphEntityRow.canonical_name.in_({n for _, _, n in names}),
                    )
                )
            ).all()
            by_key = {(r.tenant_id, r.scope_key, r.canonical_name): r for r in existing}
            for e in entities:
                prev = by_key.get((e.tenant_id, e.scope_key, e.canonical_name))
                keys = sorted(
                    set(e.visibility_keys) | set(prev.visibility_keys or [] if prev else [])
                )
                aliases = sorted(set(e.aliases) | set(prev.aliases or [] if prev else []))[:20]
                stmt = insert(GraphEntityRow).values(
                    entity_id=e.entity_id,
                    tenant_id=e.tenant_id,
                    scope_key=e.scope_key,
                    canonical_name=e.canonical_name,
                    name=e.name,
                    entity_type=e.entity_type,
                    aliases=aliases,
                    visibility_keys=keys,
                    evidence=_ev(e.evidence[:20]),
                    mention_count=e.mention_count,
                )
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_graph_entity",
                    set_={
                        "mention_count": GraphEntityRow.mention_count + 1,
                        "aliases": aliases,
                        "visibility_keys": keys,
                        "updated_at": func.now(),
                        "revision": GraphEntityRow.revision + 1,
                    },
                )
                await s.execute(stmt)

    async def upsert_relations(self, relations: Sequence[Relation]) -> None:
        if not relations:
            return
        async with self.session() as s, s.begin():
            for r in relations:
                stmt = insert(GraphRelationRow).values(
                    relation_id=r.relation_id,
                    tenant_id=r.tenant_id,
                    scope_key=r.scope_key,
                    subject_id=r.subject_id,
                    predicate=r.predicate,
                    object_id=r.object_id,
                    visibility_keys=list(r.visibility_keys),
                    valid_from=r.valid_from,
                    valid_to=r.valid_to,
                    observed_at=r.observed_at,
                    status=r.status,
                    superseded_by=r.superseded_by,
                    confidence=r.confidence,
                    evidence=_ev(r.evidence[:20]),
                    memory_id=r.memory_id,
                    document_id=r.document_id,
                    fact_text=r.fact_text,
                    attributes=r.attributes,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[GraphRelationRow.relation_id],
                    set_={
                        "confidence": func.greatest(
                            GraphRelationRow.confidence, stmt.excluded.confidence
                        ),
                        "status": stmt.excluded.status,
                        "valid_to": stmt.excluded.valid_to,
                        "superseded_by": stmt.excluded.superseded_by,
                        "visibility_keys": stmt.excluded.visibility_keys,
                        "fact_text": stmt.excluded.fact_text,
                        "updated_at": func.now(),
                    },
                )
                await s.execute(stmt)

    async def supersede(self, relation_id: str, *, by: str, at: datetime) -> None:
        async with self.session() as s, s.begin():
            await s.execute(
                update(GraphRelationRow)
                .where(GraphRelationRow.relation_id == relation_id)
                .values(status="SUPERSEDED", superseded_by=by, valid_to=at, updated_at=at)
            )

    async def supersede_for_memory(self, tenant_id: str, memory_id: str, *, at: datetime) -> int:
        async with self.session() as s, s.begin():
            res = await s.execute(
                update(GraphRelationRow)
                .where(
                    GraphRelationRow.tenant_id == tenant_id,
                    GraphRelationRow.memory_id == memory_id,
                    GraphRelationRow.status == "CURRENT",
                )
                .values(status="SUPERSEDED", valid_to=at, updated_at=at)
                .returning(GraphRelationRow.relation_id)
            )
            return len(res.scalars().all())

    async def delete_for_document(self, tenant_id: str, document_id: str) -> int:
        async with self.session() as s, s.begin():
            res = await s.execute(
                delete(GraphRelationRow)
                .where(
                    GraphRelationRow.tenant_id == tenant_id,
                    GraphRelationRow.document_id == document_id,
                )
                .returning(GraphRelationRow.relation_id)
            )
            return len(res.scalars().all())

    # -- reads ----------------------------------------------------------------------
    async def find_entities(
        self, tenant_id: str, names: Sequence[str], *, scope_keys: Sequence[str]
    ) -> list[Entity]:
        if not names or not scope_keys:
            return []
        wanted = [n for n in names if n]
        async with self.session() as s:
            rows = (
                await s.scalars(
                    select(GraphEntityRow).where(
                        GraphEntityRow.tenant_id == tenant_id,
                        or_(
                            GraphEntityRow.canonical_name.in_(wanted),
                            # aliases hold canonical (lower-cased) forms: "arr", "acme"
                            GraphEntityRow.aliases.op("?|")(array(wanted, type_=Text)),
                        ),
                        _keys_clause(GraphEntityRow.visibility_keys, scope_keys),
                    )
                )
            ).all()
        return [_entity(r) for r in rows]

    async def get_entities(
        self, tenant_id: str, entity_ids: Sequence[str], *, scope_keys: Sequence[str]
    ) -> list[Entity]:
        if not entity_ids or not scope_keys:
            return []
        async with self.session() as s:
            rows = (
                await s.scalars(
                    select(GraphEntityRow).where(
                        GraphEntityRow.tenant_id == tenant_id,
                        GraphEntityRow.entity_id.in_(list(entity_ids)),
                        _keys_clause(GraphEntityRow.visibility_keys, scope_keys),
                    )
                )
            ).all()
        return [_entity(r) for r in rows]

    async def neighborhood(
        self,
        tenant_id: str,
        entity_ids: Sequence[str],
        *,
        scope_keys: Sequence[str],
        hops: int = 1,
        max_visited: int = 200,
        as_of: datetime | None = None,
    ) -> GraphNeighborhood:
        if not entity_ids or not scope_keys:
            return GraphNeighborhood(entities=[], relations=[], visited=0)
        visited: dict[str, None] = dict.fromkeys(entity_ids)
        frontier = list(entity_ids)
        relations: dict[str, Relation] = {}
        with span("graph.neighborhood", hops=hops), stage_seconds.labels("graph.traverse").time():
            async with self.session() as s:
                for _ in range(max(0, hops)):
                    if not frontier or len(visited) >= max_visited:
                        break
                    conds = [
                        GraphRelationRow.tenant_id == tenant_id,
                        or_(
                            GraphRelationRow.subject_id.in_(frontier),
                            GraphRelationRow.object_id.in_(frontier),
                        ),
                        _keys_clause(GraphRelationRow.visibility_keys, scope_keys),
                    ]
                    if as_of is None:
                        conds.append(GraphRelationRow.status == "CURRENT")
                    else:
                        conds.append(
                            or_(
                                GraphRelationRow.valid_from.is_(None),
                                GraphRelationRow.valid_from <= as_of,
                            )
                        )
                        conds.append(
                            or_(
                                GraphRelationRow.valid_to.is_(None),
                                GraphRelationRow.valid_to > as_of,
                            )
                        )
                        conds.append(GraphRelationRow.status != "RETRACTED")
                    rows = (
                        await s.scalars(
                            select(GraphRelationRow)
                            .where(*conds)
                            .order_by(GraphRelationRow.confidence.desc())
                            .limit(max_visited * 3)
                        )
                    ).all()
                    next_frontier: list[str] = []
                    for r in rows:
                        relations.setdefault(r.relation_id, _relation(r))
                        for eid in (r.subject_id, r.object_id):
                            if eid not in visited and len(visited) < max_visited:
                                visited[eid] = None
                                next_frontier.append(eid)
                    frontier = next_frontier
                ents = (
                    await s.scalars(
                        select(GraphEntityRow).where(
                            GraphEntityRow.tenant_id == tenant_id,
                            GraphEntityRow.entity_id.in_(list(visited)),
                            _keys_clause(GraphEntityRow.visibility_keys, scope_keys),
                        )
                    )
                ).all()
        allowed = {e.entity_id for e in ents}
        rels = [r for r in relations.values() if r.subject_id in allowed and r.object_id in allowed]
        return GraphNeighborhood(
            entities=[_entity(e) for e in ents], relations=rels, visited=len(visited)
        )

    async def relations_for_document(
        self, tenant_id: str, document_id: str, *, scope_keys: Sequence[str]
    ) -> list[Relation]:
        if not scope_keys:
            return []
        async with self.session() as s:
            rows = (
                await s.scalars(
                    select(GraphRelationRow).where(
                        GraphRelationRow.tenant_id == tenant_id,
                        GraphRelationRow.document_id == document_id,
                        _keys_clause(GraphRelationRow.visibility_keys, scope_keys),
                    )
                )
            ).all()
        return [_relation(r) for r in rows]

    async def relations_for_memory(self, tenant_id: str, memory_id: str) -> list[Relation]:
        async with self.session() as s:
            rows = (
                await s.scalars(
                    select(GraphRelationRow).where(
                        GraphRelationRow.tenant_id == tenant_id,
                        GraphRelationRow.memory_id == memory_id,
                    )
                )
            ).all()
        return [_relation(r) for r in rows]

    async def count(self, tenant_id: str) -> tuple[int, int]:
        async with self.session() as s:
            e = await s.scalar(
                select(func.count())
                .select_from(GraphEntityRow)
                .where(GraphEntityRow.tenant_id == tenant_id)
            )
            r = await s.scalar(
                select(func.count())
                .select_from(GraphRelationRow)
                .where(GraphRelationRow.tenant_id == tenant_id)
            )
        return int(e or 0), int(r or 0)

    async def ping(self) -> bool:
        try:
            async with self.session() as s:
                await s.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
