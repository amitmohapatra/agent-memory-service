"""PostgreSQL GraphStore: entities + temporal relations with audience-key filtering and a
bounded, hop-by-hop traversal (one indexed query per hop, capped by ``max_visited``).

Reads run on autocommit connections: a read-only statement needs no transaction, and a
BEGIN before it and a ROLLBACK after it are two round trips for nothing.

The retrieval-time traversal (``budgeted``) is one statement on a small pool of its own whose
connections carry the graph budget as their ``statement_timeout``: the server stops a
traversal that outruns it, which bounds the query path without a wall clock in the client.
A client-side timer measured the client's scheduling as much as the graph - on a loaded box
it dropped the graph from answers whose traversal had taken 90 ms of a 500 ms wait - and it
could only stop waiting, not stop the statement, which then held a pooled connection.

The graph is derived state (rebuildable from memories and chunks) but it lives next to the
canonical rows so that a single database backup restores everything. It uses its own
sessions rather than the caller's unit of work: enrichment runs inside index jobs and is
idempotent, so a partial write is repaired by the next run.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from psycopg.errors import QueryCanceled
from sqlalchemy import (
    BindParameter,
    DateTime,
    Integer,
    Select,
    Text,
    and_,
    bindparam,
    case,
    delete,
    exists,
    func,
    literal,
    or_,
    select,
    text,
    union_all,
    update,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, array, insert
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import aliased

from memory_service.adapters.db.engine import track_pool
from memory_service.adapters.db.orm import GraphEntityRow, GraphRelationRow
from memory_service.config.constants import DATABASE, GRAPH
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.graph import INVALIDATED_BY, GraphLayer
from memory_service.modules.graph.invalidation import invalidation_edge
from memory_service.observability.metrics import stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.intelligence import (
    Entity,
    EntityFacts,
    GraphBudgetExceededError,
    GraphNeighborhood,
    Relation,
)


def _text_array(values: Iterable[str] | BindParameter[Any]) -> Any:
    """One ``text[]`` parameter, whatever its length: the statement's text does not change
    with the number of keys, so a prepared plan is reused. A bind parameter passes through
    (the cached traversal statement)."""
    if isinstance(values, BindParameter):
        return values
    return literal([str(v) for v in values], type_=ARRAY(Text))


def _keys_clause(column: Any, keys: Sequence[str] | BindParameter[Any]) -> Any:
    """``visibility_keys ?| $keys`` — any-of match on the JSONB string array."""
    return column.op("?|")(_text_array(keys))


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
        summary=r.summary or "",
    )


def _relation(r: GraphRelationRow) -> Relation:
    return Relation(
        relation_id=r.relation_id,
        tenant_id=r.tenant_id,
        subject_id=r.subject_id,
        predicate=r.predicate,
        object_id=r.object_id,
        layer=r.layer,  # type: ignore[arg-type]
        scope_key=r.scope_key,
        visibility_keys=list(r.visibility_keys or []),
        valid_from=r.valid_from,
        valid_to=r.valid_to,
        observed_at=r.observed_at,
        invalidated_at=r.invalidated_at,
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


def _time_conditions(
    as_of: datetime | BindParameter[Any] | None, valid_at: datetime | BindParameter[Any] | None
) -> list[Any]:
    """See :func:`memory_service.modules.graph.invalidation.passes_time`."""
    if as_of is None and valid_at is None:
        return [GraphRelationRow.status == "CURRENT"]
    conds: list[Any] = []
    if as_of is not None:
        # valid-time semantics: an undated fact is taken to have held before it was learned
        # (a job, a preference), so ``as_of`` inside a superseded interval returns the old
        # value, not the current one; a fact that was never right is never returned
        conds.append(
            or_(GraphRelationRow.valid_from.is_(None), GraphRelationRow.valid_from <= as_of)
        )
        conds.append(or_(GraphRelationRow.valid_to.is_(None), GraphRelationRow.valid_to > as_of))
        conds.append(GraphRelationRow.status.not_in(["RETRACTED", "INVALIDATED"]))
    if valid_at is not None:
        conds.append(GraphRelationRow.observed_at <= valid_at)
        conds.append(
            or_(
                GraphRelationRow.invalidated_at.is_(None),
                GraphRelationRow.invalidated_at > valid_at,
            )
        )
    return conds


# A triple alone is not an assertion: the same value may apply in different years,
# documents or qualified contexts. SQL and cross-hop deduplication share this identity.
_ASSERTION_FIELDS = (
    "subject_id",
    "predicate",
    "object_id",
    "document_id",
    "layer",
    "valid_from",
    "valid_to",
)
IdSource = Sequence[str] | Select[Any]


def _ids(source: IdSource) -> Any:
    """Entity ids as a bound list or as a subquery of an earlier hop (for ``IN``)."""
    return source if isinstance(source, Select) else list(source)


#: a value of a hop, or the bind parameter the cached traversal fills it from
Param = BindParameter[Any]


def neighborhood_query(
    tenant_id: str | Param,
    frontier: IdSource,
    *,
    scope_keys: Sequence[str] | Param,
    layers: Sequence[GraphLayer] | Param | None,
    as_of: datetime | Param | None,
    valid_at: datetime | Param | None,
    limit: int | Any,
    expanded: IdSource = (),
    earlier: Sequence[Any] = (),
) -> Select[Any]:
    """One hop of the traversal: the best new edges out of ``frontier``, in a total order,
    each with its ``rank`` in that order.

    Everything that would be dropped after the query is dropped *before* its limit, so the
    limit is spent only on edges the traversal keeps: an edge back to an entity an earlier
    hop already expanded (``expanded``), an assertion an earlier hop already returned
    (``earlier``, those hops' CTEs), and an edge to a neighbour the caller cannot see (the
    join on the neighbour's audience). Filtering any of them after the LIMIT let them fill
    it and starve the edges that were new.

    ORDER BY confidence alone is not an order here. Every MENTIONS edge is written at the
    same capped confidence, so over a busy entity the LIMIT returned whichever of the tied
    rows the scan reached first and the same question could be answered from a different
    sample between runs. The neighbour's mention_count breaks the tie towards the entity
    the corpus actually talks about, recency breaks what that leaves, and the relation id
    makes the order total and reproducible.
    """
    ends = _ids(frontier)
    neighbour = case(
        (GraphRelationRow.subject_id.in_(ends), GraphRelationRow.object_id),
        else_=GraphRelationRow.subject_id,
    )
    conds = [
        GraphRelationRow.tenant_id == tenant_id,
        or_(GraphRelationRow.subject_id.in_(ends), GraphRelationRow.object_id.in_(ends)),
        _keys_clause(GraphRelationRow.visibility_keys, scope_keys),
        *_time_conditions(as_of, valid_at),
    ]
    if isinstance(layers, BindParameter) or layers:
        conds.append(GraphRelationRow.layer == func.any(_text_array(layers)))
    if isinstance(expanded, Select) or expanded:
        conds.append(neighbour.not_in(_ids(expanded)))
    for hop in earlier:
        conds.append(
            ~exists().where(
                *(
                    hop.c[field].is_not_distinct_from(getattr(GraphRelationRow, field))
                    for field in _ASSERTION_FIELDS
                ),
                hop.c.attributes == GraphRelationRow.attributes,
            )
        )
    # Deduplicate in SQL, before the limit - not in Python after it. A triple is written
    # once per memory that states it, and every 'mentions' edge carries the same capped
    # confidence, so the tie-break falls to the neighbour's mention_count and the busiest
    # entity's duplicates fill the limit: measured on the benchmark corpus, 600 rows
    # collapsed to 72 distinct triples against 125 reachable, one triple alone holding 178
    # of the 600 slots. The limit was truncating exactly the diverse tail it protects.
    #
    # Two orderings, and they are different on purpose. The inner one picks WHICH row
    # survives for each assertion (confidence, neighbour mentions, then recency) and must
    # start with the DISTINCT ON columns because PostgreSQL requires it. The outer one
    # decides WHICH ASSERTIONS the limit keeps, by the same ranking the caller expects,
    # so the limit takes the best assertions rather than the alphabetically first ones.
    ranked = func.coalesce(GraphEntityRow.mention_count, 0).label("neighbour_mentions")
    identity = (
        *(getattr(GraphRelationRow, field) for field in _ASSERTION_FIELDS),
        GraphRelationRow.attributes,
    )
    inner = (
        select(GraphRelationRow, ranked)
        .join(
            GraphEntityRow,
            and_(
                GraphEntityRow.entity_id == neighbour,
                _keys_clause(GraphEntityRow.visibility_keys, scope_keys),
            ),
        )
        .where(*conds)
        .distinct(*identity)
        .order_by(
            *identity,
            GraphRelationRow.confidence.desc(),
            ranked.desc(),
            GraphRelationRow.observed_at.desc(),
            GraphRelationRow.relation_id,
        )
        .subquery()
    )
    order = (
        inner.c.confidence.desc(),
        inner.c.neighbour_mentions.desc(),
        inner.c.observed_at.desc(),
        inner.c.relation_id,
    )
    rank = func.row_number().over(order_by=order).label("rank")
    return select(inner, rank).order_by(*order).limit(limit)


@functools.cache
def traversal_query(
    hops: int, *, layers: bool = False, as_of: bool = False, valid_at: bool = False
) -> Select[Any]:
    """The whole bounded traversal as one statement: ``hops`` chained hop CTEs.

    Per hop ``k``: the frontier ``f_k`` (the seeds, then the entities hop ``k-1`` reached
    first, in rank order, while fewer than ``max_visited`` are visited), the hop's best
    ``3 * max_visited`` new assertions (``neighborhood_query``), and the visited set
    ``v_k``. A hop runs only while the visited set is below the cap, as the loop it
    replaces did. The rows are every hop's assertions whose two ends were both visited,
    with both ends' entity rows and the visited count, in (hop, rank) order.

    One round trip instead of one per hop plus a trailing read of the names. Built once
    per shape (hop count, and whether layers, ``as_of`` and ``valid_at`` constrain it) with
    every value a bind parameter (``traversal_params``): constructing it cost 34 ms of
    Python per query on the dev box, and its text is what the server prepares.
    """
    tenant_id: Any = bindparam("tenant_id", type_=Text)
    seeds = bindparam("seeds", type_=ARRAY(Text))
    scope_keys = bindparam("keys", type_=ARRAY(Text))
    max_visited: Any = bindparam("max_visited", type_=Integer)
    visited = select(func.unnest(_text_array(seeds)).label("entity_id")).cte("v1")
    frontier = visited
    steps_visited = visited
    limit = max_visited * 3
    steps: list[Any] = []
    for k in range(1, max(0, hops) + 1):
        count = select(func.count()).select_from(visited).scalar_subquery()
        step = (
            neighborhood_query(
                tenant_id,
                select(frontier.c.entity_id),
                scope_keys=scope_keys,
                layers=bindparam("layers", type_=ARRAY(Text)) if layers else None,
                as_of=bindparam("as_of", type_=DateTime(timezone=True)) if as_of else None,
                valid_at=(
                    bindparam("valid_at", type_=DateTime(timezone=True)) if valid_at else None
                ),
                limit=limit,
                expanded=select(steps_visited.c.entity_id) if k > 1 else (),
                earlier=steps,
            )
            .where(count < max_visited)
            .cte(f"h{k}")
        )
        steps.append(step)
        ends = union_all(
            select(step.c.subject_id.label("entity_id"), (step.c.rank * 2).label("ord")),
            select(step.c.object_id.label("entity_id"), (step.c.rank * 2 + 1).label("ord")),
        ).subquery()
        first_seen = func.min(ends.c.ord)
        frontier = (
            select(ends.c.entity_id)
            .where(ends.c.entity_id.not_in(select(visited.c.entity_id)))
            .group_by(ends.c.entity_id)
            .order_by(first_seen, ends.c.entity_id)
            .limit(func.greatest(0, max_visited - count))
            .cte(f"f{k + 1}")
        )
        steps_visited = visited
        visited = union_all(select(visited.c.entity_id), select(frontier.c.entity_id)).cte(
            f"v{k + 1}"
        )
    reached = union_all(
        *(
            select(step.c.relation_id, literal(k).label("hop"), step.c.rank)
            for k, step in enumerate(steps, start=1)
        )
    ).subquery()
    subject = aliased(GraphEntityRow)
    obj = aliased(GraphEntityRow)
    within = select(visited.c.entity_id)
    return (
        select(
            GraphRelationRow,
            subject,
            obj,
            select(func.count()).select_from(visited).scalar_subquery().label("visited"),
        )
        .join(reached, reached.c.relation_id == GraphRelationRow.relation_id)
        .join(subject, subject.entity_id == GraphRelationRow.subject_id)
        .join(obj, obj.entity_id == GraphRelationRow.object_id)
        .where(GraphRelationRow.subject_id.in_(within), GraphRelationRow.object_id.in_(within))
        .order_by(reached.c.hop, reached.c.rank)
    )


def traversal_params(
    tenant_id: str,
    seeds: Sequence[str],
    *,
    scope_keys: Sequence[str],
    max_visited: int,
    layers: Sequence[GraphLayer] | None,
    as_of: datetime | None,
    valid_at: datetime | None,
) -> dict[str, Any]:
    """The values of one traversal, for the statement ``traversal_query`` built its shape."""
    params: dict[str, Any] = {
        "tenant_id": tenant_id,
        "seeds": list(seeds),
        "keys": [str(k) for k in scope_keys],
        "max_visited": max_visited,
    }
    if layers:
        params["layers"] = list(layers)
    if as_of is not None:
        params["as_of"] = as_of
    if valid_at is not None:
        params["valid_at"] = valid_at
    return params


class PostgresGraphStore:
    def __init__(
        self,
        engine: AsyncEngine,
        *,
        budget_ms: int = GRAPH.prefetch_budget_ms,
        budgeted_url: str | None = None,
        budgeted_pool: tuple[int, int] | None = None,
    ) -> None:
        """``budgeted_url`` is where the traversal's own pool connects. Its statement
        timeout is a session parameter and its plan a prepared statement, both of which a
        transaction-mode pooler would hand to the next client, so behind one it goes
        direct (``DatabaseSettings.direct_url``). ``budgeted_pool`` is its (size, overflow)
        from the pod's connection budget."""
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)
        self._reads = async_sessionmaker(
            engine.execution_options(isolation_level="AUTOCOMMIT"), expire_on_commit=False
        )
        self._budget_ms = budget_ms
        size, overflow = budgeted_pool or (GRAPH.budgeted_pool_size, GRAPH.budgeted_pool_overflow)
        self._budgeted_engine = create_async_engine(
            budgeted_url or engine.url,
            isolation_level="AUTOCOMMIT",
            pool_size=size,
            max_overflow=overflow,
            pool_timeout=DATABASE.pool_timeout_seconds,
            pool_pre_ping=False,
            pool_recycle=DATABASE.pool_recycle_seconds,
            connect_args={
                "options": f"-c statement_timeout={budget_ms}",
                "connect_timeout": DATABASE.connect_timeout_seconds,
                # The traversal's text depends only on the hop count, so it is prepared on
                # first use and never planned again on that connection: planning counts
                # against the statement timeout, and it was a third of the statement.
                "prepare_threshold": 0,
            },
        )
        track_pool(self._budgeted_engine, "graph", size + overflow)
        self._budgeted = async_sessionmaker(self._budgeted_engine, expire_on_commit=False)

    async def close(self) -> None:
        await self._budgeted_engine.dispose()

    def session(self) -> AsyncSession:
        return self._sessions()

    def read(self) -> AsyncSession:
        """A session for reads only: autocommit, so no BEGIN or ROLLBACK round trip."""
        return self._reads()

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
                    layer=r.layer,
                    visibility_keys=list(r.visibility_keys),
                    valid_from=r.valid_from,
                    valid_to=r.valid_to,
                    observed_at=r.observed_at,
                    invalidated_at=r.invalidated_at,
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
                        "layer": stmt.excluded.layer,
                        "valid_to": stmt.excluded.valid_to,
                        "invalidated_at": stmt.excluded.invalidated_at,
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
                .values(
                    status="SUPERSEDED",
                    superseded_by=by,
                    valid_to=at,
                    invalidated_at=at,
                    updated_at=at,
                )
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
                .values(status="SUPERSEDED", valid_to=at, invalidated_at=at, updated_at=at)
                .returning(GraphRelationRow.relation_id)
            )
            return len(res.scalars().all())

    async def invalidate(
        self,
        relation_id: str,
        *,
        reason: str,
        at: datetime,
        by: str | None = None,
        status: str = "INVALIDATED",
        attributes: dict[str, Any] | None = None,
    ) -> Relation | None:
        async with self.session() as s, s.begin():
            row = await s.get(GraphRelationRow, relation_id)
            if row is None:
                return None
            row.status = status
            row.invalidated_at = row.invalidated_at or at
            row.attributes = {
                **dict(row.attributes or {}),
                "invalidation": {"reason": reason, "at": at.isoformat(), "by": by},
            }
            if status == "SUPERSEDED":
                row.valid_to = row.valid_to or at
                row.superseded_by = by or row.superseded_by
            row.updated_at = at
            winner = await s.get(GraphRelationRow, by) if by else None
            if winner is None:
                return None
            edge = invalidation_edge(
                _relation(row), _relation(winner), reason=reason, at=at, attributes=attributes
            )
        await self.upsert_relations([edge])
        return edge

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
        async with self.read() as s:
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

    async def search_entities(
        self,
        tenant_id: str,
        *,
        scope_keys: Sequence[str],
        prefix: str | None = None,
        entity_type: str | None = None,
        limit: int = 200,
    ) -> list[Entity]:
        if not scope_keys or limit <= 0:
            return []
        conds = [
            GraphEntityRow.tenant_id == tenant_id,
            _keys_clause(GraphEntityRow.visibility_keys, scope_keys),
        ]
        if prefix:
            # a range scan on ix_graph_entities_name_prefix (text_pattern_ops)
            conds.append(GraphEntityRow.canonical_name.startswith(prefix, autoescape=True))
        if entity_type:
            conds.append(GraphEntityRow.entity_type == entity_type)
        async with self.read() as s:
            rows = (
                await s.scalars(
                    select(GraphEntityRow)
                    .where(*conds)
                    .order_by(GraphEntityRow.mention_count.desc(), GraphEntityRow.entity_id)
                    .limit(limit)
                )
            ).all()
        return [_entity(r) for r in rows]

    async def entity_relations(
        self,
        tenant_id: str,
        entity_id: str,
        *,
        scope_keys: Sequence[str],
        current: bool,
        limit: int,
    ) -> list[Relation]:
        if not scope_keys or limit <= 0:
            return []
        status = (
            GraphRelationRow.status == "CURRENT"
            if current
            else GraphRelationRow.status != "CURRENT"
        )
        ends = [
            (GraphRelationRow.subject_id == entity_id),
            (GraphRelationRow.object_id == entity_id),
        ]
        # one indexed branch per end (ix_graph_relations_subject / _object), merged
        branches = [
            select(GraphRelationRow.relation_id)
            .where(
                GraphRelationRow.tenant_id == tenant_id,
                end,
                status,
                GraphRelationRow.predicate != INVALIDATED_BY,
                _keys_clause(GraphRelationRow.visibility_keys, scope_keys),
            )
            .order_by(GraphRelationRow.observed_at.desc(), GraphRelationRow.relation_id)
            .limit(limit)
            for end in ends
        ]
        ids = union_all(*branches).subquery()
        async with self.read() as s:
            rows = (
                await s.scalars(
                    select(GraphRelationRow)
                    .where(GraphRelationRow.relation_id.in_(select(ids.c.relation_id)))
                    .order_by(GraphRelationRow.observed_at.desc(), GraphRelationRow.relation_id)
                    .limit(limit)
                )
            ).all()
        return [_relation(r) for r in rows]

    async def summary_sources(
        self, tenant_id: str, entity_ids: Sequence[str], *, limit: int
    ) -> list[EntityFacts]:
        if not entity_ids or limit <= 0:
            return []
        target = aliased(GraphEntityRow)
        out: list[EntityFacts] = []
        async with self.read() as s:
            entities = (
                await s.scalars(
                    select(GraphEntityRow)
                    .where(
                        GraphEntityRow.tenant_id == tenant_id,
                        GraphEntityRow.entity_id.in_(list(entity_ids)),
                    )
                    .order_by(GraphEntityRow.entity_id)
                )
            ).all()
            for entity in entities:
                facts = (
                    await s.execute(
                        select(GraphRelationRow.predicate, target.name)
                        .join(target, target.entity_id == GraphRelationRow.object_id)
                        .where(
                            GraphRelationRow.tenant_id == tenant_id,
                            GraphRelationRow.subject_id == entity.entity_id,
                            GraphRelationRow.status == "CURRENT",
                            GraphRelationRow.layer != "structural",
                            # every reader of the entity can read the fact
                            GraphRelationRow.visibility_keys.op("@>")(
                                literal(list(entity.visibility_keys or []), JSONB)
                            ),
                        )
                        .order_by(
                            GraphRelationRow.confidence.desc(),
                            GraphRelationRow.observed_at.desc(),
                            GraphRelationRow.relation_id,
                        )
                        .limit(limit)
                    )
                ).all()
                out.append(
                    EntityFacts(
                        entity=_entity(entity),
                        facts=[(str(p), str(n)) for p, n in facts],
                        summary_source=entity.summary_source or "",
                    )
                )
        return out

    async def set_summary(
        self, tenant_id: str, entity_id: str, *, summary: str, source: str
    ) -> None:
        async with self.session() as s, s.begin():
            await s.execute(
                update(GraphEntityRow)
                .where(GraphEntityRow.tenant_id == tenant_id, GraphEntityRow.entity_id == entity_id)
                .values(summary=summary, summary_source=source)
            )

    async def get_entities(
        self, tenant_id: str, entity_ids: Sequence[str], *, scope_keys: Sequence[str]
    ) -> list[Entity]:
        if not entity_ids or not scope_keys:
            return []
        async with self.read() as s:
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
        valid_at: datetime | None = None,
        layers: Sequence[GraphLayer] | None = None,
        budgeted: bool = False,
    ) -> GraphNeighborhood:
        """The bounded traversal in one statement (``traversal_query``); a start entity
        with no visible relation is the one case that reads its row separately.
        ``budgeted`` runs it where the server stops it past the graph budget."""
        seeds = list(dict.fromkeys(entity_ids))[:max_visited]
        if not seeds or not scope_keys:
            return GraphNeighborhood(entities=[], relations=[], visited=0)
        params = traversal_params(
            tenant_id,
            seeds,
            scope_keys=scope_keys,
            max_visited=max_visited,
            layers=layers,
            as_of=as_of,
            valid_at=valid_at,
        )
        sessions = self._budgeted if budgeted else self._reads
        with span("graph.neighborhood", hops=hops), stage_seconds.labels("graph.traverse").time():
            async with sessions() as s:
                found = await self._walk(s, hops, params) if hops > 0 else []
                rows_by_id: dict[str, GraphEntityRow] = {}
                for _, subject, obj, _ in found:
                    rows_by_id.setdefault(subject.entity_id, subject)
                    rows_by_id.setdefault(obj.entity_id, obj)
                if unread := [eid for eid in seeds if eid not in rows_by_id]:
                    rows_by_id.update(
                        (row.entity_id, row)
                        for row in (
                            await s.scalars(
                                select(GraphEntityRow).where(
                                    GraphEntityRow.tenant_id == tenant_id,
                                    GraphEntityRow.entity_id.in_(unread),
                                    _keys_clause(GraphEntityRow.visibility_keys, scope_keys),
                                )
                            )
                        ).all()
                    )
        start = set(seeds)
        ordered = [rows_by_id[eid] for eid in seeds if eid in rows_by_id] + [
            row for eid, row in rows_by_id.items() if eid not in start
        ]
        return GraphNeighborhood(
            entities=[_entity(e) for e in ordered],
            relations=[_relation(row[0]) for row in found],
            visited=found[-1][3] if found else len(seeds),
        )

    async def _walk(self, s: AsyncSession, hops: int, params: dict[str, Any]) -> list[Any]:
        """The traversal statement's rows; a statement the server stopped at the budget is
        ``GraphBudgetExceededError``."""
        statement = traversal_query(
            hops,
            layers="layers" in params,
            as_of="as_of" in params,
            valid_at="valid_at" in params,
        )
        try:
            return list((await s.execute(statement, params)).all())
        except DBAPIError as exc:
            if isinstance(exc.orig, QueryCanceled):
                raise GraphBudgetExceededError(self._budget_ms) from exc
            raise

    async def relations_for_document(
        self,
        tenant_id: str,
        document_id: str,
        *,
        scope_keys: Sequence[str],
        include_invalidated: bool = False,
    ) -> list[Relation]:
        if not scope_keys:
            return []
        conds = [
            GraphRelationRow.tenant_id == tenant_id,
            GraphRelationRow.document_id == document_id,
            _keys_clause(GraphRelationRow.visibility_keys, scope_keys),
        ]
        if not include_invalidated:
            conds.append(GraphRelationRow.status != "INVALIDATED")
        async with self.read() as s:
            rows = (await s.scalars(select(GraphRelationRow).where(*conds))).all()
        return [_relation(r) for r in rows]

    async def count(self, tenant_id: str) -> tuple[int, int]:
        async with self.read() as s:
            e = await s.scalar(
                select(func.count())
                .select_from(GraphEntityRow)
                .where(GraphEntityRow.tenant_id == tenant_id)
            )
            r = await s.scalar(
                select(func.count())
                .select_from(GraphRelationRow)
                .where(
                    GraphRelationRow.tenant_id == tenant_id,
                    GraphRelationRow.predicate != INVALIDATED_BY,
                )
            )
        return int(e or 0), int(r or 0)

    async def ping(self) -> bool:
        try:
            async with self.session() as s:
                await s.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
