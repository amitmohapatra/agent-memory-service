"""PostgreSQL persistence of beliefs, their sources and consolidation progress."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import Text, delete, func, or_, select, update
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import array, insert
from sqlalchemy.ext.asyncio import AsyncSession

from memory_service.adapters.db.orm import BeliefProgressRow, BeliefRow, BeliefSourceRow
from memory_service.domain.beliefs import Belief, BeliefProgress

#: a current belief no newer value has made stale yet
_UNSTALE_CURRENT = (
    BeliefRow.superseded_by.is_(None),
    BeliefRow.valid_to.is_(None),
    BeliefRow.deleted_at.is_(None),
    BeliefRow.stale_since.is_(None),
)


def _belief(row: BeliefRow, sources: Sequence[str]) -> Belief:
    return Belief(
        belief_id=row.belief_id,
        tenant_id=row.tenant_id,
        scope_key=row.scope_key,
        person_key=row.person_key,
        visibility_keys=list(row.visibility_keys or []),
        text=row.text,
        facet=row.facet,
        first_seen_at=row.first_seen_at,
        last_seen_at=row.last_seen_at,
        supersedes=row.supersedes,
        superseded_by=row.superseded_by,
        valid_from=row.valid_from,
        valid_to=row.valid_to,
        content_hash=row.content_hash,
        index_fingerprint=row.index_fingerprint,
        revision=row.revision,
        created_at=row.created_at,
        updated_at=row.updated_at,
        deleted_at=row.deleted_at,
        stale_since=row.stale_since,
        source_ids=list(sources),
    )


def _progress(row: BeliefProgressRow) -> BeliefProgress:
    return BeliefProgress(
        tenant_id=row.tenant_id,
        scope_key=row.scope_key,
        person_key=row.person_key,
        source_fingerprint=row.source_fingerprint,
        last_consolidated_at=row.last_consolidated_at,
        pending_since=row.pending_since,
        failures=row.failures,
        paused_at=row.paused_at,
    )


def _fields(b: Belief) -> dict[str, object]:
    return {
        "tenant_id": b.tenant_id,
        "scope_key": b.scope_key,
        "visibility_keys": list(b.visibility_keys),
        "person_key": b.person_key,
        "text": b.text,
        "facet": b.facet,
        "first_seen_at": b.first_seen_at,
        "last_seen_at": b.last_seen_at,
        "proof_count": b.proof_count,
        "supersedes": b.supersedes,
        "superseded_by": b.superseded_by,
        "valid_from": b.valid_from,
        "valid_to": b.valid_to,
        "content_hash": b.content_hash,
        "index_fingerprint": b.index_fingerprint,
        "updated_at": b.updated_at,
        "deleted_at": b.deleted_at,
        # stale_since is never written from a belief read earlier: only mark_stale and
        # clear_stale move it, so a run cannot show again what went stale while it ran
    }


class SqlBeliefRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    # -- progress -----------------------------------------------------------------------
    async def progress(
        self, tenant_id: str, scope_key: str, person_key: str, *, lock: bool = False
    ) -> BeliefProgress | None:
        stmt = select(BeliefProgressRow).where(
            BeliefProgressRow.tenant_id == tenant_id,
            BeliefProgressRow.scope_key == scope_key,
            BeliefProgressRow.person_key == person_key,
        )
        if lock:
            stmt = stmt.with_for_update().execution_options(populate_existing=True)
        row = (await self.s.scalars(stmt)).first()
        return _progress(row) if row is not None else None

    async def save_progress(self, progress: BeliefProgress) -> None:
        values = progress.model_dump()
        stmt = insert(BeliefProgressRow).values(**values)
        await self.s.execute(
            stmt.on_conflict_do_update(
                index_elements=["tenant_id", "scope_key", "person_key"],
                set_={
                    k: stmt.excluded[k]
                    for k in values
                    if k not in ("tenant_id", "scope_key", "person_key")
                },
            )
        )

    async def mark_pending(
        self, tenant_id: str, scope_key: str, person_key: str, at: datetime
    ) -> datetime:
        stmt = insert(BeliefProgressRow).values(
            tenant_id=tenant_id, scope_key=scope_key, person_key=person_key, pending_since=at
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["tenant_id", "scope_key", "person_key"],
            set_={
                "pending_since": func.coalesce(BeliefProgressRow.pending_since, at),
                # a new fact is a new chance for a paused person
                "paused_at": None,
            },
        ).returning(BeliefProgressRow.pending_since)
        since = (await self.s.execute(stmt)).scalar_one()
        assert since is not None  # set by the insert or kept by the update
        return since

    async def pending(self, before: datetime, *, limit: int) -> list[BeliefProgress]:
        rows = await self.s.scalars(
            select(BeliefProgressRow)
            .where(
                BeliefProgressRow.pending_since.is_not(None),
                BeliefProgressRow.pending_since < before,
                BeliefProgressRow.paused_at.is_(None),
            )
            .order_by(BeliefProgressRow.pending_since)
            .limit(limit)
        )
        return [_progress(r) for r in rows]

    # -- beliefs ------------------------------------------------------------------------
    async def _with_sources(self, rows: Sequence[BeliefRow]) -> list[Belief]:
        if not rows:
            return []
        found: dict[str, list[str]] = {r.belief_id: [] for r in rows}
        pairs = await self.s.execute(
            select(BeliefSourceRow.belief_id, BeliefSourceRow.memory_id)
            .where(BeliefSourceRow.belief_id.in_(list(found)))
            .order_by(BeliefSourceRow.added_at, BeliefSourceRow.memory_id)
        )
        for belief_id, memory_id in pairs:
            found[belief_id].append(memory_id)
        return [_belief(r, found[r.belief_id]) for r in rows]

    async def of_unit(
        self, tenant_id: str, scope_key: str, person_key: str, *, current_only: bool = True
    ) -> list[Belief]:
        stmt = select(BeliefRow).where(
            BeliefRow.tenant_id == tenant_id,
            BeliefRow.scope_key == scope_key,
            BeliefRow.person_key == person_key,
            BeliefRow.deleted_at.is_(None),
        )
        if current_only:
            stmt = stmt.where(BeliefRow.superseded_by.is_(None), BeliefRow.valid_to.is_(None))
        rows = (
            await self.s.scalars(stmt.order_by(BeliefRow.first_seen_at, BeliefRow.belief_id))
        ).all()
        return await self._with_sources(rows)

    async def _add_sources(self, belief: Belief) -> None:
        if belief.source_ids:
            await self.s.execute(
                insert(BeliefSourceRow)
                .values(
                    [
                        {
                            "belief_id": belief.belief_id,
                            "memory_id": m,
                            "tenant_id": belief.tenant_id,
                        }
                        for m in belief.source_ids
                    ]
                )
                .on_conflict_do_nothing()
            )

    async def add(self, belief: Belief) -> None:
        self.s.add(
            BeliefRow(
                belief_id=belief.belief_id,
                created_at=belief.created_at,
                revision=belief.revision,
                **_fields(belief),
            )
        )
        await self.s.flush()
        await self._add_sources(belief)

    async def update(self, belief: Belief) -> None:
        await self.s.execute(
            update(BeliefRow)
            .where(BeliefRow.belief_id == belief.belief_id)
            .values(**_fields(belief), revision=BeliefRow.revision + 1)
        )
        await self._add_sources(belief)

    async def _drop(self, ids: Sequence[str], at: datetime) -> None:
        """Delete beliefs and their words; their sources go with them."""
        await self.s.execute(
            update(BeliefRow)
            .where(BeliefRow.belief_id.in_(list(ids)))
            .values(
                deleted_at=at,
                text="",
                proof_count=0,
                index_fingerprint=None,
                updated_at=at,
                revision=BeliefRow.revision + 1,
            )
        )
        await self.s.execute(
            delete(BeliefSourceRow).where(BeliefSourceRow.belief_id.in_(list(ids)))
        )

    async def delete_unit(
        self, tenant_id: str, scope_key: str, person_key: str, at: datetime
    ) -> list[Belief]:
        ids = list(
            await self.s.scalars(
                select(BeliefRow.belief_id).where(
                    BeliefRow.tenant_id == tenant_id,
                    BeliefRow.scope_key == scope_key,
                    BeliefRow.person_key == person_key,
                    BeliefRow.deleted_at.is_(None),
                )
            )
        )
        if not ids:
            return []
        await self._drop(ids, at)
        return await self.get_many(tenant_id, ids)

    async def forget_sources(
        self, tenant_id: str, memory_ids: Sequence[str], at: datetime
    ) -> list[Belief]:
        if not memory_ids:
            return []
        removed = await self.s.execute(
            delete(BeliefSourceRow)
            .where(
                BeliefSourceRow.tenant_id == tenant_id,
                BeliefSourceRow.memory_id.in_(list(memory_ids)),
            )
            .returning(BeliefSourceRow.belief_id)
        )
        ids = sorted(set(removed.scalars()))
        if not ids:
            return []
        # a belief that lost any source may still say what it said: dropped at once, words
        # and all; the person's rebuild writes again what the sources left still support
        await self._drop(ids, at)
        return await self.get_many(tenant_id, ids)

    async def mark_stale(
        self, tenant_id: str, memory_ids: Sequence[str], at: datetime
    ) -> list[Belief]:
        if not memory_ids:
            return []
        cites = select(BeliefSourceRow.belief_id).where(
            BeliefSourceRow.tenant_id == tenant_id,
            BeliefSourceRow.memory_id.in_(list(memory_ids)),
        )
        marked = await self.s.execute(
            update(BeliefRow)
            .where(
                BeliefRow.tenant_id == tenant_id,
                BeliefRow.belief_id.in_(cites),
                *_UNSTALE_CURRENT,
            )
            .values(stale_since=at, index_fingerprint=None, revision=BeliefRow.revision + 1)
            .returning(BeliefRow.belief_id)
        )
        return await self.get_many(tenant_id, sorted(marked.scalars()))

    async def clear_stale(
        self, tenant_id: str, scope_key: str, person_key: str, *, before: datetime
    ) -> list[Belief]:
        cleared = await self.s.execute(
            update(BeliefRow)
            .where(
                BeliefRow.tenant_id == tenant_id,
                BeliefRow.scope_key == scope_key,
                BeliefRow.person_key == person_key,
                BeliefRow.deleted_at.is_(None),
                BeliefRow.stale_since <= before,
            )
            .values(stale_since=None, index_fingerprint=None, revision=BeliefRow.revision + 1)
            .returning(BeliefRow.belief_id)
        )
        return await self.get_many(tenant_id, sorted(cleared.scalars()))

    async def visible(
        self,
        tenant_id: str,
        keys: Sequence[str],
        *,
        scope_keys: Sequence[str],
        people: int,
        history: int,
    ) -> list[Belief]:
        readable = (
            BeliefRow.tenant_id == tenant_id,
            BeliefRow.scope_key.in_(list(scope_keys)),
            BeliefRow.deleted_at.is_(None),
            BeliefRow.visibility_keys.op("?|")(array(list(keys), type_=Text)),
        )
        current = (
            BeliefRow.superseded_by.is_(None),
            BeliefRow.valid_to.is_(None),
            BeliefRow.stale_since.is_(None),
        )
        ranked = (
            select(BeliefRow.person_key)
            .where(*readable, *current)
            .group_by(BeliefRow.person_key)
            .order_by(func.sum(BeliefRow.proof_count).desc(), BeliefRow.person_key)
            .limit(people)
        )
        persons = list(await self.s.scalars(ranked))
        if not persons:
            return []
        now = (
            await self.s.scalars(
                select(BeliefRow).where(*readable, *current, BeliefRow.person_key.in_(persons))
            )
        ).all()
        newest = func.row_number().over(
            partition_by=BeliefRow.person_key,
            order_by=func.coalesce(BeliefRow.valid_to, BeliefRow.updated_at).desc(),
        )
        earlier = (
            select(BeliefRow, newest.label("n"))
            .where(
                *readable,
                BeliefRow.person_key.in_(persons),
                or_(BeliefRow.superseded_by.is_not(None), BeliefRow.valid_to.is_not(None)),
            )
            .subquery()
        )
        old = (
            await self.s.scalars(
                select(BeliefRow)
                .join(earlier, BeliefRow.belief_id == earlier.c.belief_id)
                .where(earlier.c.n <= history)
            )
        ).all()
        return await self._with_sources([*now, *old])

    async def get_many(self, tenant_id: str, belief_ids: Sequence[str]) -> list[Belief]:
        if not belief_ids:
            return []
        rows = (
            await self.s.scalars(
                select(BeliefRow).where(
                    BeliefRow.tenant_id == tenant_id, BeliefRow.belief_id.in_(list(belief_ids))
                )
            )
        ).all()
        return await self._with_sources(rows)

    async def mark_indexed(self, belief_ids: Sequence[str], *, fingerprint: str) -> None:
        if belief_ids:
            await self.s.execute(
                update(BeliefRow)
                .where(BeliefRow.belief_id.in_(list(belief_ids)))
                .values(index_fingerprint=fingerprint)
            )

    async def prune_history(self, *, keep: int, limit: int) -> int:
        removed = await self.s.execute(_PRUNE, {"keep": keep, "limit": limit})
        return len(removed.all())


#: A belief's versions are the rows that lead to it through ``superseded_by``, the newest
#: (fewest steps away) first; past the newest ``keep`` they go, sources with them (cascade).
_PRUNE = sql_text(
    """
    WITH RECURSIVE chain(head, belief_id, depth) AS (
        SELECT b.belief_id, b.belief_id, 0 FROM beliefs b
        WHERE b.superseded_by IS NULL
          AND EXISTS (SELECT 1 FROM beliefs c WHERE c.superseded_by = b.belief_id)
        UNION ALL
        SELECT chain.head, c.belief_id, chain.depth + 1
        FROM beliefs c JOIN chain ON c.superseded_by = chain.belief_id
    ), ranked AS (
        SELECT chain.belief_id, row_number() OVER (
            PARTITION BY chain.head ORDER BY chain.depth, b.created_at DESC, b.belief_id DESC
        ) AS n
        FROM chain JOIN beliefs b ON b.belief_id = chain.belief_id
        WHERE chain.depth > 0
    )
    DELETE FROM beliefs
    WHERE belief_id IN (SELECT belief_id FROM ranked WHERE n > :keep LIMIT :limit)
    RETURNING belief_id
    """
)
