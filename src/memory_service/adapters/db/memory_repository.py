"""PostgreSQL MemoryRepository (M7)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import Text, and_, func, literal, or_, select, text, tuple_, update
from sqlalchemy.dialects.postgresql import array, insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from memory_service.adapters.db.orm import (
    MemoryDependencyRow,
    MemoryReflectionProgressRow,
    MemoryRow,
)
from memory_service.config.constants import REFLECTION_SOURCE_CHARS
from memory_service.domain.enums import (
    Lifetime,
    MemoryType,
    Representation,
    ScopeLevel,
    TemporalStatus,
    Visibility,
)
from memory_service.domain.errors import ValidationFailed
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.language import detect_language
from memory_service.domain.memory import (
    UNVERIFIED_MEMORY_CATEGORIES,
    CanonicalMemory,
    Scope,
    TemporalState,
)


def _asserted_source_filter() -> ColumnElement[bool]:
    """The source-trust boundary reflection and connections read through."""
    return and_(
        MemoryRow.derived_slot.is_(None),
        func.coalesce(MemoryRow.system_metadata["category"].astext, "").not_in(
            UNVERIFIED_MEMORY_CATEGORIES
        ),
        MemoryRow.provider.is_distinct_from("llm"),
    )


def _to_domain(r: MemoryRow) -> CanonicalMemory:
    scope = Scope(
        level=ScopeLevel(r.scope_level),
        tenant_id=r.tenant_id,
        workspace_id=r.workspace_id,
        user_id=r.user_id,
        thread_id=r.thread_id,
        agent_id=r.agent_id,
        agent_group_id=r.agent_group_id,
    )
    temporal = TemporalState(
        status=TemporalStatus(r.temporal_status),
        valid_from=r.valid_from,
        valid_to=r.valid_to,
        observed_at=r.observed_at,
        superseded_by=r.superseded_by,
        supersedes=r.supersedes,
        contradicts=list(r.contradicts or []),
    )
    return CanonicalMemory(
        memory_id=r.memory_id,
        tenant_id=r.tenant_id,
        scope=scope,
        visibility=Visibility(r.visibility),
        owner_principal=r.owner_principal,
        lifetime=Lifetime(r.lifetime),
        memory_type=MemoryType(r.memory_type),
        custom_type=r.custom_type,
        representation=Representation.MEMORY,
        content=r.content,
        normalized_hash=r.normalized_hash,
        subject=r.subject,
        predicate=r.predicate,
        object=r.object,
        temporal=temporal,
        evidence=[EvidenceRef.model_validate(e) for e in (r.evidence or [])],
        confidence=r.confidence,
        importance=r.importance,
        reinforcement_count=r.reinforcement_count,
        access_count=r.access_count or 0,
        last_accessed_at=r.last_accessed_at,
        system_metadata={
            **(r.system_metadata or {}),
            "provider": r.provider,
            "expires_at": r.expires_at.isoformat() if r.expires_at else None,
            # The search-index receipt. Carried here rather than on CanonicalMemory, which is
            # extra="forbid" and describes what a memory IS, not where a copy of it has got to.
            "indexed_at": r.indexed_at.isoformat() if r.indexed_at else None,
            "visibility_keys": list(r.visibility_keys or []),
            "derived_slot": r.derived_slot,
        },
        custom_metadata=dict(r.custom_metadata or {}),
        created_at=r.created_at,
        updated_at=r.updated_at,
        revision=r.revision,
        deleted_at=r.deleted_at,
        lang=r.lang or "",
    )


def _evidence_json(memory: CanonicalMemory) -> list[dict[str, Any]]:
    return [json.loads(e.model_dump_json(exclude_none=True)) for e in memory.evidence]


def _invalid_derived(row: MemoryRow) -> bool:
    return bool(row.derived_slot or (row.system_metadata or {}).get("source_revisions")) and (
        row.temporal_status == TemporalStatus.RETRACTED.value
        or (row.expires_at is not None and row.expires_at <= datetime.now(UTC))
    )


class SqlMemoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session
        self.invalidated: dict[str, set[str]] = {}
        #: The dependents retracted in this transaction, as they now are: the unit of work
        #: moves the revisions their readers' caches are keyed on when it commits.
        self.invalidated_memories: list[CanonicalMemory] = []

    async def _invalidate_dependents(self, tenant_id: str, source_ids: Sequence[str]) -> None:
        if not source_ids:
            return
        result = await self.s.execute(
            text("""
            WITH RECURSIVE affected(memory_id) AS (
                SELECT derived_id FROM memory_dependencies
                WHERE tenant_id = :tenant AND source_id = ANY(:sources)
                UNION
                SELECT d.derived_id FROM memory_dependencies d
                JOIN affected a ON d.source_id = a.memory_id
                WHERE d.tenant_id = :tenant
            )
            UPDATE memories SET temporal_status = 'RETRACTED', indexed_at = NULL,
                updated_at = now(), revision = revision + 1
            WHERE tenant_id = :tenant AND temporal_status != 'RETRACTED'
                AND memory_id IN (SELECT memory_id FROM affected)
            RETURNING memory_id
        """),
            {"tenant": tenant_id, "sources": list(source_ids)},
        )
        ids = list(result.scalars())
        self.invalidated.setdefault(tenant_id, set()).update(ids)
        if ids:
            # Raw recursive SQL bypasses ORM synchronization; refresh any identity-map
            # entries before this transaction can look them up again.
            refreshed = await self.s.scalars(
                select(MemoryRow)
                .where(MemoryRow.memory_id.in_(ids))
                .execution_options(populate_existing=True)
            )
            self.invalidated_memories.extend(_to_domain(r) for r in refreshed.all())

    async def _source_rows(
        self, memory: CanonicalMemory, visibility_keys: Sequence[str]
    ) -> list[MemoryRow]:
        ids = sorted({e.source_id for e in memory.evidence if e.source_type == "memory"})
        if not ids:
            return []
        rows = list(
            (
                await self.s.scalars(
                    select(MemoryRow)
                    .where(MemoryRow.tenant_id == memory.tenant_id, MemoryRow.memory_id.in_(ids))
                    .order_by(MemoryRow.memory_id)
                    .with_for_update(read=True)
                    .execution_options(populate_existing=True)
                )
            ).all()
        )
        expected = memory.system_metadata.get("source_revisions", {})
        now = datetime.now(UTC)
        if len(rows) != len(ids) or any(
            r.deleted_at is not None
            or r.temporal_status != TemporalStatus.CURRENT.value
            or (r.expires_at is not None and r.expires_at <= now)
            or expected.get(r.memory_id) != r.revision
            or not set(visibility_keys).issubset(r.visibility_keys or [])
            for r in rows
        ):
            raise ValidationFailed("Derived sources changed or have incompatible audiences")
        if not visibility_keys:
            raise ValidationFailed("Derived memory requires a nonempty source audience")
        return rows

    async def current_derived(self, tenant_id: str, slot: str) -> CanonicalMemory | None:
        row = await self.s.scalar(
            select(MemoryRow).where(
                MemoryRow.tenant_id == tenant_id,
                MemoryRow.derived_slot == slot,
                MemoryRow.temporal_status == TemporalStatus.CURRENT.value,
                MemoryRow.deleted_at.is_(None),
            )
        )
        return _to_domain(row) if row is not None else None

    async def add(self, memory: CanonicalMemory, *, visibility_keys: Sequence[str]) -> None:
        sources = await self._source_rows(memory, visibility_keys)
        sm = dict(memory.system_metadata)
        provider = str(sm.pop("provider", "native"))
        expires_raw = sm.pop("expires_at", None)
        sm.pop("visibility_keys", None)
        expires_at = datetime.fromisoformat(expires_raw) if isinstance(expires_raw, str) else None
        self.s.add(
            MemoryRow(
                memory_id=memory.memory_id,
                tenant_id=memory.tenant_id,
                scope_level=memory.scope.level.value,
                scope_key=memory.scope.key(),
                workspace_id=memory.scope.workspace_id,
                user_id=memory.scope.user_id,
                thread_id=memory.scope.thread_id,
                agent_id=memory.scope.agent_id,
                agent_group_id=memory.scope.agent_group_id,
                visibility=memory.visibility.value,
                visibility_keys=list(visibility_keys),
                owner_principal=memory.owner_principal,
                lifetime=memory.lifetime.value,
                memory_type=memory.memory_type.value,
                custom_type=memory.custom_type,
                content=memory.content,
                normalized_hash=memory.normalized_hash,
                subject=memory.subject,
                predicate=memory.predicate,
                derived_slot=sm.get("derived_slot"),
                object=memory.object,
                temporal_status=memory.temporal.status.value,
                valid_from=memory.temporal.valid_from,
                valid_to=memory.temporal.valid_to,
                observed_at=memory.temporal.observed_at,
                superseded_by=memory.temporal.superseded_by,
                supersedes=memory.temporal.supersedes,
                contradicts=list(memory.temporal.contradicts),
                evidence=_evidence_json(memory),
                confidence=memory.confidence,
                importance=memory.importance,
                reinforcement_count=memory.reinforcement_count,
                access_count=memory.access_count,
                last_accessed_at=memory.last_accessed_at,
                provider=provider,
                system_metadata=sm,
                custom_metadata=memory.custom_metadata,
                expires_at=expires_at,
                created_at=memory.created_at,
                updated_at=memory.updated_at,
                revision=memory.revision,
                lang=memory.lang,
            )
        )
        await self.s.flush()
        self.s.add_all(
            [
                MemoryDependencyRow(
                    tenant_id=memory.tenant_id,
                    derived_id=memory.memory_id,
                    source_id=source.memory_id,
                    source_revision=source.revision,
                )
                for source in sources
            ]
        )
        if sources:
            await self.s.flush()

    async def get(self, tenant_id: str, memory_id: str) -> CanonicalMemory | None:
        r = await self.s.get(MemoryRow, memory_id)
        if r is None or r.tenant_id != tenant_id or r.deleted_at is not None or _invalid_derived(r):
            return None
        return _to_domain(r)

    async def get_many(self, tenant_id: str, memory_ids: Sequence[str]) -> list[CanonicalMemory]:
        if not memory_ids:
            return []
        rows = (
            await self.s.scalars(
                select(MemoryRow).where(
                    MemoryRow.tenant_id == tenant_id,
                    MemoryRow.memory_id.in_(list(memory_ids)),
                    MemoryRow.deleted_at.is_(None),
                )
            )
        ).all()
        by_id = {r.memory_id: _to_domain(r) for r in rows if not _invalid_derived(r)}
        return [by_id[i] for i in memory_ids if i in by_id]

    async def visibility_keys(self, tenant_id: str, memory_id: str) -> list[str]:
        r = await self.s.get(MemoryRow, memory_id)
        if r is None or r.tenant_id != tenant_id or r.deleted_at is not None:
            return []
        return list(r.visibility_keys or [])

    async def update(self, memory: CanonicalMemory) -> None:
        r = await self.s.get(MemoryRow, memory.memory_id)
        if r is None or r.tenant_id != memory.tenant_id:
            raise LookupError(memory.memory_id)
        sm = dict(memory.system_metadata)
        sm.pop("provider", None)
        expires_raw = sm.pop("expires_at", None)
        sm.pop("visibility_keys", None)
        r.content = memory.content
        r.lang = detect_language(memory.content)
        r.normalized_hash = memory.normalized_hash
        r.subject = memory.subject
        r.predicate = memory.predicate
        r.object = memory.object
        r.temporal_status = memory.temporal.status.value
        r.valid_from = memory.temporal.valid_from
        r.valid_to = memory.temporal.valid_to
        r.superseded_by = memory.temporal.superseded_by
        r.supersedes = memory.temporal.supersedes
        r.contradicts = list(memory.temporal.contradicts)  # type: ignore[assignment]
        r.evidence = _evidence_json(memory)  # type: ignore[assignment]
        r.confidence = memory.confidence
        r.importance = memory.importance
        r.reinforcement_count = memory.reinforcement_count
        r.access_count = max(r.access_count or 0, memory.access_count)
        if memory.last_accessed_at is not None:
            r.last_accessed_at = memory.last_accessed_at
        r.lifetime = memory.lifetime.value
        r.memory_type = memory.memory_type.value
        r.visibility = memory.visibility.value
        r.system_metadata = sm
        r.custom_metadata = memory.custom_metadata
        if isinstance(expires_raw, str):
            r.expires_at = datetime.fromisoformat(expires_raw)
        elif expires_raw is None and "expires_at" in memory.system_metadata:
            r.expires_at = None
        r.updated_at = memory.updated_at
        r.revision = r.revision + 1
        r.indexed_at = None  # content or state changed -> re-index
        await self.s.flush()
        memory.revision = r.revision
        await self._invalidate_dependents(memory.tenant_id, [memory.memory_id])

    async def current_with_hash(
        self, tenant_id: str, *, scope_key: str, owner_principal: str, normalized_hash: str
    ) -> CanonicalMemory | None:
        row = (
            await self.s.scalars(
                select(MemoryRow)
                .where(
                    MemoryRow.tenant_id == tenant_id,
                    MemoryRow.normalized_hash == normalized_hash,
                    MemoryRow.scope_key == scope_key,
                    MemoryRow.owner_principal == owner_principal,
                    MemoryRow.deleted_at.is_(None),
                    MemoryRow.temporal_status == TemporalStatus.CURRENT.value,
                )
                .order_by(MemoryRow.created_at, MemoryRow.memory_id)
                .limit(1)
            )
        ).first()
        return _to_domain(row) if row is not None else None

    async def twins(
        self, tenant_id: str, memory: CanonicalMemory, *, current_only: bool = True
    ) -> list[CanonicalMemory]:
        conds = [
            MemoryRow.tenant_id == tenant_id,
            MemoryRow.normalized_hash == memory.normalized_hash,
            MemoryRow.owner_principal == memory.owner_principal,
            MemoryRow.memory_id != memory.memory_id,
            MemoryRow.deleted_at.is_(None),
        ]
        if current_only:
            conds.append(MemoryRow.temporal_status == TemporalStatus.CURRENT.value)
        rows = (await self.s.scalars(select(MemoryRow).where(*conds))).all()
        sources = {(ev.source_type, ev.source_id) for ev in memory.evidence}
        return [
            twin
            for twin in map(_to_domain, rows)
            if sources & {(ev.source_type, ev.source_id) for ev in twin.evidence}
        ]

    async def candidates(
        self,
        tenant_id: str,
        *,
        scope_key: str,
        normalized_hash: str | None = None,
        subjects: Sequence[str] = (),
        limit: int = 20,
    ) -> list[CanonicalMemory]:
        conds = [
            MemoryRow.tenant_id == tenant_id,
            MemoryRow.scope_key == scope_key,
            MemoryRow.deleted_at.is_(None),
            MemoryRow.temporal_status == TemporalStatus.CURRENT.value,
        ]
        any_of = []
        if normalized_hash:
            any_of.append(MemoryRow.normalized_hash == normalized_hash)
        if subjects:
            any_of.append(MemoryRow.subject.in_(list(subjects)))
        exact = []
        if any_of:
            exact = list(
                (
                    await self.s.scalars(
                        select(MemoryRow)
                        .where(*conds, or_(*any_of))
                        .order_by(MemoryRow.updated_at.desc())
                        .limit(limit)
                    )
                ).all()
            )
        # The most recent of each kind: a statement's turn is kept beside its reading (ADR
        # 0036), so one window over both held half as many readings as before, and a turn
        # still belongs in it - the memories at hand are what a tenant's own definitions
        # ("the cross-dock facility (CDF)") are learned from.
        turn = MemoryRow.system_metadata["category"].astext == "verbatim_turn"
        recent: list[MemoryRow] = []
        for kind in (turn.is_(False) | turn.is_(None), turn.is_(True)):
            recent += (
                await self.s.scalars(
                    select(MemoryRow)
                    .where(*conds, kind)
                    .order_by(MemoryRow.updated_at.desc())
                    .limit(limit)
                )
            ).all()
        seen: dict[str, MemoryRow] = {}
        for r in exact + recent:
            seen.setdefault(r.memory_id, r)
        return [_to_domain(r) for r in seen.values()]

    async def about_user(
        self, tenant_id: str, user_id: str, *, memory_types: Sequence[str], limit: int
    ) -> list[CanonicalMemory]:
        audience = [f"user:{tenant_id}/{user_id}", f"principal:{tenant_id}/user:{user_id}"]
        rows = (
            await self.s.scalars(
                select(MemoryRow)
                .where(
                    MemoryRow.tenant_id == tenant_id,
                    MemoryRow.user_id == user_id,
                    MemoryRow.memory_type.in_(list(memory_types)),
                    MemoryRow.deleted_at.is_(None),
                    MemoryRow.temporal_status == TemporalStatus.CURRENT.value,
                    MemoryRow.visibility_keys.op("?|")(array(audience, type_=Text)),
                )
                .order_by(MemoryRow.created_at.desc(), MemoryRow.memory_id.desc())
                .limit(limit)
            )
        ).all()
        return [_to_domain(r) for r in rows]

    async def list_scope(
        self,
        tenant_id: str,
        *,
        scope_keys: Sequence[str],
        memory_types: Sequence[str] | None = None,
        current_only: bool = True,
        before: tuple[datetime, str] | None = None,
        limit: int = 200,
    ) -> list[CanonicalMemory]:
        """Newest created first. The page holds ``limit`` valid rows: a derived row that has
        lapsed is skipped and replaced by the next one, so fewer than ``limit`` rows means
        the scope is exhausted (the keyset is ``(created_at, memory_id)``, which never moves)."""
        if not scope_keys:
            return []
        out: list[CanonicalMemory] = []
        while len(out) < limit:
            stmt = select(MemoryRow).where(
                MemoryRow.tenant_id == tenant_id,
                MemoryRow.scope_key.in_(list(scope_keys)),
                MemoryRow.deleted_at.is_(None),
            )
            if current_only:
                stmt = stmt.where(MemoryRow.temporal_status == TemporalStatus.CURRENT.value)
            if memory_types:
                stmt = stmt.where(MemoryRow.memory_type.in_(list(memory_types)))
            if before is not None:
                stmt = stmt.where(
                    tuple_(MemoryRow.created_at, MemoryRow.memory_id)
                    < tuple_(literal(before[0]), literal(before[1]))
                )
            stmt = stmt.order_by(MemoryRow.created_at.desc(), MemoryRow.memory_id.desc())
            wanted = limit - len(out)
            rows = (await self.s.scalars(stmt.limit(wanted))).all()
            out.extend(_to_domain(r) for r in rows if not _invalid_derived(r))
            if len(rows) < wanted:
                break
            before = (rows[-1].created_at, rows[-1].memory_id)
        return out

    async def forget(self, tenant_id: str, memory_id: str) -> bool:
        r = await self.s.get(MemoryRow, memory_id)
        if r is None or r.tenant_id != tenant_id or r.deleted_at is not None:
            return False
        r.deleted_at = datetime.now(r.created_at.tzinfo)
        r.temporal_status = TemporalStatus.RETRACTED.value
        r.revision += 1
        r.indexed_at = None
        await self.s.flush()
        await self._invalidate_dependents(tenant_id, [memory_id])
        return True

    async def is_forgotten(self, tenant_id: str, memory_id: str) -> bool:
        r = await self.s.get(MemoryRow, memory_id)
        return r is not None and r.tenant_id == tenant_id and r.deleted_at is not None

    async def mark_indexed(
        self, memory_ids: Sequence[str], *, fingerprint: str, indexed_at: datetime
    ) -> None:
        if not memory_ids:
            return
        await self.s.execute(
            update(MemoryRow)
            .where(MemoryRow.memory_id.in_(list(memory_ids)))
            .values(indexed_at=indexed_at, index_fingerprint=fingerprint)
        )

    async def list_older_than(
        self, tenant_id: str, *, before: datetime, limit: int = 500
    ) -> list[CanonicalMemory]:
        rows = (
            await self.s.scalars(
                select(MemoryRow)
                .where(
                    MemoryRow.tenant_id == tenant_id,
                    MemoryRow.created_at < before,
                    MemoryRow.deleted_at.is_(None),
                )
                .order_by(MemoryRow.created_at)
                .limit(limit)
            )
        ).all()
        return [_to_domain(r) for r in rows]

    async def expire_due(self, *, now: datetime, limit: int = 500) -> list[tuple[str, str]]:
        rows = (
            await self.s.scalars(
                select(MemoryRow)
                .where(
                    MemoryRow.expires_at.is_not(None),
                    MemoryRow.expires_at <= now,
                    MemoryRow.deleted_at.is_(None),
                    MemoryRow.temporal_status == TemporalStatus.CURRENT.value,
                )
                .limit(limit)
            )
        ).all()
        out = []
        for r in rows:
            r.temporal_status = TemporalStatus.EXPIRED.value
            r.updated_at = now
            r.indexed_at = None
            out.append((r.tenant_id, r.memory_id))
        await self.s.flush()
        for tenant in {t for t, _ in out}:
            await self._invalidate_dependents(tenant, [i for t, i in out if t == tenant])
        return out

    async def list_recent(
        self, *, since: datetime, limit: int = 1000, tenant_id: str | None = None
    ) -> list[CanonicalMemory]:
        statement = (
            select(MemoryRow)
            .where(
                MemoryRow.updated_at >= since,
                MemoryRow.deleted_at.is_(None),
                MemoryRow.temporal_status == TemporalStatus.CURRENT.value,
                or_(MemoryRow.expires_at.is_(None), MemoryRow.expires_at > datetime.now(UTC)),
            )
            .order_by(MemoryRow.updated_at.desc(), MemoryRow.memory_id.desc())
            .limit(limit)
        )
        if tenant_id is not None:
            statement = statement.where(MemoryRow.tenant_id == tenant_id)
        rows = (await self.s.scalars(statement)).all()
        return [_to_domain(r) for r in rows]

    async def reflection_pending(
        self, *, limit: int = 1000, tenant_id: str | None = None
    ) -> list[CanonicalMemory]:
        progress = MemoryReflectionProgressRow
        statement = (
            select(MemoryRow)
            .outerjoin(
                progress,
                and_(
                    progress.tenant_id == MemoryRow.tenant_id,
                    progress.memory_id == MemoryRow.memory_id,
                ),
            )
            .where(
                MemoryRow.deleted_at.is_(None),
                MemoryRow.temporal_status == TemporalStatus.CURRENT.value,
                _asserted_source_filter(),
                or_(MemoryRow.expires_at.is_(None), MemoryRow.expires_at > datetime.now(UTC)),
                or_(progress.memory_id.is_(None), progress.source_revision != MemoryRow.revision),
                func.length(MemoryRow.content) <= REFLECTION_SOURCE_CHARS,
            )
            .order_by(MemoryRow.updated_at, MemoryRow.memory_id)
            .limit(limit)
        )
        if tenant_id is not None:
            statement = statement.where(MemoryRow.tenant_id == tenant_id)
        rows = (await self.s.scalars(statement)).all()
        return [_to_domain(row) for row in rows]

    async def mark_reflected(self, sources: Sequence[CanonicalMemory], *, at: datetime) -> None:
        if not sources:
            return
        # One statement for the bounded batch. A racing update remains pending because
        # its revision differs; a late worker may never downgrade a newer receipt.
        statement = insert(MemoryReflectionProgressRow).values(
            [
                {
                    "tenant_id": source.tenant_id,
                    "memory_id": source.memory_id,
                    "source_revision": source.revision,
                    "processed_at": at,
                }
                for source in {m.memory_id: m for m in sources}.values()
            ]
        )
        await self.s.execute(
            statement.on_conflict_do_update(
                index_elements=["tenant_id", "memory_id"],
                set_={
                    "source_revision": statement.excluded.source_revision,
                    "processed_at": statement.excluded.processed_at,
                },
                where=statement.excluded.source_revision
                >= MemoryReflectionProgressRow.source_revision,
            )
        )

    async def related(
        self,
        tenant_id: str,
        *,
        scope_key: str,
        subject: str,
        exclude: Sequence[str] = (),
        limit: int = 8,
        owner_principal: str | None = None,
        visibility_keys: Sequence[str] | None = None,
        include_derived: bool = True,
        include_verbatim: bool = False,
    ) -> list[CanonicalMemory]:
        stmt = select(MemoryRow).where(
            MemoryRow.tenant_id == tenant_id,
            MemoryRow.scope_key == scope_key,
            MemoryRow.subject == subject,
            MemoryRow.deleted_at.is_(None),
            MemoryRow.temporal_status == TemporalStatus.CURRENT.value,
        )
        if exclude:
            stmt = stmt.where(MemoryRow.memory_id.not_in(list(exclude)))
        if owner_principal is not None:
            stmt = stmt.where(MemoryRow.owner_principal == owner_principal)
        if visibility_keys is not None:
            stmt = stmt.where(
                MemoryRow.visibility_keys.contains(list(visibility_keys)),
                MemoryRow.visibility_keys.contained_by(list(visibility_keys)),
            )
        if not include_derived:
            source_categories = ["narrative_unit"]
            if include_verbatim:
                source_categories.append("verbatim_turn")
            stmt = stmt.where(
                _asserted_source_filter(),
                or_(
                    MemoryRow.memory_type.not_in(
                        [
                            MemoryType.BELIEF.value,
                            MemoryType.ENTITY_SUMMARY.value,
                            MemoryType.OBSERVATION.value,
                        ]
                    ),
                    MemoryRow.system_metadata["category"].astext.in_(source_categories),
                ),
                or_(MemoryRow.expires_at.is_(None), MemoryRow.expires_at > datetime.now(UTC)),
            )
        rows = (await self.s.scalars(stmt.order_by(MemoryRow.updated_at.desc()).limit(limit))).all()
        return [_to_domain(r) for r in rows]

    async def bump_access(self, tenant_id: str, memory_ids: Sequence[str], *, at: datetime) -> int:
        if not memory_ids:
            return 0
        result = await self.s.execute(
            update(MemoryRow)
            .where(
                MemoryRow.tenant_id == tenant_id,
                MemoryRow.memory_id.in_(list(memory_ids)),
                MemoryRow.deleted_at.is_(None),
            )
            .values(access_count=MemoryRow.access_count + 1, last_accessed_at=at)
        )
        return int(cast("CursorResult[Any]", result).rowcount or 0)

    async def list_idle(
        self, *, idle_before: datetime, limit: int = 500, tenant_id: str | None = None
    ) -> list[CanonicalMemory]:
        conds = [
            MemoryRow.deleted_at.is_(None),
            MemoryRow.temporal_status == TemporalStatus.CURRENT.value,
            MemoryRow.updated_at < idle_before,
            or_(MemoryRow.last_accessed_at.is_(None), MemoryRow.last_accessed_at < idle_before),
        ]
        if tenant_id is not None:
            conds.append(MemoryRow.tenant_id == tenant_id)
        rows = (
            await self.s.scalars(
                select(MemoryRow).where(*conds).order_by(MemoryRow.updated_at).limit(limit)
            )
        ).all()
        return [_to_domain(r) for r in rows]

    async def set_status(
        self, tenant_id: str, memory_id: str, status: TemporalStatus, *, now: datetime
    ) -> bool:
        r = await self.s.get(MemoryRow, memory_id)
        if r is None or r.tenant_id != tenant_id or r.deleted_at is not None:
            return False
        if r.temporal_status == status.value:
            return True
        r.temporal_status = status.value
        r.updated_at = now
        r.indexed_at = None
        r.revision += 1
        await self.s.flush()
        await self._invalidate_dependents(tenant_id, [memory_id])
        return True
