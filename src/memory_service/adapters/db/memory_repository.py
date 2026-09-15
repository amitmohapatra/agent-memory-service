"""PostgreSQL MemoryRepository (M7)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from memory_service.adapters.db.orm import MemoryRow
from memory_service.domain.enums import (
    Lifetime,
    MemoryType,
    Representation,
    ScopeLevel,
    TemporalStatus,
    Visibility,
)
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.memory import CanonicalMemory, Scope, TemporalState


def _to_domain(r: MemoryRow) -> CanonicalMemory:
    scope = Scope(
        level=ScopeLevel(r.scope_level),
        tenant_id=r.tenant_id,
        workspace_id=r.workspace_id,
        user_id=r.user_id,
        group_id=r.group_id,
        thread_id=r.thread_id,
        work_id=r.work_id,
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
        system_metadata={
            **(r.system_metadata or {}),
            "provider": r.provider,
            "expires_at": r.expires_at.isoformat() if r.expires_at else None,
            "visibility_keys": list(r.visibility_keys or []),
        },
        custom_metadata=dict(r.custom_metadata or {}),
        created_at=r.created_at,
        updated_at=r.updated_at,
        revision=r.revision,
        deleted_at=r.deleted_at,
    )


def _evidence_json(memory: CanonicalMemory) -> list[dict[str, Any]]:
    return [json.loads(e.model_dump_json(exclude_none=True)) for e in memory.evidence]


class SqlMemoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def add(self, memory: CanonicalMemory, *, visibility_keys: Sequence[str]) -> None:
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
                group_id=memory.scope.group_id,
                thread_id=memory.scope.thread_id,
                work_id=memory.scope.work_id,
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
                provider=provider,
                system_metadata=sm,
                custom_metadata=memory.custom_metadata,
                expires_at=expires_at,
                created_at=memory.created_at,
                updated_at=memory.updated_at,
                revision=memory.revision,
            )
        )
        await self.s.flush()

    async def get(self, tenant_id: str, memory_id: str) -> CanonicalMemory | None:
        r = await self.s.get(MemoryRow, memory_id)
        if r is None or r.tenant_id != tenant_id or r.deleted_at is not None:
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
        by_id = {r.memory_id: _to_domain(r) for r in rows}
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

    async def candidates(
        self,
        tenant_id: str,
        *,
        scope_key: str,
        normalized_hash: str | None = None,
        subject: str | None = None,
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
        if subject:
            any_of.append(MemoryRow.subject == subject)
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
        recent = list(
            (
                await self.s.scalars(
                    select(MemoryRow)
                    .where(*conds)
                    .order_by(MemoryRow.updated_at.desc())
                    .limit(limit)
                )
            ).all()
        )
        seen: dict[str, MemoryRow] = {}
        for r in exact + recent:
            seen.setdefault(r.memory_id, r)
        return [_to_domain(r) for r in list(seen.values())[: max(limit, len(exact))]]

    async def list_scope(
        self,
        tenant_id: str,
        *,
        scope_keys: Sequence[str],
        memory_types: Sequence[str] | None = None,
        current_only: bool = True,
        limit: int = 200,
    ) -> list[CanonicalMemory]:
        if not scope_keys:
            return []
        stmt = select(MemoryRow).where(
            MemoryRow.tenant_id == tenant_id,
            MemoryRow.scope_key.in_(list(scope_keys)),
            MemoryRow.deleted_at.is_(None),
        )
        if current_only:
            stmt = stmt.where(MemoryRow.temporal_status == TemporalStatus.CURRENT.value)
        if memory_types:
            stmt = stmt.where(MemoryRow.memory_type.in_(list(memory_types)))
        rows = (await self.s.scalars(stmt.order_by(MemoryRow.updated_at.desc()).limit(limit))).all()
        return [_to_domain(r) for r in rows]

    async def forget(self, tenant_id: str, memory_id: str) -> bool:
        r = await self.s.get(MemoryRow, memory_id)
        if r is None or r.tenant_id != tenant_id or r.deleted_at is not None:
            return False
        r.deleted_at = datetime.now(r.created_at.tzinfo)
        r.temporal_status = TemporalStatus.RETRACTED.value
        await self.s.flush()
        return True

    async def list_unindexed(self, tenant_id: str, *, limit: int = 500) -> list[CanonicalMemory]:
        rows = (
            await self.s.scalars(
                select(MemoryRow)
                .where(
                    MemoryRow.tenant_id == tenant_id,
                    MemoryRow.indexed_at.is_(None),
                    MemoryRow.deleted_at.is_(None),
                )
                .order_by(MemoryRow.created_at)
                .limit(limit)
            )
        ).all()
        return [_to_domain(r) for r in rows]

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
        return out
