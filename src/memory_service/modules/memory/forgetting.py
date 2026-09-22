"""Forgetting policy: importance x recency x access decay.

``score = importance * 0.5 ** (idle_days / half_life) * (1 - 0.5 ** (accesses + reinforcements))``

A memory that has been idle (neither updated nor recalled) for at least
``forgetting_min_idle_days`` and scores below ``forgetting_archive_threshold`` is marked
``ARCHIVED``: the row stays, evidence stays, it leaves the search index and every default
listing, and ``restore`` brings it back. Nothing is ever deleted by this policy. Working
memory (Dragonfly lists) is evicted by the same score with the cache TTL as its half-life.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from memory_service.config.constants import MemoryIntelligenceSettings
from memory_service.domain.enums import TemporalStatus
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.memory.pipeline import TASK_MEMORY_INDEX
from memory_service.observability.logging import get_logger
from memory_service.ports.cache import CacheProvider, CacheUnavailable
from memory_service.ports.tasks import JobSpec, Queue
from memory_service.ports.uow import UnitOfWork, UnitOfWorkFactory

log = get_logger(__name__)

TASK_MEMORY_FORGET = "memory.forget"


def forgetting_score(memory: CanonicalMemory, *, now: datetime, half_life_days: float) -> float:
    touched = max(
        [t for t in (memory.updated_at, memory.last_accessed_at, memory.created_at) if t],
        default=now,
    )
    idle_days = max(0.0, (now - touched).total_seconds() / 86_400)
    recency = 0.5 ** (idle_days / half_life_days)
    access = 1.0 - 0.5 ** (memory.access_count + memory.reinforcement_count)
    return round(max(memory.importance, 0.0) * recency * access, 6)


@dataclass
class ForgettingReport:
    scanned: int = 0
    archived: int = 0
    kept: int = 0
    evicted_working: int = 0
    min_score: float | None = None
    threshold: float = 0.0
    archived_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ForgettingService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        *,
        settings: MemoryIntelligenceSettings,
        cache: CacheProvider | None = None,
        working_ttl_seconds: int = 1800,
    ) -> None:
        self.uow_factory = uow_factory
        self.cfg = settings
        self.cache = cache
        self.working_ttl = working_ttl_seconds

    def score(self, memory: CanonicalMemory, *, now: datetime | None = None) -> float:
        return forgetting_score(
            memory, now=now or datetime.now(UTC), half_life_days=self.cfg.forgetting_half_life_days
        )

    async def sweep(self, *, now: datetime | None = None) -> ForgettingReport:
        now = now or datetime.now(UTC)
        report = ForgettingReport(threshold=self.cfg.forgetting_archive_threshold)
        idle_before = now - timedelta(days=self.cfg.forgetting_min_idle_days)
        by_tenant: dict[str, list[str]] = {}
        touched: set[tuple[str, RevisionKind, str]] = set()
        async with self.uow_factory() as uow:
            rows = await uow.memories.list_idle(
                idle_before=idle_before, limit=self.cfg.forgetting_batch
            )
            report.scanned = len(rows)
            for m in rows:
                s = self.score(m, now=now)
                report.min_score = s if report.min_score is None else min(report.min_score, s)
                if s >= self.cfg.forgetting_archive_threshold:
                    report.kept += 1
                    continue
                await self._archive(uow, m, score=s, now=now)
                by_tenant.setdefault(m.tenant_id, []).append(m.memory_id)
                report.archived_ids.append(m.memory_id)
                for kind, ident in (
                    (RevisionKind.USER, m.scope.user_id),
                    (RevisionKind.THREAD, m.scope.thread_id),
                    (RevisionKind.AGENT, m.scope.agent_id),
                ):
                    if ident:
                        touched.add((m.tenant_id, kind, ident))
            for tenant_id, ids in by_tenant.items():
                await uow.enqueue(
                    JobSpec(
                        task_name=TASK_MEMORY_INDEX,
                        queue=Queue.EMBEDDING,
                        payload={"tenant_id": tenant_id, "memory_ids": sorted(ids)},
                        idempotency_key=f"memidx:forget:{tenant_id}:{now.timestamp():.0f}",
                        tenant_id=tenant_id,
                    )
                )
            for tenant_id, kind, ident in sorted(touched):
                await uow.revisions.bump(tenant_id, kind, ident)
            await uow.commit()
        report.archived = len(report.archived_ids)
        report.evicted_working = await self.evict_working(now=now)
        if report.archived or report.evicted_working:
            log.info(
                "memory.forgotten",
                archived=report.archived,
                evicted_working=report.evicted_working,
                scanned=report.scanned,
            )
        return report

    async def _archive(
        self, uow: UnitOfWork, memory: CanonicalMemory, *, score: float, now: datetime
    ) -> None:
        memory.temporal = memory.temporal.model_copy(update={"status": TemporalStatus.ARCHIVED})
        memory.system_metadata["forgetting"] = {
            "score": score,
            "threshold": self.cfg.forgetting_archive_threshold,
            "archived_at": now.isoformat(),
            "restored_at": None,
        }
        memory.updated_at = now
        await uow.memories.update(memory)

    async def restore(
        self, uow: UnitOfWork, tenant_id: str, memory_id: str, *, now: datetime | None = None
    ) -> CanonicalMemory | None:
        """Bring an archived memory back into circulation (CURRENT, re-indexed)."""
        now = now or datetime.now(UTC)
        memory = await uow.memories.get(tenant_id, memory_id)
        if memory is None or memory.temporal.status is not TemporalStatus.ARCHIVED:
            return memory
        memory.temporal = memory.temporal.model_copy(update={"status": TemporalStatus.CURRENT})
        record = dict(memory.system_metadata.get("forgetting") or {})
        record["restored_at"] = now.isoformat()
        memory.system_metadata["forgetting"] = record
        memory.last_accessed_at = now
        memory.access_count += 1
        memory.updated_at = now
        await uow.memories.update(memory)
        await uow.enqueue(
            JobSpec(
                task_name=TASK_MEMORY_INDEX,
                queue=Queue.EMBEDDING,
                payload={"tenant_id": tenant_id, "memory_ids": [memory_id]},
                idempotency_key=f"memidx:restore:{memory_id}:{now.timestamp():.0f}",
                tenant_id=tenant_id,
            )
        )
        for kind, ident in (
            (RevisionKind.USER, memory.scope.user_id),
            (RevisionKind.THREAD, memory.scope.thread_id),
            (RevisionKind.AGENT, memory.scope.agent_id),
        ):
            if ident:
                await uow.revisions.bump(tenant_id, kind, ident)
        return memory

    async def evict_working(self, *, now: datetime | None = None) -> int:
        """Drop working-memory items whose importance x recency fell below the threshold."""
        if self.cache is None:
            return 0
        now = now or datetime.now(UTC)
        evicted = 0
        try:
            keys = [k async for k in self.cache.scan("wm:*")]
        except (CacheUnavailable, NotImplementedError, AttributeError):
            return 0
        for key in keys:
            with contextlib.suppress(CacheUnavailable):
                raw = await self.cache.list_range(key)
                keep: list[bytes] = []
                for item in raw:
                    try:
                        d = json.loads(item)
                        at = datetime.fromisoformat(str(d.get("at")))
                    except (ValueError, TypeError):
                        keep.append(item)
                        continue
                    idle = max(0.0, (now - at).total_seconds())
                    recency = 0.5 ** (idle / max(1.0, self.working_ttl / 2))
                    if (
                        float(d.get("importance", 0.5)) * recency
                        < self.cfg.forgetting_archive_threshold
                    ):
                        evicted += 1
                    else:
                        keep.append(item)
                if len(keep) != len(raw):
                    await self.cache.delete(key)
                    for item in keep:
                        await self.cache.list_push(
                            key, item, max_len=len(keep), ttl_seconds=self.working_ttl
                        )
        return evicted
