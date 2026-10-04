"""The forgetting policy without a database (``modules.memory.forgetting``): the score, what
is protected, what a sweep archives or keeps and what it queues, restoring, and the eviction of
working memory from the cache."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from memory_service.adapters.cache.memory_cache import MemoryCache
from memory_service.config.constants import MEMORY_INTELLIGENCE
from memory_service.domain.enums import Lifetime, MemoryType, TemporalStatus
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.memory.forgetting import (
    ForgettingReport,
    ForgettingService,
    forgetting_score,
    protected,
)
from memory_service.modules.memory.pipeline import TASK_MEMORY_INDEX
from memory_service.ports.cache import CacheUnavailable
from memory_service.ports.tasks import JobSpec, Queue

pytestmark = pytest.mark.unit

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
LONG_AGO = NOW - timedelta(days=365)


def _memory(content: str = "The offsite agenda draft is v3.", **fields: Any) -> CanonicalMemory:
    base: dict[str, Any] = {
        "tenant_id": "acme",
        "content": content,
        "memory_type": MemoryType.SEMANTIC,
        "lifetime": "LONG_TERM",
        "visibility": "USER",
        "scope": {"level": "USER", "tenant_id": "acme", "user_id": "u1"},
        "owner_principal": "user:u1",
        "normalized_hash": content,
        "temporal": {"observed_at": LONG_AGO},
        "evidence": [{"source_type": "message", "source_id": "msg_1", "observed_at": LONG_AGO}],
        "created_at": LONG_AGO,
        "updated_at": LONG_AGO,
    }
    return CanonicalMemory.model_validate({**base, **fields})


class _Memories:
    def __init__(self, rows: list[CanonicalMemory]) -> None:
        self.rows = {m.memory_id: m for m in rows}
        self.updated: list[CanonicalMemory] = []
        self.idle_query: dict[str, Any] = {}

    async def list_idle(self, *, idle_before: datetime, limit: int) -> list[CanonicalMemory]:
        self.idle_query = {"idle_before": idle_before, "limit": limit}
        return list(self.rows.values())

    async def get(self, tenant_id: str, memory_id: str) -> CanonicalMemory | None:
        m = self.rows.get(memory_id)
        return m if m is not None and m.tenant_id == tenant_id else None

    async def update(self, memory: CanonicalMemory) -> None:
        self.updated.append(memory.model_copy(deep=True))


class _Revisions:
    def __init__(self) -> None:
        self.bumped: list[tuple[str, RevisionKind, str]] = []

    async def bump(self, tenant_id: str, kind: RevisionKind, object_id: str = "") -> int:
        self.bumped.append((tenant_id, kind, object_id))
        return len(self.bumped)


class _Uow:
    def __init__(self, rows: list[CanonicalMemory]) -> None:
        self.memories = _Memories(rows)
        self.revisions = _Revisions()
        self.jobs: list[JobSpec] = []
        self.commits = 0

    async def __aenter__(self) -> _Uow:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def enqueue(self, spec: JobSpec) -> None:
        self.jobs.append(spec)

    async def commit(self) -> None:
        self.commits += 1


def _service(uow: _Uow, *, cache: Any = None, **cfg: Any) -> ForgettingService:
    settings = MEMORY_INTELLIGENCE.model_copy(update=cfg) if cfg else MEMORY_INTELLIGENCE
    return ForgettingService(lambda: uow, settings=settings, cache=cache)  # type: ignore[arg-type,return-value]


# --------------------------------------------------------------------------- score


def test_the_score_is_importance_times_recency_times_access() -> None:
    fresh = _memory(importance=0.8, access_count=1, reinforcement_count=1, updated_at=NOW)
    # idle 0 days, 2 touches: 0.8 * 1 * (1 - 0.25)
    assert forgetting_score(fresh, now=NOW, half_life_days=30) == 0.6


def test_the_score_halves_with_every_half_life_idle() -> None:
    idle = _memory(importance=1.0, access_count=0, reinforcement_count=1)
    idle.created_at = NOW - timedelta(days=30)
    idle.updated_at = NOW - timedelta(days=30)
    assert forgetting_score(idle, now=NOW, half_life_days=30) == 0.25
    assert forgetting_score(idle, now=NOW, half_life_days=15) == 0.125


def test_the_latest_touch_counts_whether_an_update_or_a_recall() -> None:
    recalled = _memory(importance=1.0, last_accessed_at=NOW, access_count=0)
    assert forgetting_score(recalled, now=NOW, half_life_days=30) == 0.5


def test_a_touch_in_the_future_counts_as_no_idle_time() -> None:
    ahead = _memory(importance=1.0, updated_at=NOW + timedelta(days=3), access_count=0)
    assert forgetting_score(ahead, now=NOW, half_life_days=30) == 0.5


def test_every_access_and_reinforcement_raises_the_score() -> None:
    once = _memory(importance=1.0, updated_at=NOW, reinforcement_count=1)
    often = _memory(importance=1.0, updated_at=NOW, reinforcement_count=3, access_count=2)
    assert forgetting_score(once, now=NOW, half_life_days=30) == 0.5
    assert forgetting_score(often, now=NOW, half_life_days=30) == 0.96875


def test_the_service_scores_with_its_half_life_and_the_clock() -> None:
    service = _service(_Uow([]), forgetting_half_life_days=1.0)
    m = _memory(importance=1.0, access_count=0)
    m.updated_at = NOW - timedelta(days=1)
    m.created_at = m.updated_at
    assert service.score(m, now=NOW) == 0.25
    assert service.score(m) < 0.25, "a day later still, by the wall clock"


# --------------------------------------------------------------------------- protected


@pytest.mark.parametrize("category", ["verbatim_turn", "rule"])
def test_verbatim_turns_and_rules_are_protected(category: str) -> None:
    assert protected(_memory(system_metadata={"category": category}))


@pytest.mark.parametrize("memory_type", [MemoryType.USER, MemoryType.PREFERENCE])
def test_what_the_user_said_about_themselves_is_protected_when_long_term(
    memory_type: MemoryType,
) -> None:
    assert protected(_memory(memory_type=memory_type, lifetime=Lifetime.LONG_TERM))
    assert not protected(_memory(memory_type=memory_type, lifetime=Lifetime.SHORT_TERM))


def test_an_ordinary_memory_is_not_protected() -> None:
    assert not protected(_memory(system_metadata={"category": "attribute"}))


# --------------------------------------------------------------------------- sweep


async def test_a_sweep_archives_the_unused_keeps_the_rest_and_queues_a_reindex() -> None:
    stale = _memory("Stale note.", importance=0.1)
    useful = _memory("Useful note.", importance=1.0, access_count=20)
    useful.updated_at = NOW - timedelta(days=31)
    core = _memory("I am vegetarian.", memory_type=MemoryType.PREFERENCE, importance=0.0)
    uow = _Uow([stale, useful, core])
    stale_score = forgetting_score(stale, now=NOW, half_life_days=30)

    report = await _service(uow).sweep(now=NOW)

    assert report.scanned == 3
    assert report.archived_ids == [stale.memory_id] and report.archived == 1
    assert (report.kept, report.protected) == (2, 1)
    assert report.threshold == MEMORY_INTELLIGENCE.forgetting_archive_threshold
    assert report.min_score == stale_score
    assert uow.memories.idle_query == {
        "idle_before": NOW - timedelta(days=MEMORY_INTELLIGENCE.forgetting_min_idle_days),
        "limit": MEMORY_INTELLIGENCE.forgetting_batch,
    }
    [archived] = uow.memories.updated
    assert archived.memory_id == stale.memory_id
    assert archived.temporal.status is TemporalStatus.ARCHIVED
    assert archived.updated_at == NOW
    assert archived.system_metadata["forgetting"] == {
        "score": report.min_score,
        "threshold": MEMORY_INTELLIGENCE.forgetting_archive_threshold,
        "archived_at": NOW.isoformat(),
        "restored_at": None,
    }
    [job] = uow.jobs
    assert job.task_name == TASK_MEMORY_INDEX and job.queue is Queue.EMBEDDING
    assert job.payload == {"tenant_id": "acme", "memory_ids": [stale.memory_id]}
    assert job.tenant_id == "acme"
    assert ("acme", RevisionKind.USER, "u1") in uow.revisions.bumped
    assert uow.commits == 1
    assert report.as_dict()["archived_ids"] == [stale.memory_id]


async def test_a_sweep_queues_one_reindex_per_tenant() -> None:
    a = _memory("A.", importance=0.0)
    b = _memory("B.", importance=0.0)
    other = _memory(
        "C.",
        importance=0.0,
        tenant_id="globex",
        scope={"level": "USER", "tenant_id": "globex", "user_id": "u9"},
    )
    uow = _Uow([a, b, other])
    report = await _service(uow).sweep(now=NOW)
    assert report.archived == 3
    jobs = {job.payload["tenant_id"]: job.payload["memory_ids"] for job in uow.jobs}
    assert jobs == {"acme": sorted([a.memory_id, b.memory_id]), "globex": [other.memory_id]}


async def test_without_core_protection_a_preference_is_scored_like_anything_else() -> None:
    core = _memory("I am vegetarian.", memory_type=MemoryType.PREFERENCE, importance=0.0)
    uow = _Uow([core])
    report = await _service(uow, forgetting_protect_core=False).sweep(now=NOW)
    assert report.protected == 0 and report.archived_ids == [core.memory_id]


async def test_a_sweep_with_nothing_idle_changes_nothing() -> None:
    uow = _Uow([])
    report = await _service(uow).sweep(now=NOW)
    assert report == ForgettingReport(threshold=MEMORY_INTELLIGENCE.forgetting_archive_threshold)
    assert uow.jobs == [] and uow.memories.updated == [] and uow.revisions.bumped == []


async def test_a_sweep_without_a_clock_runs_at_the_current_time() -> None:
    uow = _Uow([])
    before = datetime.now(UTC)
    await _service(uow).sweep()
    idle_before = uow.memories.idle_query["idle_before"]
    window = timedelta(days=MEMORY_INTELLIGENCE.forgetting_min_idle_days)
    assert before - window <= idle_before <= datetime.now(UTC) - window


async def test_a_sweep_reports_the_working_memory_it_evicted() -> None:
    cache = MemoryCache()
    await cache.list_push("wm:acme:thr_1", _item(NOW - timedelta(days=1), importance=0.1))
    report = await _service(_Uow([]), cache=cache).sweep(now=NOW)
    assert report.evicted_working == 1 and report.archived == 0


# --------------------------------------------------------------------------- restore


async def test_restoring_brings_an_archived_memory_back_and_queues_its_reindex() -> None:
    m = _memory(
        temporal={"observed_at": LONG_AGO, "status": "ARCHIVED"},
        system_metadata={"forgetting": {"score": 0.01, "restored_at": None}},
        access_count=2,
    )
    uow = _Uow([m])
    restored = await _service(uow).restore(uow, "acme", m.memory_id, now=NOW)  # type: ignore[arg-type]
    assert restored is not None
    assert restored.temporal.status is TemporalStatus.CURRENT
    assert restored.system_metadata["forgetting"] == {
        "score": 0.01,
        "restored_at": NOW.isoformat(),
    }
    assert (restored.last_accessed_at, restored.updated_at) == (NOW, NOW)
    assert restored.access_count == 3
    assert [u.memory_id for u in uow.memories.updated] == [m.memory_id]
    [job] = uow.jobs
    assert job.task_name == TASK_MEMORY_INDEX
    assert job.payload == {"tenant_id": "acme", "memory_ids": [m.memory_id]}
    assert ("acme", RevisionKind.USER, "u1") in uow.revisions.bumped


async def test_restoring_an_archived_memory_without_a_forgetting_record_starts_one() -> None:
    m = _memory(temporal={"observed_at": LONG_AGO, "status": "ARCHIVED"})
    uow = _Uow([m])
    restored = await _service(uow).restore(uow, "acme", m.memory_id)  # type: ignore[arg-type]
    assert restored is not None
    assert set(restored.system_metadata["forgetting"]) == {"restored_at"}


async def test_restoring_a_current_memory_changes_nothing() -> None:
    m = _memory()
    uow = _Uow([m])
    same = await _service(uow).restore(uow, "acme", m.memory_id, now=NOW)  # type: ignore[arg-type]
    assert same is not None and same.temporal.status is TemporalStatus.CURRENT
    assert uow.memories.updated == [] and uow.jobs == []


async def test_restoring_an_unknown_or_foreign_memory_returns_nothing() -> None:
    m = _memory()
    uow = _Uow([m])
    service = _service(uow)
    assert await service.restore(uow, "acme", "mem_missing", now=NOW) is None  # type: ignore[arg-type]
    assert await service.restore(uow, "globex", m.memory_id, now=NOW) is None  # type: ignore[arg-type]
    assert uow.jobs == []


# --------------------------------------------------------------------------- working memory


def _item(at: datetime | str | None, *, importance: float | None = 0.5, **extra: Any) -> bytes:
    body: dict[str, Any] = {"content": "brb", **extra}
    if at is not None:
        body["at"] = at.isoformat() if isinstance(at, datetime) else at
    if importance is not None:
        body["importance"] = importance
    return json.dumps(body).encode()


async def test_without_a_cache_there_is_no_working_memory_to_evict() -> None:
    assert await _service(_Uow([])).evict_working(now=NOW) == 0


async def test_stale_unimportant_working_items_are_evicted_and_the_rest_rewritten() -> None:
    cache = MemoryCache()
    key = "wm:acme:thr_1"
    fresh = _item(NOW, importance=0.9)
    stale = _item(NOW - timedelta(hours=6), importance=0.2)
    unreadable = b"not json"
    undated = _item(None)
    default_importance = _item(NOW, importance=None)
    await cache.list_push(key, fresh, stale, unreadable, undated, default_importance)
    await cache.set("wm:not-a-list", b"x")  # a scanned key that holds no list
    service = ForgettingService(
        lambda: _Uow([]),  # type: ignore[arg-type,return-value]
        settings=MEMORY_INTELLIGENCE,
        cache=cache,
        working_ttl_seconds=1800,
    )

    assert await service.evict_working(now=NOW) == 1
    assert await cache.list_range(key) == [fresh, unreadable, undated, default_importance]


async def test_a_list_with_nothing_to_evict_is_left_as_it_is() -> None:
    class _Watching(MemoryCache):
        def __init__(self) -> None:
            super().__init__()
            self.deleted: list[str] = []

        async def delete(self, *keys: str) -> int:
            self.deleted.extend(keys)
            return await super().delete(*keys)

    cache = _Watching()
    await cache.list_push("wm:acme:run_1", _item(NOW, importance=1.0))
    assert await _service(_Uow([]), cache=cache).evict_working(now=NOW) == 0
    assert cache.deleted == []
    assert len(await cache.list_range("wm:acme:run_1")) == 1


async def test_a_cache_outage_evicts_nothing_and_raises_nothing() -> None:
    cache = MemoryCache()
    await cache.list_push("wm:acme:thr_1", _item(NOW - timedelta(days=1), importance=0.1))
    cache.available = False
    assert await _service(_Uow([]), cache=cache).evict_working(now=NOW) == 0


async def test_a_cache_that_cannot_scan_evicts_nothing() -> None:
    class _NoScan:
        def scan(self, pattern: str) -> AsyncIterator[str]:
            raise NotImplementedError

    assert await _service(_Uow([]), cache=_NoScan()).evict_working(now=NOW) == 0
    assert await _service(_Uow([]), cache=object()).evict_working(now=NOW) == 0


async def test_an_outage_on_one_list_does_not_stop_the_others() -> None:
    class _Flaky(MemoryCache):
        async def list_range(self, key: str, start: int = 0, stop: int = -1) -> list[bytes]:
            if key.endswith("broken"):
                raise CacheUnavailable("one shard down")
            return await super().list_range(key, start, stop)

    cache = _Flaky()
    await cache.list_push("wm:acme:broken", _item(NOW - timedelta(days=1), importance=0.1))
    await cache.list_push("wm:acme:ok", _item(NOW - timedelta(days=1), importance=0.1))
    assert await _service(_Uow([]), cache=cache).evict_working(now=NOW) == 1
    assert await cache.list_range("wm:acme:ok") == []
