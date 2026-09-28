"""Retention forgets only what is older than the tenant's policy, in bounded batches."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from memory_service.modules.tenancy import retention as retention_module
from memory_service.modules.tenancy.retention import RetentionService

pytestmark = pytest.mark.unit
NOW = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def _memory(memory_id: str, age_days: int) -> SimpleNamespace:
    return SimpleNamespace(memory_id=memory_id, created_at=NOW - timedelta(days=age_days))


class _Memories:
    def __init__(self, rows: dict[str, list[SimpleNamespace]]) -> None:
        self.rows, self.forgotten = rows, []

    async def list_older_than(self, tenant_id: str, *, before: datetime, limit: int = 500):
        return sorted(
            (m for m in self.rows.get(tenant_id, []) if m.created_at < before),
            key=lambda m: m.created_at,
        )[:limit]

    async def forget(self, tenant_id: str, memory_id: str) -> bool:
        self.forgotten.append((tenant_id, memory_id))
        self.rows[tenant_id] = [m for m in self.rows[tenant_id] if m.memory_id != memory_id]
        return True


class _Tenants:
    def __init__(self, policies: dict[str, int]) -> None:
        self.policies = policies

    async def retention_policies(self) -> dict[str, int]:
        return dict(self.policies)


class _Uow:
    def __init__(self, memories: _Memories, tenants: _Tenants) -> None:
        self.memories, self.tenants, self.jobs, self.commits = memories, tenants, [], 0

    async def enqueue(self, spec):  # type: ignore[no-untyped-def]
        self.jobs.append(spec)
        return len(self.jobs)

    async def commit(self) -> None:
        self.commits += 1


def _service(
    policies: dict[str, int],
    rows: dict[str, list[SimpleNamespace]],
    batch: int = 500,
    max_batches: int = 20,
):
    uow = _Uow(_Memories(rows), _Tenants(policies))

    @asynccontextmanager
    async def factory():
        yield uow

    return RetentionService(factory, batch=batch, max_batches=max_batches), uow  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _no_revision_bumps(monkeypatch: pytest.MonkeyPatch) -> None:
    async def bump(uow, memories):  # type: ignore[no-untyped-def]
        uow.bumped = [m.memory_id for m in memories]

    monkeypatch.setattr(retention_module, "bump_memory_revisions", bump)


async def test_only_memories_older_than_the_policy_are_forgotten() -> None:
    service, uow = _service(
        {"acme": 30}, {"acme": [_memory("old", 31), _memory("edge", 30), _memory("new", 1)]}
    )
    assert await service.sweep(now=NOW) == 1
    assert uow.memories.forgotten == [("acme", "old")]
    assert uow.jobs[0].payload == {"tenant_id": "acme", "memory_ids": ["old"]}
    assert uow.bumped == ["old"] and uow.commits == 1


async def test_tenants_without_a_policy_are_untouched_and_a_backlog_drains_in_batches() -> None:
    service, uow = _service(
        {"acme": 7},
        {"acme": [_memory(f"m{i}", 10) for i in range(5)], "globex": [_memory("g", 400)]},
        batch=2,
    )
    assert await service.sweep(now=NOW) == 5, "three batches in one run, each its own transaction"
    assert uow.commits == 3 and len(uow.jobs) == 3
    assert all(t == "acme" for t, _ in uow.memories.forgotten)
    assert await service.sweep(now=NOW) == 0


async def test_the_per_run_cap_bounds_one_tenant_s_share_and_the_rest_waits_for_the_next_run() -> (
    None
):
    service, _ = _service(
        {"acme": 7}, {"acme": [_memory(f"m{i}", 10) for i in range(5)]}, batch=2, max_batches=1
    )
    assert await service.sweep(now=NOW) == 2
    assert await service.sweep(now=NOW) == 2 and await service.sweep(now=NOW) == 1
    assert await service.sweep(now=NOW) == 0


async def test_one_tenant_s_failure_does_not_skip_the_next() -> None:
    service, uow = _service(
        {"aaa": 7, "zzz": 7}, {"aaa": [_memory("a", 10)], "zzz": [_memory("z", 10)]}
    )
    original = uow.memories.forget

    async def forget(tenant_id: str, memory_id: str) -> bool:
        if tenant_id == "aaa":
            raise RuntimeError("index away")
        return await original(tenant_id, memory_id)

    uow.memories.forget = forget  # type: ignore[method-assign]
    assert await service.sweep(now=NOW) == 1
    assert uow.memories.forgotten == [("zzz", "z")]


async def test_nothing_due_means_no_transaction() -> None:
    service, uow = _service({"acme": 30}, {"acme": [_memory("new", 1)]})
    assert await service.sweep(now=NOW) == 0
    assert uow.commits == 0 and uow.jobs == []
