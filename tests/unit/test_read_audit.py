"""The read audit never costs a request anything and never takes the service down."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.audit.service import ReadAudit
from memory_service.observability.metrics import read_audit_dropped_total

pytestmark = pytest.mark.unit
CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="finance")


class _Repo:
    def __init__(self, fail: bool = False, purgeable: int = 0) -> None:
        self.batches, self.fail, self.purgeable, self.purges = [], fail, purgeable, []

    async def add_many(self, entries) -> None:  # type: ignore[no-untyped-def]
        if self.fail:
            raise RuntimeError("database away")
        self.batches.append(list(entries))

    async def purge_before(self, before, *, limit: int = 5000) -> int:  # type: ignore[no-untyped-def]
        self.purges.append((before, limit))
        n = min(limit, self.purgeable)
        self.purgeable -= n
        return n


class _Uow:
    def __init__(self, repo: _Repo) -> None:
        self.read_audit, self.commits = repo, 0

    async def commit(self) -> None:
        self.commits += 1


def _audit(repo: _Repo, **kwargs) -> ReadAudit:  # type: ignore[no-untyped-def]
    uow = _Uow(repo)

    @asynccontextmanager
    async def factory():
        yield uow

    return ReadAudit(factory, **kwargs)  # type: ignore[arg-type]


def _dropped() -> float:
    return read_audit_dropped_total._value.get()  # noqa: SLF001 - reading a counter in a test


async def test_entries_are_batched_hash_the_query_and_carry_the_scope() -> None:
    repo = _Repo()
    audit = _audit(repo, batch=2)
    for i in range(5):
        audit.record(
            CTX, "recall", f"query {i}", [f"mem_{i}"], scope_fingerprint="fp", credential="key:k1"
        )
    assert await audit.flush() == 5
    assert [len(b) for b in repo.batches] == [2, 2, 1]
    entry = repo.batches[0][0]
    assert entry.tenant_id == "acme" and entry.principal == "user:u1" and entry.kind == "recall"
    assert entry.credential == "key:k1", "the authenticated caller, not only the asserted user"
    assert entry.record_ids == ["mem_0"] and "query" not in entry.query_hash
    assert len(entry.query_hash) == 64


async def test_a_full_queue_drops_and_counts_rather_than_blocking() -> None:
    audit = _audit(_Repo(), max_pending=2)
    before = _dropped()
    for i in range(5):
        audit.record(CTX, "context", f"q{i}", (), scope_fingerprint="fp", credential="key:k1")
    assert _dropped() == before + 3
    assert await audit.flush() == 2


async def test_a_failing_store_loses_the_batch_loudly_but_never_raises() -> None:
    audit = _audit(_Repo(fail=True))
    audit.record(CTX, "recall", "q", ["m"], scope_fingerprint="fp", credential="key:k1")
    before = _dropped()
    assert await audit.flush() == 0
    assert _dropped() == before + 1


async def test_close_flushes_what_is_left() -> None:
    repo = _Repo()
    audit = _audit(repo)
    audit.start()
    audit.record(CTX, "recall", "q", ["m"], scope_fingerprint="fp", credential="key:k1")
    await audit.close()
    assert repo.batches and repo.batches[0][0].record_ids == ["m"]
    await audit.close()  # idempotent


async def test_purge_drains_in_bounded_batches_up_to_a_cap() -> None:
    from datetime import UTC, datetime, timedelta

    repo = _Repo(purgeable=7)
    audit = _audit(repo)
    assert await audit.purge(older_than_days=400, limit=3) == 7
    assert [limit for _, limit in repo.purges] == [3, 3, 3], "stops after a short batch"
    before = repo.purges[0][0]
    assert timedelta(days=399) < datetime.now(UTC) - before < timedelta(days=401)
    repo = _Repo(purgeable=10)
    assert await _audit(repo).purge(older_than_days=1, limit=3, max_batches=2) == 6, "capped"


async def test_flush_writes_a_snapshot_not_everything_that_keeps_arriving() -> None:
    repo = _Repo()
    audit = _audit(repo, batch=10)
    for i in range(3):
        audit.record(CTX, "recall", f"q{i}", [], scope_fingerprint="fp", credential="key:k1")
    original = repo.add_many

    async def add_many_and_more(entries):  # type: ignore[no-untyped-def]
        await original(entries)
        audit.record(CTX, "recall", "late", [], scope_fingerprint="fp", credential="key:k1")

    repo.add_many = add_many_and_more  # type: ignore[method-assign]
    assert await audit.flush() == 3, "the entry recorded during the write waits for the next flush"
    assert audit.queue.qsize() == 1


async def test_close_waits_for_a_write_in_flight_when_the_flusher_is_cancelled() -> None:
    import asyncio

    started, release = asyncio.Event(), asyncio.Event()

    class _Slow(_Repo):
        async def add_many(self, entries) -> None:  # type: ignore[no-untyped-def]
            started.set()
            await release.wait()
            self.batches.append(list(entries))

    repo = _Slow()
    audit = _audit(repo, flush_every=0.01)
    audit.start()
    audit.record(CTX, "recall", "q", ["m"], scope_fingerprint="fp", credential="key:k1")
    await started.wait()  # the loop task is inside the shielded write
    closer = asyncio.ensure_future(audit.close())
    await asyncio.sleep(0.05)
    assert not closer.done(), "close() returned while the batch was still being written"
    release.set()
    await closer
    assert repo.batches and repo.batches[0][0].record_ids == ["m"]


async def test_one_bad_row_does_not_sink_the_batch() -> None:
    class _Picky(_Repo):
        async def add_many(self, entries) -> None:  # type: ignore[no-untyped-def]
            if any(e.query_hash.startswith("bad") for e in entries):
                raise RuntimeError("row rejected")
            self.batches.append(list(entries))

    repo = _Picky()
    audit = _audit(repo)
    from memory_service.domain.ids import content_hash

    bad_query = next(f"q{i}" for i in range(100000) if content_hash(f"q{i}").startswith("bad"))
    audit.record(CTX, "recall", "fine 1", ["a"], scope_fingerprint="fp", credential="key:k1")
    audit.record(CTX, "recall", bad_query, ["b"], scope_fingerprint="fp", credential="key:k1")
    audit.record(CTX, "recall", "fine 2", ["c"], scope_fingerprint="fp", credential="key:k1")
    before = _dropped()
    assert await audit.flush() == 2
    assert _dropped() == before + 1, "only the rejected row is lost"
    assert sorted(b[0].record_ids[0] for b in repo.batches) == ["a", "c"]


async def test_a_long_credential_is_bounded_to_the_column() -> None:
    audit = _audit(_Repo())
    audit.record(CTX, "recall", "q", [], scope_fingerprint="fp", credential="x" * 600)
    assert len(audit.queue.get_nowait().credential) == 512
