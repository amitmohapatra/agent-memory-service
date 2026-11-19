"""Under concurrent load a process keeps its connections for work, not for waiting: a unit of
work queued behind another's lock waits without one, and many concurrent requests on a small
pool are answered without exhausting it."""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.integration


async def test_units_of_work_queued_on_one_lock_wait_without_a_connection(
    container, uow_factory
) -> None:
    """``serialize`` is an advisory lock, and the lock needs a connection to wait on. The
    memory pipeline takes one per user (``statements:<tenant>:<user>``) and a message append
    one per thread: eight jobs of one user's chat each held a pooled connection idle while
    they queued, and a pool of ten had two left for every request of the process. Waiters
    of one process now queue in the process; only the first waits in PostgreSQL."""
    pool = container.database.engine.pool
    peak = 0
    holding = asyncio.Event()

    async def writer(n: int) -> None:
        nonlocal peak
        async with uow_factory() as uow:
            await uow.serialize("statements:acme:u1")
            holding.set()
            peak = max(peak, pool.checkedout())
            await asyncio.sleep(0.05)
            await uow.commit()

    async def watch() -> None:
        nonlocal peak
        await holding.wait()
        for _ in range(40):
            peak = max(peak, pool.checkedout())
            await asyncio.sleep(0.01)

    await asyncio.gather(watch(), *(writer(n) for n in range(8)))
    assert peak <= 2, f"{peak} connections held for one lock: the holder and the next only"


async def test_the_lock_is_still_held_across_processes(container, uow_factory) -> None:
    """The process-local queue is in front of the advisory lock, never instead of it: a unit
    of work of another process (another engine) still waits for the holder."""
    from memory_service.adapters.db.engine import Database

    other = Database(container.settings.database)
    order: list[str] = []
    try:
        from memory_service.adapters.db.uow import SqlUnitOfWorkFactory

        elsewhere = SqlUnitOfWorkFactory(other.session_factory, None)

        holding = asyncio.Event()

        async def first() -> None:
            async with uow_factory() as uow:
                await uow.serialize("thread:acme/t1")
                order.append("first in")
                holding.set()
                await asyncio.sleep(0.2)
                order.append("first out")
                await uow.commit()

        async def second() -> None:
            await holding.wait()  # the lock is held: the order is not left to the scheduler
            async with elsewhere() as uow:
                await uow.serialize("thread:acme/t1")
                order.append("second in")
                await uow.commit()

        await asyncio.gather(first(), second())
    finally:
        await other.engine.dispose()
    assert order == ["first in", "first out", "second in"]


async def test_a_key_serialized_again_after_a_commit_is_locked_again(uow_factory) -> None:
    """``serialize`` skips a key its unit of work already holds (the local queue is not
    re-entrant). A commit or a rollback ends the transaction and with it every
    ``pg_advisory_xact_lock``: a key serialized after it must be locked again, not skipped."""
    from sqlalchemy import text

    held = text(
        "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND pid = pg_backend_pid()"
    )
    async with uow_factory() as uow:
        await uow.serialize("thread:acme/relock")
        assert (await uow.session.execute(held)).scalar_one() == 1
        await uow.commit()
        await uow.serialize("thread:acme/relock")
        assert (await uow.session.execute(held)).scalar_one() == 1, "skipped after the commit"
        await uow.rollback()
        await uow.serialize("thread:acme/relock")
        assert (await uow.session.execute(held)).scalar_one() == 1, "skipped after a rollback"
        await uow.commit()


async def _waiting_in_postgres(uow_factory, count: int) -> None:
    """Until ``count`` advisory locks are waited for in PostgreSQL (``pg_locks``, not
    granted). A condition, not a clock: the caller's ceiling catches a hang."""
    from sqlalchemy import text

    waiting = text("SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND NOT granted")
    while True:
        async with uow_factory() as probe:
            if (await probe.session.execute(waiting)).scalar_one() >= count:
                return
        await asyncio.sleep(0.01)


async def test_a_cross_key_deadlock_is_left_to_postgresql_to_break(
    uow_factory, monkeypatch
) -> None:
    """A holds k1 and asks for k2; B holds k2 and asks for k1; D waits for k2 in PostgreSQL
    and so holds k2's place at the head of this process's queue. A, holding a lock, used to
    queue behind D in the process, where PostgreSQL's deadlock detector cannot see it: B
    waited on A, D on B, A on nothing PostgreSQL knew of, and the cycle stood for the whole
    local wait (then the statement timeout) with three connections and two locks held. A unit
    of work that already holds a lock now waits for the next one in PostgreSQL, which breaks
    the cycle by failing one of the two within its ``deadlock_timeout``.

    The local wait is made an hour long, so only the detector can end this in time."""
    import psycopg.errors
    from sqlalchemy.exc import DBAPIError

    from memory_service.adapters.db import uow as uow_module

    monkeypatch.setattr(uow_module, "_HEAD_WAIT_SECONDS", 3600.0)
    k1, k2 = "deadlock:acme/k1", "deadlock:acme/k2"
    a_holds, b_holds, a_asked = asyncio.Event(), asyncio.Event(), asyncio.Event()
    outcome: dict[str, str] = {}

    async def run(name: str, body) -> None:  # type: ignore[no-untyped-def]
        try:
            async with uow_factory() as uow:
                await body(uow)
                await uow.commit()
            outcome[name] = "committed"
        except DBAPIError as exc:
            if not isinstance(exc.orig, psycopg.errors.DeadlockDetected):
                raise
            outcome[name] = "deadlock"

    async def a(uow) -> None:  # type: ignore[no-untyped-def]
        await uow.serialize(k1)
        a_holds.set()
        await b_holds.wait()
        await _waiting_in_postgres(uow_factory, 1)  # D waits for k2, at its queue's head
        a_asked.set()
        await uow.serialize(k2)

    async def b(uow) -> None:  # type: ignore[no-untyped-def]
        await a_holds.wait()
        await uow.serialize(k2)
        b_holds.set()
        await a_asked.wait()
        await asyncio.sleep(0)  # A is in serialize(k2): queued, or waiting in PostgreSQL
        await uow.serialize(k1)

    async def d(uow) -> None:  # type: ignore[no-untyped-def]
        await b_holds.wait()
        await uow.serialize(k2)

    async with asyncio.timeout(10):  # a hang, not a timing: the detector takes ~1 s
        await asyncio.gather(run("a", a), run("b", b), run("d", d))
    # Which of them PostgreSQL fails depends on whose detector runs first (D, queued ahead
    # of A for k2, may close a cycle of its own); that it settles, and in time, does not.
    assert set(outcome) == {"a", "b", "d"}, outcome
    assert "deadlock" in outcome.values() and "committed" in outcome.values(), outcome
    assert "deadlock" in (outcome["a"], outcome["b"]), outcome


async def test_a_unit_of_work_holding_a_connection_waits_for_a_lock_in_postgresql(
    uow_factory, monkeypatch
) -> None:
    """The local queue saves a connection only to a unit of work that has none yet. One that
    has already run a statement (``_thread_for_upload``, ``profile._writable``, key issuance)
    holds its connection while it queues, so queueing it saves nothing and only hides its
    wait from PostgreSQL: it waits in PostgreSQL, beside the head."""
    from sqlalchemy import text

    from memory_service.adapters.db import uow as uow_module

    monkeypatch.setattr(uow_module, "_HEAD_WAIT_SECONDS", 3600.0)
    key = "connected:acme/k"
    holding, done = asyncio.Event(), asyncio.Event()

    async def holder() -> None:
        async with uow_factory() as uow:
            await uow.serialize(key)
            holding.set()
            await done.wait()
            await uow.commit()

    async def waiter(*, connected: bool) -> None:
        await holding.wait()
        async with uow_factory() as uow:
            if connected:
                await uow.session.execute(text("SELECT 1"))
            await uow.serialize(key)
            await uow.commit()

    async def head_then_connected() -> None:
        first = asyncio.ensure_future(waiter(connected=False))
        await holding.wait()
        await _waiting_in_postgres(uow_factory, 1)
        second = asyncio.ensure_future(waiter(connected=True))
        try:
            await _waiting_in_postgres(uow_factory, 2)
        finally:
            done.set()
            await asyncio.gather(first, second)

    async with asyncio.timeout(10):  # a hang: the connected waiter never reaches PostgreSQL
        await asyncio.gather(holder(), head_then_connected())


async def test_the_local_wait_is_spent_from_the_lock_statement_budget(
    uow_factory, monkeypatch
) -> None:
    """A unit of work that queued in the process and then waits in PostgreSQL gets what is
    left of the statement budget there, not a whole one: the two waits together are one
    statement's timeout, as they were when the wait was in PostgreSQL alone. The queue's
    clock is replaced so that every local wait appears to take all but 50 ms of it."""
    import psycopg.errors
    from sqlalchemy.exc import DBAPIError

    from memory_service.adapters.db import uow as uow_module
    from memory_service.config.constants import DATABASE

    step = DATABASE.statement_timeout_ms / 1000 - 0.05
    reads = 0

    def clock() -> float:
        nonlocal reads
        reads += 1
        return reads * step

    monkeypatch.setattr(uow_module, "_monotonic", clock, raising=False)
    key = "budget:acme/k"
    first_in, done = asyncio.Event(), asyncio.Event()

    async def first() -> None:
        async with uow_factory() as uow:
            await uow.serialize(key)
            first_in.set()
            await done.wait()  # held until the second has given up
            await uow.commit()

    async def second() -> None:
        await first_in.wait()
        try:
            async with uow_factory() as uow:
                with pytest.raises(DBAPIError) as raised:
                    await uow.serialize(key)  # 50 ms of the budget left for PostgreSQL
            assert isinstance(raised.value.orig, psycopg.errors.QueryCanceled)
        finally:
            done.set()

    async with asyncio.timeout(10):  # a hang: a whole statement timeout is 15 s
        await asyncio.gather(first(), second())


async def test_the_local_wait_bound_follows_the_servers_deadlock_timeout(uow_factory) -> None:
    """``DATABASE.deadlock_timeout_ms`` is the server's ``deadlock_timeout``, which the local
    queue's bound (``lock_queue_wait_ms``) is twice of: kept true against the server."""
    from sqlalchemy import text

    from memory_service.config.constants import DATABASE

    async with uow_factory() as uow:
        ms = (
            await uow.session.execute(
                text("SELECT setting::int FROM pg_settings WHERE name = 'deadlock_timeout'")
            )
        ).scalar_one()
    assert ms == DATABASE.deadlock_timeout_ms
    assert DATABASE.lock_queue_wait_ms < DATABASE.statement_timeout_ms
