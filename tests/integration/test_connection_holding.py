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

        async def first() -> None:
            async with uow_factory() as uow:
                await uow.serialize("thread:acme/t1")
                order.append("first in")
                await asyncio.sleep(0.2)
                order.append("first out")
                await uow.commit()

        async def second() -> None:
            await asyncio.sleep(0.05)
            async with elsewhere() as uow:
                await uow.serialize("thread:acme/t1")
                order.append("second in")
                await uow.commit()

        await asyncio.gather(first(), second())
    finally:
        await other.engine.dispose()
    assert order == ["first in", "first out", "second in"]
