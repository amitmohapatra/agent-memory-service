"""The process-local queue in front of ``SqlUnitOfWork.serialize``'s advisory lock.

No database here: the session is a stand-in whose ``execute`` is the advisory lock, so what
is under test is only the queue."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from memory_service.adapters.db import uow as uow_module
from memory_service.adapters.db.uow import SqlUnitOfWork

pytestmark = pytest.mark.unit


class _Session:
    """``execute`` blocks until ``release`` is set: the advisory lock, held elsewhere."""

    def __init__(self, release: asyncio.Event | None = None) -> None:
        self.release = release

    async def execute(self, *_: Any, **__: Any) -> Any:
        if self.release is not None:
            await self.release.wait()
        return None


def _unit_of_work(session: _Session) -> SqlUnitOfWork:
    uow = SqlUnitOfWork(None, None)  # type: ignore[arg-type]
    uow._session = session  # type: ignore[assignment]
    return uow


def test_a_loop_torn_down_mid_wait_does_not_hold_the_next_loops_queue(monkeypatch) -> None:
    """The queue's locks are ``asyncio.Lock``s, which belong to one event loop. A loop torn
    down while a unit of work stood at a key's head (a test's loop, a benchmark's
    ``asyncio.run``) left that key's place taken for good, and a unit of work of the next
    loop queued behind a task that would never run again. Each loop has a queue of its own.

    The local wait is made an hour long, so a unit of work of the second loop either goes
    straight to its lock or hangs."""
    monkeypatch.setattr(uow_module, "_HEAD_WAIT_SECONDS", 3600.0)
    key = "statements:acme:u1"

    first = asyncio.new_event_loop()
    try:
        stuck = _unit_of_work(_Session(release=asyncio.Event()))
        entered = first.create_task(stuck.serialize(key))
        # run until the stand-in lock is being waited for at the head, then abandon the loop
        first.run_until_complete(asyncio.sleep(0.01))
        assert not entered.done()
    finally:
        first.close()  # no clean-up runs: the task is simply gone with its loop

    async def second() -> None:
        async with asyncio.timeout(5):  # a hang, not a timing: there is nothing to wait for
            await _unit_of_work(_Session()).serialize(key)

    asyncio.run(second())
