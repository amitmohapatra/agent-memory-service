"""Concurrent forgets of related memories on real PostgreSQL.

A forget removes the memory and its twins (the same statement kept twice) and reinstates what
it replaced, so two forgets of related memories touch the same rows. The capability eval's
purge forgets up to 16 memories at a time across users, and it failed with a deadlock on the
``memories`` update and with ``memory ... is not visible in this scope`` for a memory a twin's
forget had just removed. Each forget here runs in its own transaction, as each
``DELETE /v1/memories/{id}`` does.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import text

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import (
    Lifetime,
    MemoryType,
    ScopeLevel,
    TemporalStatus,
    Visibility,
)
from memory_service.domain.errors import ScopeDenied
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.memory import CanonicalMemory, Scope, TemporalState
from memory_service.modules.authz.visibility import readable_by
from memory_service.modules.memory.native import normalized_hash

pytestmark = pytest.mark.integration

TENANT = "acme"
NOW = datetime.now(UTC)


def _ctx(user: str) -> MemoryExecutionContext:
    return MemoryExecutionContext(tenant_id=TENANT, user_id=user, thread_id=f"{user}-t1")


def _memory(
    memory_id: str,
    user: str,
    content: str,
    *,
    source: str,
    level: ScopeLevel = ScopeLevel.USER,
    memory_type: MemoryType = MemoryType.EPISODIC,
    observed: datetime = NOW,
    **changes: Any,
) -> CanonicalMemory:
    scope = Scope(
        tenant_id=TENANT,
        level=level,
        user_id=user,
        thread_id=f"{user}-t1" if level is ScopeLevel.THREAD else None,
    )
    return CanonicalMemory(
        memory_id=memory_id,
        tenant_id=TENANT,
        scope=scope,
        visibility=Visibility.USER,
        owner_principal=f"user:{user}",
        lifetime=Lifetime.LONG_TERM,
        memory_type=memory_type,
        content=content,
        normalized_hash=normalized_hash(content),
        temporal=TemporalState(observed_at=observed, valid_from=observed),
        evidence=[EvidenceRef(source_type="message", source_id=source, observed_at=observed)],
    ).model_copy(update=changes)


def _family(user: str) -> list[CanonicalMemory]:
    """One user's related memories: a statement kept three times (twins: the turn, and two
    readings of it in its words), and a replaced chain - an old value superseded by a new
    statement that is itself kept twice."""
    said = "I moved to Lisbon last spring."
    old, new = "My team is Corrimax.", "My team is Brightway now."
    earlier = NOW - timedelta(days=3)
    newer = (f"mem_{user}_new", f"mem_{user}_new2")
    return [
        _memory(f"mem_{user}_twin_a", user, said, source=f"msg_{user}_1"),
        _memory(
            f"mem_{user}_twin_b",
            user,
            said,
            source=f"msg_{user}_1",
            level=ScopeLevel.THREAD,
            memory_type=MemoryType.SEMANTIC,
        ),
        _memory(
            f"mem_{user}_twin_c",
            user,
            said,
            source=f"msg_{user}_1",
            memory_type=MemoryType.SEMANTIC,
        ),
        _memory(
            f"mem_{user}_old",
            user,
            old,
            source=f"msg_{user}_0",
            observed=earlier,
            temporal=TemporalState(
                observed_at=earlier,
                valid_from=earlier,
                valid_to=NOW,
                status=TemporalStatus.SUPERSEDED,
                superseded_by=newer[0],
            ),
            system_metadata={
                "replaced_by": {"memory_id": newer[0], "grade": "SUPERSEDED", "score": 0.97},
                "supersede_reason": "replaced in a later statement",
            },
        ),
        _memory(newer[0], user, new, source=f"msg_{user}_2"),
        _memory(
            newer[1],
            user,
            new,
            source=f"msg_{user}_2",
            level=ScopeLevel.THREAD,
            memory_type=MemoryType.SEMANTIC,
        ),
    ]


async def _seed(uow_factory, users: list[str]) -> dict[str, list[CanonicalMemory]]:
    families = {u: _family(u) for u in users}
    async with uow_factory() as uow:
        for memories in families.values():
            for m in memories:
                keys = readable_by(
                    TENANT, m.visibility, owner_principal=m.owner_principal, scope=m.scope
                )
                m.system_metadata["visibility_keys"] = keys
                await uow.memories.add(m, visibility_keys=keys)
        await uow.commit()
    return families


async def _forget(container, uow_factory, ctx: MemoryExecutionContext, memory_id: str) -> None:
    """One ``DELETE /v1/memories/{id}``: its own transaction."""
    async with uow_factory() as uow:
        await container.services["memory"].forget(uow, ctx, memory_id)
        await uow.commit()


async def _all_forgotten(uow_factory, ids: list[str]) -> list[str]:
    async with uow_factory() as uow:
        return [i for i in ids if not await uow.memories.is_forgotten(TENANT, i)]


async def test_parallel_forgets_of_twins_and_a_replaced_chain_across_users(
    container, uow_factory
) -> None:
    """The purge's shape: every memory of several users forgotten at once, twins and a
    replaced chain among them. None fails, and every one ends forgotten."""
    errors: list[str] = []
    sem = asyncio.Semaphore(12)  # the request pool's share a purge of 16 at a time reaches

    async def one(user: str, memory_id: str) -> None:
        async with sem:
            try:
                await _forget(container, uow_factory, _ctx(user), memory_id)
            except Exception as exc:  # noqa: BLE001 - every failure is the finding
                errors.append(f"forget {memory_id}: {type(exc).__name__}: {exc}"[:300])

    for round_ in range(5):
        families = await _seed(uow_factory, [f"r{round_}u{i}" for i in range(6)])
        await asyncio.gather(*(one(u, m.memory_id) for u, ms in families.items() for m in ms))
        ids = [m.memory_id for ms in families.values() for m in ms]
        assert errors == [], "\n".join(e.splitlines()[0] for e in errors)
        assert await _all_forgotten(uow_factory, ids) == []


def _pause_after(
    repo: Any, name: str, reached: asyncio.Event, go: asyncio.Event, *, wait: float
) -> None:
    """Make the first call of ``repo.<name>`` signal ``reached`` and then wait (at most
    ``wait`` seconds) for ``go``: one transaction's step held open while another runs."""
    original: Callable[..., Awaitable[Any]] = getattr(repo, name)
    calls = 0

    async def paused(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        result = await original(*args, **kwargs)
        calls += 1
        if calls == 1:
            reached.set()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(go.wait(), wait)
        return result

    setattr(repo, name, paused)


async def test_a_twin_forgotten_meanwhile_is_not_reported_invisible(container, uow_factory) -> None:
    """Forget A (which forgets its twin B) commits between B's own forget reading B and
    checking B's visibility. The check must not read the half-done state as "not visible
    in this scope": B is forgotten, which is what was asked."""
    families = await _seed(uow_factory, ["v1"])
    a, b = families["v1"][0].memory_id, families["v1"][1].memory_id
    ctx = _ctx("v1")
    service = container.services["memory"]
    read, committed = asyncio.Event(), asyncio.Event()

    async with uow_factory() as first:
        await service.forget(first, ctx, a)  # A and twin B gone, not yet committed

        async def second() -> None:
            async with uow_factory() as uow:
                _pause_after(uow.memories, "get", read, committed, wait=10)
                await service.forget(uow, ctx, b)
                await uow.commit()

        task = asyncio.create_task(second())
        await asyncio.wait_for(read.wait(), 10)
        await first.commit()
    committed.set()
    await task  # before the fix: ScopeDenied("memory ... is not visible in this scope")
    assert await _all_forgotten(uow_factory, [a, b]) == []


async def test_twins_forgotten_at_once_do_not_deadlock(container, uow_factory) -> None:
    """Forget A and forget B (twins) each take their own row first and then the other's:
    opposite orders, a deadlock PostgreSQL breaks by failing one of them."""
    families = await _seed(uow_factory, ["d1"])
    a, b = families["d1"][0].memory_id, families["d1"][1].memory_id
    ctx = _ctx("d1")
    service = container.services["memory"]
    a_in, b_in = asyncio.Event(), asyncio.Event()

    async def forget(memory_id: str, mine: asyncio.Event, theirs: asyncio.Event) -> None:
        async with uow_factory() as uow:
            # after its first row, each waits (briefly) until the other has its own
            _pause_after(uow.memories, "forget", mine, theirs, wait=2)
            await service.forget(uow, ctx, memory_id)
            await uow.commit()

    results = await asyncio.gather(
        forget(a, a_in, b_in), forget(b, b_in, a_in), return_exceptions=True
    )
    assert [r for r in results if isinstance(r, BaseException)] == []
    assert await _all_forgotten(uow_factory, [a, b]) == []


async def test_a_forget_and_the_reinstatement_it_races_leave_the_old_value_forgotten(
    container, uow_factory
) -> None:
    """Forgetting the new statement reinstates the old value; forgetting the old value at the
    same time must leave it forgotten, never reopened as CURRENT behind the forget."""
    families = await _seed(uow_factory, ["c1"])
    old, new = "mem_c1_old", "mem_c1_new"
    assert {old, new} <= {m.memory_id for m in families["c1"]}
    ctx = _ctx("c1")
    service = container.services["memory"]
    read, committed = asyncio.Event(), asyncio.Event()

    async def forget_new() -> None:
        async with uow_factory() as uow:
            # reads what the new statement replaced, then the old value's forget commits
            _pause_after(uow.memories, "replaced_by_any", read, committed, wait=3)
            await service.forget(uow, ctx, new)
            await uow.commit()

    task = asyncio.create_task(forget_new())
    await asyncio.wait_for(read.wait(), 10)
    forget_old = asyncio.create_task(_forget(container, uow_factory, ctx, old))
    await asyncio.sleep(0.5)
    committed.set()
    await asyncio.gather(task, forget_old)
    async with uow_factory() as uow:
        status, deleted = (
            await uow.session.execute(
                text("SELECT temporal_status, deleted_at FROM memories WHERE memory_id = :i"),
                {"i": old},
            )
        ).one()
    assert deleted is not None
    assert status == TemporalStatus.RETRACTED.value


async def test_a_memory_outside_the_callers_scope_is_still_refused(container, uow_factory) -> None:
    """The fix keeps the check strict: another user's live memory is not forgotten, it is
    refused as not visible, and stays as it was."""
    families = await _seed(uow_factory, ["s1", "s2"])
    theirs = families["s2"][0].memory_id
    with pytest.raises(ScopeDenied, match="not visible in this scope"):
        await _forget(container, uow_factory, _ctx("s1"), theirs)
    assert await _all_forgotten(uow_factory, [theirs]) == [theirs]
