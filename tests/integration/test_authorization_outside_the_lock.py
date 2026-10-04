"""Authorization is asked before the thread's lock and the write's connection (ADR 0031).

``append_message`` used to call OpenFGA after taking the per-thread advisory lock, so every
writer of a thread queued behind one authorization round trip while holding a pooled
connection and the lock."""

from __future__ import annotations

import asyncio

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import MessageRole
from memory_service.domain.errors import ScopeDenied
from memory_service.domain.ids import new_id
from tests.integration.conftest import requires_pg

pytestmark = [pytest.mark.integration, requires_pg]


def _ctx(user: str, thread_id: str) -> MemoryExecutionContext:
    return MemoryExecutionContext(
        tenant_id="acme", user_id=user, workspace_id="ws1", thread_id=thread_id
    )


async def test_an_authorized_writer_asks_before_the_lock(container, uow_factory) -> None:
    service = container.services["conversation"]
    authz = container.services["authz"]
    owner = _ctx("u-owner", new_id("thread"))
    async with uow_factory() as uow:
        await service.create_thread(uow, owner)
        await uow.commit()
    order: list[str] = []
    allowed = authz.allowed

    async def spy_allowed(*args, **kwargs):  # type: ignore[no-untyped-def]
        order.append("authz")
        return await allowed(*args, **kwargs)

    authz.allowed = spy_allowed
    may_write = await service.may_write_thread(owner, owner.thread_id)
    async with uow_factory() as uow:
        serialize = uow.serialize

        async def spy_serialize(*keys: str) -> None:
            order.append("lock")
            await serialize(*keys)

        uow.serialize = spy_serialize  # type: ignore[method-assign]
        await service.append_message(
            uow, owner, role=MessageRole.USER, content="hello", may_write=may_write
        )
        await uow.commit()
    assert order == ["authz", "lock"], "authorization ran under the lock"


async def test_a_writer_without_access_is_still_refused(container, uow_factory) -> None:
    service = container.services["conversation"]
    owner = _ctx("u-owner", new_id("thread"))
    async with uow_factory() as uow:
        await service.create_thread(uow, owner)
        await uow.commit()
    stranger = _ctx("u-stranger", owner.thread_id or "")
    may_write = await service.may_write_thread(stranger, owner.thread_id)
    assert may_write is False
    with pytest.raises(ScopeDenied):
        async with uow_factory() as uow:
            await service.append_message(
                uow, stranger, role=MessageRole.USER, content="let me in", may_write=may_write
            )


async def test_concurrent_first_messages_to_a_new_thread_both_land(container, uow_factory) -> None:
    """Both pre-checks run before the thread exists and answer no; the one that waits for the
    lock finds the thread its twin created and asks again rather than refusing."""
    service = container.services["conversation"]
    ctx = _ctx("u-owner", new_id("thread"))
    checks = await asyncio.gather(
        service.may_write_thread(ctx, ctx.thread_id), service.may_write_thread(ctx, ctx.thread_id)
    )
    assert checks == [False, False]

    async def write(text: str, may_write: bool) -> None:
        async with uow_factory() as uow:
            await service.append_message(
                uow, ctx, role=MessageRole.USER, content=text, may_write=may_write
            )
            await uow.commit()

    await asyncio.gather(write("one", checks[0]), write("two", checks[1]))
    async with uow_factory() as uow:
        messages = await service.list_messages(uow, ctx, ctx.thread_id or "", limit=10)
    assert sorted(m.content for m in messages) == ["one", "two"]
