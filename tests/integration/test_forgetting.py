"""Automatic forgetting: what nobody used is archived, what months-long recall depends on is
kept, and an archived memory comes back when restored."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from benchmark.common import submit_observation

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType, TemporalStatus, Visibility
from memory_service.modules.jobs.registry import register_handlers

pytestmark = pytest.mark.integration

U1 = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
A_YEAR = timedelta(days=365)


async def _say(container, uow_factory, content: str) -> None:
    register_handlers(container)
    async with uow_factory() as uow:
        await submit_observation(uow, U1, content=content)
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()


async def _all(container, uow_factory):
    async with uow_factory() as uow:
        return await container.services["memory"].list_memories(uow, U1, include_superseded=True)


async def _remember(container, uow_factory, content: str, memory_type: MemoryType):
    async with uow_factory() as uow:
        ack = await container.services["memory"].remember(
            uow,
            U1,
            content=content,
            memory_type=memory_type,
            lifetime=Lifetime.LONG_TERM,
            visibility=Visibility.USER,
        )
        await uow.commit()
    await container.tasks.drain()
    return ack.memory_id


async def test_a_year_unused_archives_the_rest_and_keeps_what_recall_needs(
    container, uow_factory
) -> None:
    await _say(container, uow_factory, "My dog's name is Buster and he loves the beach.")
    await _say(container, uow_factory, "I live in Seattle.")
    await _say(container, uow_factory, "Never suggest recipes with cilantro.")
    note = await _remember(
        container, uow_factory, "The offsite agenda draft is v3.", MemoryType.SEMANTIC
    )

    report = await container.services["forgetting"].sweep(now=datetime.now(UTC) + A_YEAR)

    by_id = {m.memory_id: m for m in await _all(container, uow_factory)}
    assert report.archived_ids == [note], "only the unused, unprotected memory is archived"
    assert by_id[note].temporal.status is TemporalStatus.ARCHIVED
    kept = [m for m in by_id.values() if m.memory_id != note]
    assert kept and all(m.temporal.status is TemporalStatus.CURRENT for m in kept)
    categories = {m.system_metadata.get("category") for m in kept}
    assert {"verbatim_turn", "rule"} <= categories
    assert any(m.predicate == "lives_in" for m in kept)
    assert report.protected == len(kept)

    # archived leaves search; the protected turn is still found
    await container.tasks.drain()
    engine = container.services["retrieval"]
    found = {
        c.record_id
        for c in (await engine.retrieve(U1, "offsite agenda", kinds=("memory",))).candidates
    }
    assert note not in found
    dog = await engine.retrieve(U1, "what is my dog called", kinds=("memory",))
    assert any("Buster" in c.text for c in dog.candidates)


async def test_protection_can_be_turned_off(container, uow_factory) -> None:
    await _say(container, uow_factory, "My dog's name is Buster and he loves the beach.")
    forgetting = container.services["forgetting"]
    forgetting.cfg = forgetting.cfg.model_copy(update={"forgetting_protect_core": False})
    report = await forgetting.sweep(now=datetime.now(UTC) + A_YEAR)
    assert report.protected == 0 and report.archived >= 1


async def test_a_recently_used_memory_is_not_archived(container, uow_factory) -> None:
    note = await _remember(
        container, uow_factory, "The offsite agenda draft is v3.", MemoryType.SEMANTIC
    )
    report = await container.services["forgetting"].sweep(now=datetime.now(UTC) + timedelta(days=5))
    assert note not in report.archived_ids


async def test_restore_brings_an_archived_memory_back_into_search(container, uow_factory) -> None:
    note = await _remember(
        container, uow_factory, "The offsite agenda draft is v3.", MemoryType.SEMANTIC
    )
    await container.services["forgetting"].sweep(now=datetime.now(UTC) + A_YEAR)
    await container.tasks.drain()
    async with uow_factory() as uow:
        restored = await container.services["memory"].restore(
            uow, U1, note, container.services["forgetting"]
        )
        await uow.commit()
    assert restored.temporal.status is TemporalStatus.CURRENT
    assert restored.system_metadata["forgetting"]["restored_at"]
    await container.tasks.drain()
    engine = container.services["retrieval"]
    found = {
        c.record_id
        for c in (await engine.retrieve(U1, "offsite agenda", kinds=("memory",))).candidates
    }
    assert note in found
