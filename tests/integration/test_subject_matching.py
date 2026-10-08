"""Same-subject matching on the write path: observation -> pipeline -> memories rows.

A subject respelled ("FORKLIFT-4", "Forklift 4"), abbreviated ("PO-4471", "purchase order
4471") or abbreviated the tenant's own way (defined once, in an earlier message) reinforces
the memory it names - also when that memory is older than the newest rows consolidation
reads, because candidates are looked up by every spelling the matcher allows. Another
identifier never does."""

from __future__ import annotations

import pytest
from benchmark.common import submit_observation

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import ObservationKind
from memory_service.modules.jobs.registry import register_handlers

pytestmark = pytest.mark.integration

OPS = MemoryExecutionContext(tenant_id="acme", user_id="ops1", workspace_id="ws1")


async def _observe(container, uow_factory, ctx, content) -> None:
    register_handlers(container)
    async with uow_factory() as uow:
        await submit_observation(uow, ctx, kind=ObservationKind.MESSAGE, content=content)
        await uow.commit()
    await container.tasks.drain()  # process_observation
    await container.tasks.drain()  # memory.index


async def _facts(container, uow_factory, ctx):
    async with uow_factory() as uow:
        mems = await container.services["memory"].list_memories(uow, ctx)
    return [m for m in mems if m.system_metadata.get("category") == "fact"]


async def test_a_respelled_subject_reinforces_even_past_the_recent_window(
    container, uow_factory
) -> None:
    await _observe(container, uow_factory, OPS, "FORKLIFT-4 uses 48V batteries.")
    # more than dedup_candidate_k newer memories: the first is out of the recency window
    k = container.tuning.memory_intelligence.dedup_candidate_k
    for i in range(k // 2 + 1):
        await _observe(container, uow_factory, OPS, f"Cooler {i + 20} is set to minus {i + 2}C.")
    await _observe(container, uow_factory, OPS, "Forklift 4 uses 48V batteries.")
    forklift = [m for m in await _facts(container, uow_factory, OPS) if "48v" in (m.object or "")]
    assert len(forklift) == 1, [m.content for m in forklift]
    assert forklift[0].reinforcement_count == 2


async def test_an_alias_reinforces_and_another_identifier_stays_apart(
    container, uow_factory
) -> None:
    for text in (
        "PO-4471 is approved.",
        "Purchase order 4471 is approved.",
        "PO-4417 is approved.",
        "Warehouse 3 is closed for inventory.",
        "Warehouse 13 is closed for inventory.",
    ):
        await _observe(container, uow_factory, OPS, text)
    facts = await _facts(container, uow_factory, OPS)
    by_subject = {m.subject: m for m in facts}
    assert set(by_subject) == {"po-4471", "po-4417", "warehouse 3", "warehouse 13"}
    assert by_subject["po-4471"].reinforcement_count == 2
    assert by_subject["po-4417"].reinforcement_count == 1
    assert by_subject["warehouse 13"].reinforcement_count == 1


async def test_an_abbreviation_the_tenant_defined_is_learned_from_its_own_memories(
    container, uow_factory
) -> None:
    await _observe(
        container, uow_factory, OPS, "All inbound goes through the cross-dock facility (CDF)."
    )
    await _observe(container, uow_factory, OPS, "Cross-dock facility door 3 is blocked.")
    await _observe(container, uow_factory, OPS, "CDF door 3 is blocked.")
    doors = [m for m in await _facts(container, uow_factory, OPS) if "door" in (m.subject or "")]
    assert len(doors) == 1 and doors[0].reinforcement_count == 2, [m.subject for m in doors]


async def test_person_facts_still_replace_their_value(container, uow_factory) -> None:
    await _observe(container, uow_factory, OPS, "My timezone is Europe/Berlin.")
    await _observe(container, uow_factory, OPS, "My timezone is America/New_York.")
    async with uow_factory() as uow:
        mems = await container.services["memory"].list_memories(uow, OPS)
    tz = [m for m in mems if m.predicate == "timezone"]
    assert [m.object for m in tz] == ["america/new_york"]
