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
from memory_service.domain.enums import Lifetime, MemoryType, ObservationKind
from memory_service.domain.subjects import spellings
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


async def _push_out_of_the_window(container, uow_factory, ctx, first_content: str) -> None:
    """More than ``dedup_candidate_k`` newer facts in the scope, then proof: the newest-k
    window consolidation reads no longer holds the fact said first, so only the subject
    lookup can find it."""
    k = container.tuning.memory_intelligence.dedup_candidate_k
    for i in range(k + 2):
        await _observe(container, uow_factory, ctx, f"Cooler {i + 20} is set to minus {i + 2}C.")
    async with uow_factory() as uow:
        rows = await container.services["memory"].list_memories(uow, ctx)
    # the window is per kind: the newest k memories that are not a verbatim turn
    readings = [m for m in rows if m.system_metadata.get("category") != "verbatim_turn"]
    [first] = [m for m in readings if m.content == first_content]
    newer = [m for m in readings if m.updated_at > first.updated_at]
    assert len(newer) > k, "still in the window"


async def _facts(container, uow_factory, ctx):
    async with uow_factory() as uow:
        mems = await container.services["memory"].list_memories(uow, ctx)
    return [m for m in mems if m.system_metadata.get("category") == "fact"]


async def test_a_respelled_subject_reinforces_even_past_the_recent_window(
    container, uow_factory
) -> None:
    await _observe(container, uow_factory, OPS, "FORKLIFT-4 uses 48V batteries.")
    # k // 2 + 1 newer facts left it inside the window (each kind has a window of its own,
    # ADR 0036), so this test passed with the lookup switched off: now it really leaves
    await _push_out_of_the_window(container, uow_factory, OPS, "FORKLIFT-4 uses 48V batteries.")
    await _observe(container, uow_factory, OPS, "Forklift 4 uses 48V batteries.")
    forklift = [m for m in await _facts(container, uow_factory, OPS) if "48v" in (m.object or "")]
    assert len(forklift) == 1, [m.content for m in forklift]
    assert forklift[0].reinforcement_count == 2


@pytest.mark.parametrize(
    ("first", "again", "other"),
    [
        (
            "Forklift-4 uses 48V batteries.",
            "Forklift 4 uses 48V batteries.",
            "Forklift 3 uses 48V batteries.",
        ),
        (
            "SKU1001 is stored in Leipzig.",
            "SKU 1001 is stored in Leipzig.",
            "SKU 1002 is stored in Leipzig.",
        ),
    ],
)
async def test_a_punctuation_respelling_finds_a_fact_outside_the_recent_window(
    container, uow_factory, first: str, again: str, other: str
) -> None:
    """The fact rule stores its subject lower-cased ("forklift-4"), so the later statement
    asks for "forklift 4" as written: only the spelling lookup can find the earlier one once
    more than ``dedup_candidate_k`` newer facts (and turns) push it out of the window of
    recent rows consolidation reads. Another identifier, written the same way, stays apart."""
    await _observe(container, uow_factory, OPS, first)
    await _push_out_of_the_window(container, uow_factory, OPS, first)
    await _observe(container, uow_factory, OPS, again)
    await _observe(container, uow_factory, OPS, other)
    facts = await _facts(container, uow_factory, OPS)
    same = [m for m in facts if m.content in (first, again)]
    assert len(same) == 1, [(m.subject, m.content) for m in same]
    assert same[0].reinforcement_count == 2
    apart = [m for m in facts if m.content == other]
    assert len(apart) == 1 and apart[0].reinforcement_count == 1, [m.subject for m in apart]


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


async def test_a_person_fact_past_the_window_is_found_by_its_mixed_case_id(
    container, uow_factory
) -> None:
    """A principal id keeps its case ("user:Alice"): the subject lookup must ask for it as
    written, or an old fact outside the newest rows is never found and never replaced."""
    alice = MemoryExecutionContext(tenant_id="acme", user_id="Alice", workspace_id="ws1")
    await _observe(container, uow_factory, alice, "My timezone is Europe/Berlin.")
    await _push_out_of_the_window(container, uow_factory, alice, "My timezone is Europe/Berlin.")
    await _observe(container, uow_factory, alice, "My timezone is America/New_York.")
    async with uow_factory() as uow:
        mems = await container.services["memory"].list_memories(uow, alice)
    assert [m.object for m in mems if m.predicate == "timezone"] == ["america/new_york"]


async def test_a_mixed_case_subject_is_found_past_the_window_whatever_the_case(
    container, uow_factory
) -> None:
    """A subject stored as written ("Forklift 4", through the API) is found by a spelling in
    any case and join ("forklift-4", "FORKLIFT4"): both sides are compared case-folded.
    Another identifier is not, and an identity matches only as written."""
    memory = container.services["memory"]
    k = container.tuning.memory_intelligence.dedup_candidate_k

    async def remember(content: str, subject: str) -> str:
        async with uow_factory() as uow:
            ack = await memory.remember(
                uow,
                OPS,
                content=content,
                memory_type=MemoryType.SEMANTIC,
                lifetime=Lifetime.LONG_TERM,
                subject=subject,
            )
            await uow.commit()
        return ack.memory_id

    first = await remember("Forklift 4 is due for service in May.", "Forklift 4")
    for i in range(k + 2):
        await remember(f"Cooler {i + 20} is set to minus {i + 2}C.", f"Cooler {i + 20}")
    async with uow_factory() as uow:
        stored = await uow.memories.get("acme", first)
        assert stored is not None and stored.subject == "Forklift 4"
        scope_key = stored.scope.key()

        async def found(asked: str) -> bool:
            got = await uow.memories.candidates(
                "acme", scope_key=scope_key, subjects=spellings(asked), limit=k
            )
            return first in {m.memory_id for m in got}

        assert not await found("Pallet jack 9"), "past the window: only the lookup finds it"
        for asked in ("forklift-4", "FORKLIFT4", "forklift no. 4", "Forklift 4"):
            assert await found(asked), asked
        for other in ("forklift 3", "Forklift-40", "forklift 4b"):
            assert not await found(other), other


async def test_subjects_are_folded_by_the_database_and_identities_never_are(
    container, uow_factory
) -> None:
    """Both sides are folded by PostgreSQL: Python's lower() writes "ΟΔΟΣ" as "οδος" and
    PostgreSQL as "οδοσ", so a spelling folded in Python missed a stored "ΟΔΟΣ 4". And an
    identity in another case ("User:Alice") is no one: it folds onto neither "user:alice"
    nor "user:Alice", which are two people."""
    memory = container.services["memory"]

    async def remember(content: str, subject: str) -> str:
        async with uow_factory() as uow:
            ack = await memory.remember(
                uow,
                OPS,
                content=content,
                memory_type=MemoryType.SEMANTIC,
                lifetime=Lifetime.LONG_TERM,
                subject=subject,
            )
            await uow.commit()
        return ack.memory_id

    street = await remember("ΟΔΟΣ 4 is closed for resurfacing.", "ΟΔΟΣ 4")
    lower_alice = await remember("Prefers the morning shift.", "user:alice")
    upper_alice = await remember("Prefers the night shift.", "user:Alice")
    await remember("Dock 9 opens at six.", "Dock 9")  # the newest: the window of one
    async with uow_factory() as uow:
        stored = await uow.memories.get("acme", street)
        assert stored is not None
        scope_key = stored.scope.key()

        async def found(asked: str) -> set[str]:
            got = await uow.memories.candidates(
                "acme", scope_key=scope_key, subjects=spellings(asked), limit=1
            )
            return {m.memory_id for m in got} & {street, lower_alice, upper_alice}

        # limit 1: only the newest row of each kind comes through the recent window, so
        # anything else found here was found by its subject
        assert street in await found("ΟΔΟΣ-4")
        assert street in await found("οδοσ 4")
        assert await found("User:Alice") == set()
        assert await found("user:Alice") == {upper_alice}
        assert lower_alice in await found("user:alice")
        assert upper_alice not in await found("user:alice")
