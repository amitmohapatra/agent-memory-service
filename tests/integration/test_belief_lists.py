"""List beliefs end to end against PostgreSQL with a stub model, through the worker's own
job: the incremental path (a belief, a list at the second instance, a longer list at each
later one, in the one consolidation call a run makes), the citation check over every
instance, the page with the list first, forgetting an instance's fact and correcting one."""

from __future__ import annotations

import pytest

from memory_service.domain.beliefs import LIST_FACET
from memory_service.domain.context_bundle import people_section
from tests.integration.test_beliefs import (
    _BELIEF,
    ASKER,
    StubModel,
    beliefs,
    ids,
    say,
    settle,
    wired,  # noqa: F401 - the fixture
)

pytestmark = pytest.mark.integration

MINE = ASKER.model_copy(update={"user_id": "caroline"})
CITIES = ("Paris", "Rome", "Berlin", "Milan")
TITLE = "Trips taken"


def trips(prompt: str) -> list[dict]:
    """A trip is a belief of its own until a second one comes; then both are one list, and
    each later trip is one more item of it. A correction rewrites the list's line."""
    facts, held = ids(prompt), ids(prompt, _BELIEF)
    new = {city: f for said, f in facts.items() for city in CITIES if city in said}
    if not new:
        return []
    plain = {c: b for line, b in held.items() for c in CITIES if c in line and TITLE not in line}
    listed = next(((line, b) for line, b in held.items() if line.startswith(TITLE)), None)
    entry = {"op": "update", "facts": sorted(new.values()), "facet": "places"}
    if corrected := next((f for said, f in facts.items() if "not Rome" in said), None):
        assert listed is not None
        line = listed[0].replace("Rome", "Milan")
        return [{**entry, "facts": [corrected], "belief": listed[1], "text": line, "items": []}]
    items = [{"text": city, "fact": f} for city, f in new.items()]
    if listed is not None:  # only the new instances: what the list holds is not sent again
        return [{**entry, "belief": listed[1], "text": "", "items": items}]
    if plain or len(new) > 1:
        folded = [{"text": city, "fact": b} for city, b in plain.items()]
        target = next(iter(plain.values()), "")
        op = "update" if plain else "add"
        return [{**entry, "op": op, "belief": target, "text": TITLE, "items": folded + items}]
    [city] = new
    return [{**entry, "op": "add", "belief": "", "text": f"Went to {city}.", "items": []}]


async def page(container) -> str:
    bundle = await container.services["context_builder"].build(ASKER, "Where has Caroline been?")
    return people_section(bundle.people) or ""


async def three_trips(container, uow_factory) -> None:
    await say(container, uow_factory, "caroline", "I went to Paris.")
    await settle(container)
    await say(container, uow_factory, "caroline", "I flew to Rome yesterday.", day=10)
    await settle(container)
    await say(container, uow_factory, "caroline", "Back from Berlin!", day=20)
    await settle(container)


async def test_the_second_instance_makes_a_list_and_later_ones_extend_it(
    container,
    uow_factory,
    wired,  # noqa: F811
) -> None:
    stub = StubModel(trips)
    wired(stub)
    await say(container, uow_factory, "caroline", "I went to Paris.")
    await say(container, uow_factory, "caroline", "I paint sunsets on weekends.")
    await settle(container)
    [first] = [b for b in await beliefs(uow_factory) if "Paris" in b.text]
    assert first.text == "Went to Paris." and first.facet == "places"  # one instance: no list

    await say(container, uow_factory, "caroline", "I flew to Rome yesterday.", day=10)
    await settle(container)
    [two] = await beliefs(uow_factory)
    # dated by the code: the day the first was said, the day "yesterday" resolves to
    assert two.text == "Trips taken: Paris (said 2025-01-06); Rome (2025-01-15)"
    assert two.facet == LIST_FACET and len(two.source_ids) == 2
    history = {b.belief_id: b for b in await beliefs(uow_factory, current=False)}
    assert history[first.belief_id].superseded_by == two.belief_id  # gathered into the list

    await say(container, uow_factory, "caroline", "Back from Berlin!", day=20)
    await settle(container)
    [three] = await beliefs(uow_factory)
    assert three.text == two.text + "; Berlin (said 2025-01-26)"
    assert three.supersedes == two.belief_id and len(three.source_ids) == 3  # carried over
    assert stub.calls["consolidate"] == 3  # one call a run, the list kept by it
    # the citation check read the whole list against every instance's memory
    claim = stub.checked[-1]
    assert f"L1: {three.text}" in claim
    assert all(said in claim for said in ("I went to Paris.", "Rome yesterday", "Berlin!"))
    # the page: the list, and nothing of it under earlier values (it grew, nothing was replaced)
    shown = await page(container)
    assert shown.splitlines()[2] == f"- {three.text}"
    assert "Earlier values" not in shown and "Went to Paris." not in shown


async def test_forgetting_an_instances_fact_takes_its_entry_out_of_the_list(
    container,
    uow_factory,
    wired,  # noqa: F811
) -> None:
    wired(StubModel(trips))
    await three_trips(container, uow_factory)
    async with uow_factory() as uow:
        mine = await container.services["memory"].list_memories(uow, MINE)
        rome = next(m for m in mine if "Rome" in m.content)  # the turn, or its fact
        await container.services["memory"].forget(uow, MINE, rome.memory_id)
        await uow.commit()
    assert await beliefs(uow_factory) == []  # in the forget's commit: the line named Rome
    assert "Rome" not in await page(container)
    await settle(container)  # the person's rebuild, from the facts that are left
    [left] = await beliefs(uow_factory)
    assert left.text == "Trips taken: Paris (said 2025-01-06); Berlin (said 2025-01-26)"
    assert left.facet == LIST_FACET and rome.memory_id not in left.source_ids


async def test_a_corrected_instance_hides_the_list_until_the_run_has_rewritten_it(
    container,
    uow_factory,
    wired,  # noqa: F811
) -> None:
    wired(StubModel(trips))
    await three_trips(container, uow_factory)
    [before] = await beliefs(uow_factory)
    async with uow_factory() as uow:
        mine = await container.services["memory"].list_memories(uow, MINE)
        rome = next(m for m in mine if "Rome" in m.content)  # the turn, or its fact
        await container.services["memory"].supersede(
            uow, MINE, rome.memory_id, content="I flew to Milan, not Rome.", reason="corrected"
        )
        await uow.commit()
    assert "Trips taken" not in await page(container)  # stale: no page still says Rome
    await settle(container)
    [after] = await beliefs(uow_factory)
    assert after.text == before.text.replace("Rome", "Milan")
    assert after.facet == LIST_FACET and after.supersedes == before.belief_id
    shown = await page(container)
    assert after.text in shown and "Rome" not in shown  # the old version is no earlier value
