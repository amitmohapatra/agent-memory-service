"""List beliefs end to end against PostgreSQL with a stub model, through the worker's own
job: the incremental path (a line, a list at the second instance, a longer list at each
later one, all in the one consolidation call a run makes), each instance checked against
its own memory, the page with the list first, forgetting an instance's fact and correcting
one."""

from __future__ import annotations

import re

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
    """Every trip is a line of its own; from the second trip on there is also one list of
    them, and each later trip is one more item of it. A correction rewrites both lines."""
    facts, held = ids(prompt), ids(prompt, _BELIEF)
    entry = {"facts": [], "belief": "", "text": "", "facet": "places", "items": []}
    listed = next(((line, b) for line, b in held.items() if line.startswith(TITLE)), None)
    lines = {c: b for line, b in held.items() for c in CITIES if line == f"Went to {c}."}
    if fixed := next((f for said, f in facts.items() if "not Rome" in said), None):
        assert listed is not None
        line = {**entry, "op": "update", "facts": [fixed]}
        return [
            {**line, "belief": lines["Rome"], "text": "Went to Milan."},
            {**line, "belief": listed[1], "text": listed[0].replace("Rome", "Milan")},
        ]
    new = {city: f for said, f in facts.items() for city in CITIES if city in said}
    out = [{**entry, "op": "add", "facts": [f], "text": f"Went to {c}."} for c, f in new.items()]
    items = [{"text": city, "fact": f} for city, f in new.items()]
    cited = sorted(new.values())
    if new and listed is not None:  # only the new instances: what the list holds is not sent
        out.append({**entry, "op": "update", "facts": cited, "belief": listed[1], "items": items})
    elif new and len(lines) + len(new) > 1:
        earlier = [{"text": city, "fact": b} for city, b in lines.items()]
        out.append({**entry, "op": "add", "facts": cited, "text": TITLE, "items": earlier + items})
    return out


async def trip_list(uow_factory):
    """The person's current list of trips (None when there is none)."""
    found = [b for b in await beliefs(uow_factory) if b.facet == LIST_FACET]
    assert len(found) <= 1
    return found[0] if found else None


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
    await settle(container)
    [first] = await beliefs(uow_factory)
    assert first.text == "Went to Paris." and first.facet == "places"  # one instance: no list

    await say(container, uow_factory, "caroline", "I flew to Rome yesterday.", day=10)
    await settle(container)
    two = await trip_list(uow_factory)
    # dated by the code: the day the first was said, the day "yesterday" resolves to
    assert two.text == "Trips taken: Paris (said 2025-01-06); Rome (2025-01-15)"
    assert len(two.source_ids) == 2
    # the list is kept besides the ordinary lines, which stay as they were
    plain = sorted(b.text for b in await beliefs(uow_factory) if b.facet != LIST_FACET)
    assert plain == ["Went to Paris.", "Went to Rome."]
    assert first.belief_id in {b.belief_id for b in await beliefs(uow_factory)}

    await say(container, uow_factory, "caroline", "Back from Berlin!", day=20)
    await settle(container)
    three = await trip_list(uow_factory)
    assert three.text == two.text + "; Berlin (said 2025-01-26)"
    assert three.supersedes == two.belief_id and len(three.source_ids) == 3  # carried over
    assert stub.calls["consolidate"] == 3  # one call a run, the list kept by it
    # the citation check read each instance, as the list says it, against its own memory
    read = "\n".join(stub.checked)
    for claim, memory in (
        ("Trips taken: Paris (said 2025-01-06)", "[2025-01-06] (caroline): I went to Paris."),
        (
            "Trips taken: Rome (2025-01-15)",
            "(caroline): I flew to Rome yesterday. (when: 2025-01-15)",
        ),
        ("Trips taken: Berlin (said 2025-01-26)", "[2025-01-26] (caroline): Back from Berlin!"),
    ):
        assert re.search(rf"^L\d+: {re.escape(claim)}\n  - .*{re.escape(memory)}$", read, re.M)
    assert "I went to Paris." not in stub.checked[-1]  # what the list held is not read again
    # the page: the list first, and no version of it under earlier values (it only grew)
    shown = await page(container)
    assert shown.splitlines()[2] == f"- {three.text}"
    assert "- Went to Paris." in shown and "Earlier values" not in shown


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
    # in the forget's commit: every line that cited it is gone, the list among them
    assert sorted(b.text for b in await beliefs(uow_factory)) == [
        "Went to Berlin.",
        "Went to Paris.",
    ]
    assert "Rome" not in await page(container)
    await settle(container)  # the person's rebuild, from the facts that are left
    left = await trip_list(uow_factory)
    assert left.text == "Trips taken: Paris (said 2025-01-06); Berlin (said 2025-01-26)"
    assert rome.memory_id not in left.source_ids


async def test_a_corrected_instance_hides_the_list_until_the_run_has_rewritten_it(
    container,
    uow_factory,
    wired,  # noqa: F811
) -> None:
    wired(StubModel(trips))
    await three_trips(container, uow_factory)
    before = await trip_list(uow_factory)
    async with uow_factory() as uow:
        mine = await container.services["memory"].list_memories(uow, MINE)
        rome = next(m for m in mine if "Rome" in m.content)  # the turn, or its fact
        await container.services["memory"].supersede(
            uow, MINE, rome.memory_id, content="I flew to Milan, not Rome.", reason="corrected"
        )
        await uow.commit()
    assert "Trips taken" not in await page(container)  # stale: no page still lists Rome
    await settle(container)
    after = await trip_list(uow_factory)
    assert after.text == before.text.replace("Rome", "Milan")  # still a list, rewritten whole
    assert after.supersedes == before.belief_id
    shown = await page(container)
    # the list's old version is no earlier value of the person's
    assert f"- {after.text}" in shown and before.text not in shown
