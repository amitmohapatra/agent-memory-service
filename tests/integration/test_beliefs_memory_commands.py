"""The user's memory commands with consolidation on, end to end through ``/v1/context`` on
PostgreSQL with a stub model: they behave as they do without beliefs. What is remembered is
in the context at once and later on the person's page; what is forgotten (each forget path)
leaves the page and the search in the forget's own commit, before any rebuild, and a belief
that also cites other memories stays hidden until it is rebuilt; a corrected value is the
current one in the memories at once and, once consolidated, on the page, the old one only
under earlier values. A page never makes a context look as if it had a profile."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy import text

from memory_service.api.app import create_app
from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.memory.consolidation import ConsolidationService, beliefs_changed
from tests.integration.test_beliefs import (
    _BELIEF,
    StubModel,
    T,
    beliefs,
    consolidate_all,
    entry,
    ids,
    say,
    settle,
    wired,  # noqa: F401 - the fixture
)

pytestmark = pytest.mark.integration

ME = MemoryExecutionContext(tenant_id=T, user_id="caroline", workspace_id="ws")
HEADERS = {"X-API-Key": "test-key", "X-Trellis-Tenant": T, "X-Trellis-User": "caroline"}
SCOPE = {"workspace_id": "ws"}
SISTER = "My sister Anna lives in Lisbon."
DOG = "I have a dog named Rex."
HIKING = "I like hiking in the Alps."
#: the heading the harness adds its note on remembered rules under (agent-harness
#: ``trellis.harness.agent.PROFILE``): present only when the user has a profile
HARNESS_PROFILE = "## Profile\n"


def answer(prompt: str) -> list[dict]:
    """The sister and the dog in one line when both are new (a belief citing two memories),
    a manager's name kept current by update, any other fact a line of its own."""
    f, held = ids(prompt), ids(prompt, _BELIEF)
    out = []
    pair = [f[t] for t in (SISTER, DOG) if t in f]
    if pair:
        line = {2: "Has a sister Anna in Lisbon and a dog named Rex."}.get(len(pair))
        out.append(entry("add", pair, line or ("Has a dog named Rex." if DOG in f else SISTER)))
    for said, short in f.items():
        if said in (SISTER, DOG):
            continue
        if "manager" in said:
            old = next((b for text, b in held.items() if "manager" in text.lower()), "")
            name = said.rstrip(".").split()[-1]
            out.append(entry("update" if old else "add", [short], f"Manager: {name}.", old, "work"))
        else:
            out.append(entry("add", [short], said))
    return out


@pytest.fixture
async def api(container, wired) -> AsyncIterator[httpx.AsyncClient]:  # noqa: F811
    """The API on the test's container, consolidation on a stub model."""
    wired(StubModel(answer))
    app = create_app(container.settings, container=container)
    app.state.container = container  # the fixture's own: no second lifespan to close it
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://memory.test"
    ) as client:
        yield client


async def remember(api: httpx.AsyncClient, content: str) -> str:
    r = await api.post(
        "/v1/memories",
        headers=HEADERS,
        json={"scope": SCOPE, "content": content, "memory_type": "SEMANTIC"},
    )
    assert r.status_code == 201, r.text
    return r.json()["memory_id"]


async def context(api: httpx.AsyncClient, query: str) -> dict:
    r = await api.post("/v1/context", headers=HEADERS, json={"scope": SCOPE, "query": query})
    assert r.status_code == 200, r.text
    return r.json()


def section(rendered: str, heading: str) -> str:
    """One ``## heading`` section of a rendered context ("" when absent)."""
    found = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", rendered, re.M | re.S)
    return found.group(1) if found else ""


async def jobs_but_consolidation(container, uow_factory) -> None:
    """Every queued job but consolidation's (the index jobs, as on main): a run finds no
    model, so the beliefs stay as the write left them."""
    real = container.services["consolidation"]
    container.services["consolidation"] = ConsolidationService(uow_factory)
    await settle(container)
    container.services["consolidation"] = real


# -- remember -------------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["api", "chat"])
async def test_what_is_remembered_is_in_the_context_at_once_and_later_on_the_page(
    container, uow_factory, api, how
) -> None:
    if how == "api":
        await remember(api, SISTER)
    else:
        await say(container, uow_factory, "caroline", SISTER)
    await jobs_but_consolidation(container, uow_factory)  # not waiting for consolidation
    now = (await context(api, "Where does my sister live?"))["rendered"]
    assert "Anna lives in Lisbon" in now and not section(now, "People")
    await consolidate_all(container.services["consolidation"], uow_factory)
    later = (await context(api, "Where does my sister live?"))["rendered"]
    assert "Anna" in section(later, "People")


# -- forget ---------------------------------------------------------------------------------


async def forget_by_api(api, container, memory_id: str) -> None:
    r = await api.delete(f"/v1/memories/{memory_id}", headers=HEADERS)
    assert r.status_code == 204, r.text


async def forget_by_handle(api, container, memory_id: str) -> None:
    """The conversational forget: "forget that" names the memory by the handle the context
    showed it under."""
    shown = await context(api, "Where does my sister live?")
    [handle] = re.findall(r"\[(m\d+)\][^\n]*Anna lives in Lisbon", shown["rendered"])[:1]
    r = await api.delete(
        f"/v1/memories/{handle}",
        headers={**HEADERS, "X-Trellis-Workspace": "ws"},  # the scope the context was given
        params={"bundle_id": shown["bundle_id"]},
    )
    assert r.status_code == 204, r.text


async def forget_by_tool(api, container, memory_id: str) -> None:
    out = await container.services["agent_tools"].call(ME, "memory_forget", {"id": memory_id})
    assert out == {"id": memory_id, "forgotten": True}


@pytest.mark.parametrize("forget", [forget_by_api, forget_by_handle, forget_by_tool])
async def test_what_is_forgotten_leaves_the_page_and_the_search_before_any_rebuild(
    container, uow_factory, api, forget
) -> None:
    sister = await remember(api, SISTER)
    await remember(api, DOG)
    await remember(api, HIKING)
    await settle(container)
    page = section((await context(api, "Tell me about my family"))["rendered"], "People")
    assert "Has a sister Anna in Lisbon and a dog named Rex." in page and HIKING in page

    await forget(api, container, sister)
    # in the forget's own commit, before any job: the line citing it is hidden though it
    # also cites the dog, the line that does not stays
    shown = await context(api, "Tell me about Anna and Rex")
    page = section(shown["rendered"], "People")
    assert "Anna" not in page and "Rex" not in page and HIKING in page
    assert "Anna" not in section(shown["rendered"], "Summaries")
    found = await container.services["retrieval"].retrieve(ME, "sister Anna", kinds=("belief",))
    assert not any("Anna" in c.text for c in found.candidates)
    # the index jobs run (the memory's own point goes, as on main), no rebuild yet
    await jobs_but_consolidation(container, uow_factory)
    assert "Anna" not in (await context(api, "Where does Anna live?"))["rendered"]
    await consolidate_all(container.services["consolidation"], uow_factory)  # the rebuild
    page = section((await context(api, "Tell me about my family"))["rendered"], "People")
    assert "Has a dog named Rex." in page and HIKING in page and "Anna" not in page


async def test_forgetting_every_memory_of_the_user_leaves_no_belief(
    container, uow_factory, api
) -> None:
    for said in (SISTER, DOG, HIKING):
        await remember(api, said)
    await settle(container)
    assert await beliefs(uow_factory)
    async with uow_factory() as uow:
        mine = await container.services["memory"].list_memories(uow, ME)
    for memory in mine:
        await forget_by_api(api, container, memory.memory_id)
    assert await beliefs(uow_factory) == []  # in the forgets' own commits
    await settle(container)
    assert await beliefs(uow_factory) == []
    assert not section((await context(api, "Tell me about me"))["rendered"], "People")


# -- correct --------------------------------------------------------------------------------


async def test_a_corrected_value_is_current_at_once_and_the_old_one_only_an_earlier_value(
    container, uow_factory, api
) -> None:
    old = await remember(api, "My manager is Priya.")
    await settle(container)
    assert "Manager: Priya." in section(
        (await context(api, "Who is my manager?"))["rendered"], "People"
    )
    r = await api.post(
        f"/v1/memories/{old}/supersede",
        headers=HEADERS,
        json={"scope": SCOPE, "content": "My manager is now Ravi."},
    )
    assert r.status_code == 200, r.text
    # before consolidation: the memories' own supersession says who it is now, and no page
    # still says the old name
    await jobs_but_consolidation(container, uow_factory)
    before = (await context(api, "Who is my manager?"))["rendered"]
    assert "Priya" not in section(before, "People")
    current = section(before, "Memories") + section(before, "Most relevant")
    assert "Ravi" in current and "Priya" not in current
    # consolidated: the page says Ravi, and Priya only under its earlier values
    await consolidate_all(container.services["consolidation"], uow_factory)
    page = section((await context(api, "Who is my manager?"))["rendered"], "People")
    now, _, earlier = page.partition("Earlier values (since replaced):")
    assert "Manager: Ravi." in now and "Priya" not in now and "Manager: Priya." in earlier


# -- the profile heading the harness keys on ------------------------------------------------


async def test_a_page_alone_does_not_make_a_context_look_as_if_it_had_a_profile(
    container, uow_factory, api
) -> None:
    await remember(api, HIKING)
    await settle(container)
    rendered = (await context(api, "What do I like?"))["rendered"]
    assert HIKING in section(rendered, "People")
    assert HARNESS_PROFILE not in rendered  # no note on remembered rules switched on
    full = await api.post(
        "/v1/context",
        headers=HEADERS,
        json={"scope": SCOPE, "query": "What do I like?", "format": "full"},
    )
    assert full.status_code == 200, full.text
    assert "profile" not in full.json()  # the profile blocks are the user's, as on main


# -- the pages come on top of the memories --------------------------------------------------


async def test_the_memories_are_byte_identical_with_and_without_beliefs(
    container, uow_factory, api
) -> None:
    """With the pages not charged (the default), what the context had on main is untouched:
    the same memories, in the same order, for the same question and budget, whether or not
    the person has beliefs. More facts than the memories' count limit, each also a belief
    the question's search finds, so a belief that took a memory's place would show."""
    from memory_service.modules.context import sections

    assert sections.PEOPLE_CHARGED is False
    for n in range(110):
        # a day each: a distinct date orders the timeline and breaks the ranking's ties,
        # which are in no fixed order between two builds, beliefs or not
        await say(
            container, uow_factory, "caroline", f"I visited museum number {n} in Lisbon.", day=n
        )
    await settle(container)
    question = "Which museums in Lisbon did I visit?"
    with_beliefs = (await context(api, question))["rendered"]
    assert len(await beliefs(uow_factory)) == 110 and section(with_beliefs, "People")
    async with uow_factory() as uow:  # the same memories, no belief (as on main)
        rows = await uow.session.execute(text("SELECT DISTINCT scope_key FROM beliefs"))
        for (scope,) in rows.all():
            gone = await uow.beliefs.delete_unit(T, scope, "user:caroline", datetime.now(UTC))
            await beliefs_changed(uow, gone)  # as a write does: points and caches follow
        await uow.commit()
    await jobs_but_consolidation(container, uow_factory)  # their points go too
    without = (await context(api, question))["rendered"]
    assert not section(without, "People")
    for heading in ("Most relevant", "Memories"):
        assert section(with_beliefs, heading) == section(without, heading), heading
    assert section(with_beliefs, "Memories")  # there are memories to compare
