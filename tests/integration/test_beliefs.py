"""Beliefs end to end against PostgreSQL with a stub model: the trigger, the incremental
keep/update/retire/add/skip, the citation check, the per-person page in the context, recall
hits, the delete cascade, reader visibility, and two workers on one person."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from benchmark.common import submit_observation
from sqlalchemy import text

from memory_service.adapters.models.llm import EndpointLLM
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.context_bundle import people_section
from memory_service.domain.enums import Visibility
from memory_service.domain.observation import ProcessingHints
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.jobs.names import TASK_CONSOLIDATE
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.memory import consolidation
from memory_service.modules.memory.consolidation import ConsolidationService
from tests.support_llm import (
    NO_BACKOFF,
    StaticPolicies,
    chat_response,
    llm_settings,
    mocked_gateway,
)

pytestmark = pytest.mark.integration

T = "acme"
ASKER = MemoryExecutionContext(tenant_id=T, user_id="asker", workspace_id="ws")
CAROLINE, MELANIE = "user:caroline", "user:melanie"
DAY = datetime(2025, 1, 6, 10, 0, tzinfo=UTC)
_FACT = re.compile(r"^(F\d+) \[[^\]]*\] \([^)]*\): (.*)$", re.M)
_BELIEF = re.compile(r"^(B\d+) \[[^\]]*\] (.*)$", re.M)
_LINE = re.compile(r"^(L\d+): (.*)$", re.M)


def ids(prompt: str, pattern: re.Pattern[str] = _FACT) -> dict[str, str]:
    """Text -> short id, as the prompt numbered them."""
    return {text: short for short, text in pattern.findall(prompt)}


def entry(
    op: str, facts: list[str], text: str = "", belief: str = "", facet: str = "other"
) -> dict:
    return {"op": op, "facts": facts, "belief": belief, "text": text, "facet": facet, "items": []}


class StubModel:
    """The endpoint: consolidation answers come from ``answer(user_prompt)``, every citation
    check line gets ``verdict(claim)``, every rewrite ``rewrite(claim)``."""

    def __init__(
        self,
        answer: Callable[[str], list[dict]],
        *,
        verdict: Callable[[str], str] = lambda claim: "SUPPORTED",
        rewrite: Callable[[str], str] = lambda claim: claim,
    ) -> None:
        self.answer, self.verdict, self.rewrite = answer, verdict, rewrite
        self.calls = {"consolidate": 0, "check": 0, "rewrite": 0}
        self.prompts: list[str] = []
        #: the citation check's prompts: each claim with the memories it cites under it
        self.checked: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        messages = json.loads(request.content)["messages"]
        system, user = messages[0]["content"], messages[1]["content"]
        if system.startswith(consolidation.SYSTEM[:40]):
            self.calls["consolidate"] += 1
            self.prompts.append(user)
            out: dict[str, Any] = {"entries": self.answer(user)}
        elif system.startswith(consolidation.CHECK_SYSTEM[:40]):
            self.calls["check"] += 1
            self.checked.append(user)
            lines = _LINE.findall(user)
            out = {"verdicts": [{"line": n, "label": self.verdict(c)} for n, c in lines]}
        else:
            self.calls["rewrite"] += 1
            lines = _LINE.findall(user)
            out = {"rewrites": [{"line": n, "text": self.rewrite(c)} for n, c in lines]}
        return httpx.Response(200, json=chat_response(json.dumps(out)))


async def say(container, uow_factory, speaker: str, text: str, *, day: int = 0, **hints) -> None:
    register_handlers(container)
    async with uow_factory() as uow:
        await submit_observation(
            uow,
            ASKER.model_copy(update={"user_id": speaker}),
            content=text,
            hints=ProcessingHints(visibility=hints.get("visibility", Visibility.TENANT)),
            occurred_at=DAY + timedelta(days=day),
        )
        await uow.commit()


async def settle(container) -> None:
    for _ in range(4):
        await container.tasks.drain()


async def units(uow_factory, person: str = CAROLINE) -> list[tuple[str, str, str]]:
    """Every (tenant, scope, person) unit the person has: a person's USER and PREFERENCE
    facts are anchored in their own scope, the rest in the conversation's."""
    async with uow_factory() as uow:
        rows = await uow.session.execute(
            text(
                "SELECT scope_key FROM belief_progress WHERE tenant_id = :t AND person_key = :p "
                "ORDER BY scope_key"
            ),
            {"t": T, "p": person},
        )
        return [(T, scope, person) for (scope,) in rows]


@pytest.fixture
def wired(container):
    """The container's pipeline and consolidation running on a stub model (``use(stub)``)."""
    with mocked_gateway() as gw:

        def use(stub: StubModel) -> ConsolidationService:
            gw.route.side_effect = stub
            settings = llm_settings()
            assist = LLMAssist(
                EndpointLLM(settings, transport=NO_BACKOFF),
                settings,
                StaticPolicies([consolidation.CONSOLIDATION_USE]),  # type: ignore[arg-type]
            )
            container.services["observation_pipeline"].assist = assist
            container.services["memory"].assist = assist  # remember, supersede, forget
            container.services["llm_assist"] = assist  # a deployment that reaches a model
            container.services["context_builder"].assist = assist
            service = ConsolidationService(container.services["uow_factory"], assist=assist)
            container.services["consolidation"] = service
            return service

        yield use


async def beliefs(uow_factory, person: str = CAROLINE, *, current: bool = True):
    """The person's beliefs in every scope (current only, or with their history)."""
    found = []
    for unit in await units(uow_factory, person):
        async with uow_factory() as uow:
            found += await uow.beliefs.of_unit(*unit, current_only=current)
    return found


async def consolidate_all(service: ConsolidationService, uow_factory, person: str = CAROLINE):
    for unit in await units(uow_factory, person):
        await service.consolidate(*unit)


def consolidate_jobs(container) -> list[dict]:
    """The jobs new facts queued (not a run's continuation over facts it left)."""
    return [
        s.payload
        for s in container.tasks.payloads.values()
        if s.task_name == TASK_CONSOLIDATE and ":next:" not in (s.idempotency_key or "")
    ]


# -- 1. the trigger -------------------------------------------------------------------------


async def test_a_burst_of_facts_queues_one_job_per_person_and_none_without_a_model(
    container, uow_factory, wired
) -> None:
    await say(container, uow_factory, "caroline", "I paint sunsets on weekends.")
    await settle(container)
    assert consolidate_jobs(container) == []  # no model: nothing is queued
    wired(StubModel(lambda prompt: []))
    for i in range(20):
        await say(container, uow_factory, "caroline", f"I visited museum number {i} today.")
    await say(container, uow_factory, "melanie", "Caroline showed me her new painting.")
    await settle(container)
    queued = [(p["scope_key"], p["person_key"]) for p in consolidate_jobs(container)]
    assert len(queued) == len(set(queued))  # one job per person and scope, burst or not
    assert {person for _, person in queued} == {CAROLINE, MELANIE}  # Melanie named Caroline
    scope = next(scope for scope, person in queued if person == MELANIE)
    assert (scope, CAROLINE) in queued


# -- 2. incremental operations, history on the page -----------------------------------------


async def test_keep_update_retire_add_skip_and_the_page_with_earlier_values(
    container, uow_factory, wired
) -> None:
    def answer(prompt: str) -> list[dict]:
        f, b = ids(prompt), ids(prompt, _BELIEF)
        out = []
        if "I work at Microsoft." in f:
            out.append(
                entry(
                    "add",
                    [f["I work at Microsoft."]],
                    "Works at Microsoft (Jan 2025).",
                    facet="work",
                )
            )
        if "I paint sunsets on weekends." in f:
            out.append(
                entry(
                    "add", [f["I paint sunsets on weekends."]], "Paints sunsets.", facet="hobbies"
                )
            )
        if "I started at Google in March." in f:
            out.append(
                entry(
                    "update",
                    [f["I started at Google in March."]],
                    "Works at Google (since Mar 2025).",
                    belief=b.get("Works at Microsoft (Jan 2025).", ""),
                )
            )
        if "I painted another sunset." in f:
            out.append(
                entry("keep", [f["I painted another sunset."]], belief=b.get("Paints sunsets.", ""))
            )
        if "Hello there!" in f:
            out.append(entry("skip", [f["Hello there!"]]))
        if "I stopped painting." in f:
            out.append(
                entry("retire", [f["I stopped painting."]], belief=b.get("Paints sunsets.", ""))
            )
        return out

    service = wired(StubModel(answer))
    await say(container, uow_factory, "caroline", "I work at Microsoft.")
    await say(container, uow_factory, "caroline", "I paint sunsets on weekends.")
    await settle(container)
    assert sorted(x.text for x in await beliefs(uow_factory)) == [
        "Paints sunsets.",
        "Works at Microsoft (Jan 2025).",
    ]
    await say(container, uow_factory, "caroline", "I started at Google in March.", day=60)
    await say(container, uow_factory, "caroline", "I painted another sunset.", day=60)
    await say(container, uow_factory, "caroline", "Hello there!", day=60)
    await settle(container)
    now = {x.text: x for x in await beliefs(uow_factory)}
    assert set(now) == {"Paints sunsets.", "Works at Google (since Mar 2025)."}
    assert now["Paints sunsets."].proof_count == 2
    everything = {x.text: x for x in await beliefs(uow_factory, current=False)}
    old = everything["Works at Microsoft (Jan 2025)."]
    assert old.superseded_by == now["Works at Google (since Mar 2025)."].belief_id
    assert len(now["Works at Google (since Mar 2025)."].source_ids) == 2  # carried over
    await say(container, uow_factory, "caroline", "I stopped painting.", day=90)
    await settle(container)
    painted = {x.text: x for x in await beliefs(uow_factory, current=False)}["Paints sunsets."]
    assert painted.valid_to == DAY + timedelta(days=90)
    bundle = await container.services["context_builder"].build(ASKER, "Where does Caroline work?")
    page = next(b.text for b in bundle.people if b.block == "caroline")
    current, _, earlier = page.partition("Earlier values (since replaced):")
    assert "Works at Google" in current and "Microsoft" not in current
    assert "Microsoft" in earlier and "Paints sunsets." in earlier
    assert "### caroline" in bundle.render()
    assert service is container.services["consolidation"]


# -- 3. the citation check ------------------------------------------------------------------


async def test_the_citation_check_drops_what_its_cited_memories_do_not_support(
    container, uow_factory, wired
) -> None:
    await say(container, uow_factory, "melanie", "I took my kids camping.")
    await settle(container)
    async with uow_factory() as uow:
        theirs = await container.services["memory"].list_memories(
            uow, ASKER.model_copy(update={"user_id": "melanie"})
        )

    def answer(prompt: str) -> list[dict]:
        f = ids(prompt)["I visited Paris in May 2023."]
        return [
            entry("add", [], "Uncited line."),
            entry("add", [theirs[0].memory_id], "Took kids camping."),
            entry("add", [f], "Visited Paris with her sister."),
            entry("add", [f], "Visited Paris in 2024."),
            entry("add", [f], "Visited Paris (May 2023)."),
            entry("add", [f], "Went to France in May."),
        ]

    verdicts = {
        "Visited Paris with her sister.": "UNSUPPORTED",
        "Visited Paris in 2024.": "PARTIAL",
        "Went to France in May.": "PARTIAL",
        "Visited Paris in May 2023.": "SUPPORTED",
    }
    rewrites = {"Visited Paris in 2024.": "Visited Paris in May 2023."}
    stub = StubModel(
        answer,
        verdict=lambda c: verdicts.get(
            c, "SUPPORTED" if c == "Visited Paris (May 2023)." else "PARTIAL"
        ),
        rewrite=lambda c: rewrites.get(c, c),
    )
    wired(stub)
    await say(container, uow_factory, "caroline", "I visited Paris in May 2023.")
    await settle(container)
    assert sorted(x.text for x in await beliefs(uow_factory)) == [
        "Visited Paris (May 2023).",
        "Visited Paris in May 2023.",
    ]
    assert stub.calls == {"consolidate": 1, "check": 2, "rewrite": 1}  # all batched


# -- 4. the fingerprint skip ----------------------------------------------------------------


async def test_unchanged_sources_make_no_call_and_a_new_fact_calls_for_its_person_only(
    container, uow_factory, wired
) -> None:
    def answer(prompt: str) -> list[dict]:
        return [entry("add", [short], text) for text, short in ids(prompt).items()]

    stub = StubModel(answer)
    service = wired(stub)
    await say(container, uow_factory, "caroline", "I adopted a dog named Max.")
    await say(container, uow_factory, "melanie", "I run marathons.")
    await settle(container)
    assert stub.calls["consolidate"] == 2
    await consolidate_all(service, uow_factory, CAROLINE)
    await consolidate_all(service, uow_factory, MELANIE)
    assert stub.calls["consolidate"] == 2  # nothing new: zero calls
    await say(container, uow_factory, "melanie", "I ran the Boston marathon.", day=3)
    await settle(container)
    assert stub.calls["consolidate"] == 3
    assert [p["person_key"] for p in consolidate_jobs(container)][-1] == MELANIE


# -- 5./6. the page and recall hits in the context, no model call on the read ---------------


async def test_the_context_pins_the_page_and_recalls_beliefs_without_a_model_call(
    container, uow_factory, wired
) -> None:
    long = "Volunteers at the animal shelter every Saturday morning " + "and more " * 36

    def answer(prompt: str) -> list[dict]:
        f = ids(prompt)["I volunteer at the animal shelter."]
        return [entry("add", [f], long.strip(), facet="hobbies")]

    gw_stub = StubModel(answer)
    wired(gw_stub)
    await say(container, uow_factory, "caroline", "I volunteer at the animal shelter.")
    await settle(container)
    builder = container.services["context_builder"]
    calls = sum(gw_stub.calls.values())
    pinned = await builder.build(ASKER, "What does Caroline do on Saturdays?")
    assert any(b.block == "caroline" for b in pinned.people)
    assert not pinned.summaries  # pinned already: not repeated as a recall hit
    small = await builder.build(ASKER, "animal shelter Saturday volunteers", token_budget=200)
    assert not any(b.block == "caroline" for b in small.people)  # over the pinned share
    assert [s.text for s in small.summaries] == [long.strip()]
    assert "[s1]" in small.render()
    assert sum(gw_stub.calls.values()) == calls  # the read path never calls the model


# -- 8. the delete cascade ------------------------------------------------------------------


def _cats(prompt: str) -> list[dict]:
    """A line from whichever cat facts are left, and one from Ohio."""
    f = ids(prompt)
    cats = [f[t] for t in ("I have two cats.", "My cats are named Luna and Sol.") if t in f]
    out = []
    if cats:
        out.append(
            entry("add", cats, "Has two cats, Luna and Sol." if len(cats) == 2 else "Has two cats.")
        )
    if "I grew up in Ohio." in f:
        out.append(entry("add", [f["I grew up in Ohio."]], "Grew up in Ohio."))
    return out


async def _three_facts(container, uow_factory) -> dict[str, str]:
    """Caroline's three facts, consolidated; their memory ids by text."""
    for said in ("I have two cats.", "My cats are named Luna and Sol.", "I grew up in Ohio."):
        await say(container, uow_factory, "caroline", said)
    await settle(container)
    async with uow_factory() as uow:
        mine = await container.services["memory"].list_memories(uow, MINE)
    return {m.content: m.memory_id for m in mine}


MINE = ASKER.model_copy(update={"user_id": "caroline"})


async def test_a_belief_that_loses_one_of_its_sources_is_hidden_at_once_and_rebuilt(
    container, uow_factory, wired
) -> None:
    service = wired(StubModel(_cats))
    said = await _three_facts(container, uow_factory)
    by_text = {x.text: x for x in await beliefs(uow_factory)}
    cats, ohio = by_text["Has two cats, Luna and Sol."], by_text["Grew up in Ohio."]
    async with uow_factory() as uow:
        before = await uow.revisions.get_many(T, [(RevisionKind.TENANT, "")])
    memory = container.services["memory"]
    async with uow_factory() as uow:
        await memory.forget(uow, MINE, said["My cats are named Luna and Sol."])
        await uow.commit()
    [unit] = await units(uow_factory)
    async with uow_factory() as uow:  # in the forget's own commit, before any job runs
        left = {
            b.belief_id: b for b in await uow.beliefs.get_many(T, [cats.belief_id, ohio.belief_id])
        }
        after = await uow.revisions.get_many(T, [(RevisionKind.TENANT, "")])
        progress = await uow.beliefs.progress(*unit)
    gone = left[cats.belief_id]  # it still cites "I have two cats.", and is hidden all the same
    assert gone.deleted_at is not None and gone.text == "" and not gone.source_ids
    assert left[ohio.belief_id].current  # a belief the forget did not touch stays
    assert after != before
    assert progress is not None and progress.last_consolidated_at is None  # rebuilt next
    # before any job runs (its point still in the index): not on the page, not found
    builder = container.services["context_builder"]
    bundle = await builder.build(ASKER, "What are Caroline's cats called?")
    page = people_section(bundle.people) or ""
    assert "Luna" not in page and "Grew up in Ohio." in page
    assert not any("Luna" in s.text for s in bundle.summaries)
    found = await container.services["retrieval"].retrieve(
        ASKER, "cats Luna Sol", kinds=("belief",)
    )
    assert cats.belief_id not in {c.record_id for c in found.candidates}
    # the index jobs run (the memory's own point goes, as on main), the rebuild not yet
    container.services["consolidation"] = ConsolidationService(uow_factory)
    await settle(container)
    assert "Luna" not in (await builder.build(ASKER, "Caroline's cats Luna?")).render()
    container.services["consolidation"] = service
    await consolidate_all(service, uow_factory)  # the rebuild: what the sources left support
    assert sorted(x.text for x in await beliefs(uow_factory)) == [
        "Grew up in Ohio.",
        "Has two cats.",
    ]
    async with uow_factory() as uow:
        for m in await memory.list_memories(uow, MINE):
            await memory.forget(uow, MINE, m.memory_id)
        await uow.commit()
    await settle(container)
    assert await beliefs(uow_factory) == []  # the user's memories gone: no belief left


async def test_without_a_model_a_belief_that_lost_a_source_stays_deleted(
    container, uow_factory, wired
) -> None:
    wired(StubModel(_cats))
    said = await _three_facts(container, uow_factory)
    cats = next(x for x in await beliefs(uow_factory) if "Luna" in x.text)
    container.services["consolidation"] = ConsolidationService(uow_factory)  # no model now
    async with uow_factory() as uow:
        await container.services["memory"].forget(uow, MINE, said["I grew up in Ohio."])
        await container.services["memory"].forget(uow, MINE, said["I have two cats."])
        await uow.commit()
    await settle(container)  # the rebuild finds no model: nothing written again
    assert await beliefs(uow_factory) == []
    async with uow_factory() as uow:
        [dropped] = await uow.beliefs.get_many(T, [cats.belief_id])
    assert dropped.deleted_at is not None and dropped.text == ""
    bundle = await container.services["context_builder"].build(ASKER, "Caroline's cats?")
    assert not bundle.people and "Luna" not in "".join(s.text for s in bundle.summaries)


# -- pruning: the newest superseded versions of a belief stay -------------------------------


async def test_the_hourly_sweep_keeps_the_newest_superseded_versions_of_a_belief(
    container, uow_factory, wired
) -> None:
    def answer(prompt: str) -> list[dict]:
        f, b = ids(prompt), ids(prompt, _BELIEF)
        [(said, short)] = f.items()
        line = "Works at " + said.removeprefix("I work at ").rstrip(".") + "."
        current = next(iter(b.values()), "")
        return [entry("update" if current else "add", [short], line, current, "work")]

    wired(StubModel(answer))
    for n in range(6):
        await say(container, uow_factory, "caroline", f"I work at Company{n}.", day=10 * n)
        await settle(container)
    history = await beliefs(uow_factory, current=False)
    assert len(history) == 6 and [b.text for b in history if b.current] == ["Works at Company5."]
    assert await container.services["consolidation"].prune() == 2
    kept = sorted(b.text for b in await beliefs(uow_factory, current=False))
    assert kept == [f"Works at Company{n}." for n in (2, 3, 4, 5)]  # the current and 3 before
    assert await container.services["consolidation"].prune() == 0


# -- 9. reader visibility -------------------------------------------------------------------


async def test_a_reader_who_cannot_see_a_cited_memory_does_not_see_the_belief(
    container, uow_factory, wired
) -> None:
    def answer(prompt: str) -> list[dict]:
        return [entry("add", [short], text) for text, short in ids(prompt).items()]

    wired(StubModel(answer))
    await say(
        container, uow_factory, "caroline", "I am allergic to peanuts.", visibility=Visibility.USER
    )
    await settle(container)
    builder = container.services["context_builder"]
    theirs = await builder.build(ASKER, "Is Caroline allergic to anything?")
    assert not any(b.block == "caroline" for b in theirs.people)
    own = await builder.build(
        ASKER.model_copy(update={"user_id": "caroline"}), "What am I allergic to?"
    )
    assert any("peanuts" in b.text for b in own.people if b.block == "caroline")


# -- 10. two workers, and a retry after a crash ---------------------------------------------


async def test_two_workers_serialise_and_a_retry_after_a_crash_writes_once(
    container, uow_factory, wired, monkeypatch
) -> None:
    def answer(prompt: str) -> list[dict]:
        return [entry("add", [short], text) for text, short in ids(prompt).items()]

    service = wired(StubModel(answer))
    container.services["consolidation"] = ConsolidationService(uow_factory)  # jobs: no model
    await say(container, uow_factory, "caroline", "I speak three languages.")
    await settle(container)
    real = service._write
    crashed = []

    async def crash_once(*args, **kwargs):
        if not crashed:
            crashed.append(True)
            raise ConnectionError("worker killed mid-job")
        return await real(*args, **kwargs)

    monkeypatch.setattr(service, "_write", crash_once)
    [unit] = await units(uow_factory)
    with pytest.raises(ConnectionError):
        await service.consolidate(*unit)
    assert await beliefs(uow_factory) == []  # nothing half-written
    await asyncio.gather(service.consolidate(*unit), service.consolidate(*unit))
    assert [x.text for x in await beliefs(uow_factory)] == ["I speak three languages."]


# -- a corpus ingested before beliefs existed (the acceptance run's path) --------------------


async def test_a_tenant_ingested_without_beliefs_is_consolidated_through_the_job(
    container, uow_factory, wired
) -> None:
    from memory_service.tools.consolidate import consolidate_tenant

    for text_ in ("I bake bread on Sundays.", "My sister lives in Lisbon."):
        await say(container, uow_factory, "caroline", text_)
    await say(container, uow_factory, "melanie", "Caroline gave me a loaf of her bread.")
    await settle(container)  # no model yet: no belief, nothing queued
    assert consolidate_jobs(container) == []

    def answer(prompt: str) -> list[dict]:
        return [entry("add", [short], text) for text, short in ids(prompt).items()]

    stub = StubModel(answer)
    wired(stub)
    report = await consolidate_tenant(container, T)
    assert report["units"] == 2 and report["current_beliefs"] >= 3
    calls = stub.calls["consolidate"]
    again = await consolidate_tenant(container, T)
    assert again["current_beliefs"] == report["current_beliefs"]
    assert stub.calls["consolidate"] == calls  # nothing new: no call
    before = {b.belief_id for b in await beliefs(uow_factory)}
    rebuilt = await consolidate_tenant(container, T, rebuild=True)
    assert rebuilt["current_beliefs"] == report["current_beliefs"]
    assert not before & {b.belief_id for b in await beliefs(uow_factory)}  # replaced


async def test_over_the_pinned_share_a_page_keeps_the_lines_the_question_finds(
    container, uow_factory, wired
) -> None:
    filler = " and kept a note of it in her travel journal for later"

    def answer(prompt: str) -> list[dict]:
        return [entry("add", [s], t + filler, facet="hobbies") for t, s in ids(prompt).items()]

    wired(StubModel(answer))
    said = [f"I visited harbour market number {i} and bought a souvenir." for i in range(12)]
    for text_ in [*said, "I play the theremin every Tuesday evening."]:
        await say(container, uow_factory, "caroline", text_)
    await settle(container)
    assert len(await beliefs(uow_factory)) == 13
    builder = container.services["context_builder"]
    whole = await builder.build(ASKER, "What does Caroline do?")
    assert len(next(b for b in whole.people if b.block == "caroline").text.splitlines()) == 13
    small = await builder.build(ASKER, "theremin Tuesday evening", token_budget=600)
    page = next(b.text for b in small.people if b.block == "caroline").splitlines()
    assert 0 < len(page) < 13  # over the share: trimmed, not dropped
    assert any("theremin" in line for line in page)  # the line the question is about stays


async def test_a_search_index_rebuild_restores_the_beliefs(container, uow_factory, wired) -> None:
    """Beliefs live in a collection of their own: a whole-store rebuild drops it with the
    others and indexes the current beliefs again (after a model change it starts empty)."""
    from memory_service.tools.reindex import rebuild_search_index

    def answer(prompt: str) -> list[dict]:
        return [entry("add", [short], text_) for text_, short in ids(prompt).items()]

    wired(StubModel(answer))
    await say(container, uow_factory, "caroline", "I keep bees on my balcony.")
    await settle(container)
    [held] = await beliefs(uow_factory)
    report = await rebuild_search_index(container, drop=True)
    assert report.ok and any(name.startswith("beliefs_") for name in report.dropped)
    found = await container.services["retrieval"].retrieve(ASKER, "bees balcony", kinds=("belief",))
    assert held.belief_id in {c.record_id for c in found.candidates}
