"""Beliefs (consolidation phase 1), the parts that need no database: who a new fact is about,
what the model's keep/update/retire/add/skip entries become (the code compares the dates, not
the model), the citation verdicts, the per-person page and its place in the token budget."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from memory_service.domain.beliefs import Belief, person_name
from memory_service.domain.context_bundle import (
    ProfileBlockView,
    profile_section,
)
from memory_service.domain.enums import Lifetime, MemoryType, ScopeLevel, Visibility
from memory_service.domain.evidence import EvidenceRef, EvidenceSource
from memory_service.domain.memory import CanonicalMemory, Scope, TemporalState
from memory_service.modules.context.sections import Pinned, within_budget
from memory_service.modules.ingestion.hierarchy import estimate_tokens
from memory_service.modules.memory.consolidation import (
    checked,
    fact_scopes,
    plan,
    unit_scope,
    units_touched,
)

T = "acme"
SCOPE = Scope(level=ScopeLevel.TENANT, tenant_id=T, workspace_id="ws")
UNIT = unit_scope(SCOPE)
KEYS = [f"tenant:{T}"]
DAY = datetime(2023, 5, 8, tzinfo=UTC)


def fact(text: str, *, by: str = "caroline", day: int = 0, **extra: object) -> CanonicalMemory:
    when = DAY + timedelta(days=day)
    return CanonicalMemory(
        tenant_id=T,
        scope=SCOPE,
        visibility=Visibility.TENANT,
        owner_principal=f"user:{by}",
        lifetime=extra.pop("lifetime", Lifetime.LONG_TERM),  # type: ignore[arg-type]
        memory_type=MemoryType.SEMANTIC,
        content=text,
        normalized_hash=text,
        subject=extra.pop("subject", f"user:{by}"),  # type: ignore[arg-type]
        temporal=TemporalState(observed_at=when),
        evidence=[
            EvidenceRef(source_type=EvidenceSource.MESSAGE, source_id="msg_1", observed_at=when)
        ],
        system_metadata={"visibility_keys": KEYS, **extra},
        created_at=when,
        updated_at=when,
    )


def belief(text: str, *, day: int = 0, facet: str = "work", sources: int = 1) -> Belief:
    when = DAY + timedelta(days=day)
    return Belief.new(
        tenant_id=T,
        scope_key=UNIT,
        person_key="user:caroline",
        visibility_keys=KEYS,
        text=text,
        facet=facet,
        seen=(when, when),
        source_ids=[f"mem_{i}" for i in range(sources)],
        now=when,
    )


# -- who a fact is about --------------------------------------------------------------------


def test_a_fact_touches_its_speaker_and_the_principals_it_names() -> None:
    mine = fact("I moved from Sweden four years ago.")
    theirs = fact("Caroline told me she loves painting.", by="melanie")
    units = units_touched([mine, theirs], known={"user:caroline", "user:melanie"})
    assert units == {(UNIT, "user:caroline"), (UNIT, "user:melanie")}
    # a capitalised word that names no principal of the tenant touches nobody
    assert units_touched([fact("Sweden was cold.", by="melanie")], known=set()) == {
        (UNIT, "user:melanie")
    }


def test_short_term_and_derived_memories_and_agents_touch_nobody() -> None:
    short = fact("Remind me at five.", lifetime=Lifetime.SHORT_TERM)
    derived = fact("An insight.", source_revisions={"mem_1": 1})
    agent = fact("I will look it up.", by="x", subject="agent:x")
    agent = agent.model_copy(update={"owner_principal": "agent:x"})
    assert units_touched([short, derived, agent], known={"user:caroline"}) == set()


def test_a_persons_own_and_shared_facts_are_one_unit_and_a_conversations_another() -> None:
    own = Scope(level=ScopeLevel.USER, tenant_id=T, workspace_id="ws", user_id="caroline")
    assert unit_scope(own) == UNIT == unit_scope(SCOPE) == f"t={T}/l=WORKSPACE/workspace=ws"
    assert fact_scopes(T, UNIT, "user:caroline") == sorted([UNIT, SCOPE.key(), own.key()])
    thread = Scope(level=ScopeLevel.THREAD, tenant_id=T, workspace_id="ws", thread_id="thr_1")
    assert unit_scope(thread) == thread.key()
    assert fact_scopes(T, thread.key(), "user:caroline") == [thread.key()]


def test_person_name_is_the_principal_without_its_kind() -> None:
    assert person_name("user:caroline") == "caroline"


# -- the model's entries become a plan ------------------------------------------------------


def test_keep_adds_the_fact_as_a_source() -> None:
    b = belief("Works as a counselor (since May 2023).")
    f = fact("I still work as a counselor.", day=3)
    p = plan([{"op": "keep", "facts": ["F1"], "belief": "B1", "text": "", "facet": ""}], [b], [f])
    assert p.keeps == {b.belief_id: [f.memory_id]} and not p.lines


def test_update_replaces_the_belief_with_its_sources_carried_over() -> None:
    b = belief("Works at Microsoft (Jan 2025).", sources=2)
    f = fact("I started at Google in March.", day=30)
    entry = {
        "op": "update",
        "facts": ["F1"],
        "belief": "B1",
        "text": "Works at Google.",
        "facet": "",
    }
    p = plan([entry], [b], [f])
    [line] = p.lines
    assert line.replaces == b and line.history_of is None
    assert line.text == "Works at Google." and line.facet == "work"  # facet kept when blank
    assert line.source_ids == (*b.source_ids, f.memory_id)


def test_an_older_fact_never_overrides_a_newer_belief() -> None:
    b = belief("Works at Google (since Mar 2026).", day=30)
    old = fact("I worked at Microsoft.", day=1)
    entry = {
        "op": "update",
        "facts": ["F1"],
        "belief": "B1",
        "text": "Worked at Microsoft.",
        "facet": "",
    }
    retire = {"op": "retire", "facts": ["F1"], "belief": "B1", "text": "", "facet": ""}
    p = plan([entry, retire], [b], [old])
    [line] = p.lines
    assert line.replaces is None and line.history_of == b  # an earlier value, not the current
    assert line.source_ids == (old.memory_id,)
    assert p.retires == {}  # nor may it end the newer belief


def test_retire_ends_the_belief_at_the_facts_date() -> None:
    b = belief("Lives in Sweden.")
    f = fact("I moved away from Sweden.", day=9)
    p = plan([{"op": "retire", "facts": ["F1"], "belief": "B1", "text": "", "facet": ""}], [b], [f])
    assert p.retires == {b.belief_id: (f.temporal.observed_at, [f.memory_id])}


def test_add_cites_only_facts_of_the_batch_and_skip_writes_nothing() -> None:
    f1, f2 = fact("I paint sunsets.", day=1), fact("I like tea.", day=1)
    entries = [
        {
            "op": "add",
            "facts": ["F1", "F7", f2.memory_id],
            "belief": "",
            "text": "Paints.",
            "facet": "hobbies",
        },
        {"op": "add", "facts": ["F9"], "belief": "", "text": "Invented.", "facet": "x"},
        {"op": "add", "facts": ["F2"], "belief": "", "text": "  ", "facet": "x"},
        {"op": "skip", "facts": ["F2"], "belief": "", "text": "", "facet": ""},
        {"op": "keep", "facts": ["F2"], "belief": "B4", "text": "", "facet": ""},
    ]
    p = plan(entries, [], [f1, f2])
    [line] = p.lines
    assert line.text == "Paints." and line.source_ids == (f1.memory_id,)
    assert not p.keeps and not p.retires


def test_a_person_with_many_beliefs_still_gets_new_ones() -> None:
    """No working bound: 200 current beliefs (past the old 60) take another add."""
    current = [belief(f"Line {i}.") for i in range(200)]
    add = {"op": "add", "facts": ["F1"], "belief": "", "text": "Likes tea.", "facet": "values"}
    found = plan([add], current, [fact("I like tea.", day=1)])
    assert [x.text for x in found.lines] == ["Likes tea."] and found.over_ceiling == 0


def test_only_the_safety_ceiling_stops_an_add_and_says_how_many(monkeypatch) -> None:
    from memory_service.modules.memory import consolidation

    monkeypatch.setattr(consolidation, "BELIEFS_CEILING", 3)
    current = [belief(f"Line {i}.") for i in range(3)]
    f = fact("I like tea.", day=1)
    add = {"op": "add", "facts": ["F1"], "belief": "", "text": "Likes tea.", "facet": "values"}
    found = plan([add], current, [f])
    assert found.lines == [] and found.over_ceiling == 1  # counted, and logged by the run
    retire = {"op": "retire", "facts": ["F1"], "belief": "B1", "text": "", "facet": ""}
    assert [x.text for x in plan([retire, add], current, [f]).lines] == ["Likes tea."]


# -- the citation check ---------------------------------------------------------------------


def test_verdicts_keep_supported_drop_unsupported_and_return_partial_for_one_rewrite() -> None:
    raw = {
        "verdicts": [
            {"line": "L1", "label": "SUPPORTED"},
            {"line": "L2", "label": "UNSUPPORTED"},
            {"line": "L3", "label": "PARTIAL"},
            {"line": "L9", "label": "SUPPORTED"},
        ]
    }
    supported, partial = checked(raw, 4)
    assert supported == {0} and partial == {2}  # L4 got no verdict: dropped
    assert checked(None, 3) == (set(), set())  # no check, nothing stored


# -- the pinned profile in the token budget ------------------------------------------------


def test_without_people_the_pinned_profile_is_what_it_was() -> None:
    profile = [ProfileBlockView(block="user", text="- likes tea", version=1)]
    kept, used = within_budget(Pinned(profile=list(profile)), 8000)
    assert kept.profile == profile and used == estimate_tokens(profile_section(profile) or "")


# -- the answer's room ----------------------------------------------------------------------


async def test_the_consolidation_use_gets_room_for_its_answer_and_others_keep_the_ceiling() -> None:
    from memory_service.config.constants import LLM
    from memory_service.modules.memory.consolidation import CONSOLIDATION_USE
    from tests.support_llm import mocked_gateway

    with mocked_gateway(['{"entries": []}']) as gw:
        assist = gw.assist(uses=[CONSOLIDATION_USE, "summaries"])
        schema = {"type": "object"}
        await assist.structured(
            CONSOLIDATION_USE, system="s", user="u", schema=schema, max_tokens=99_999
        )
        await assist.structured("summaries", system="s", user="u", schema=schema, max_tokens=99_999)
        sent = [p.get("max_tokens") or p.get("max_completion_tokens") for p in gw.prompts()]
    assert sent == [LLM.long_max_tokens + LLM.thinking_room, LLM.max_tokens + LLM.thinking_room]
    assert LLM.long_max_tokens > LLM.max_tokens
