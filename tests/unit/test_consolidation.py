"""Beliefs (consolidation phase 1), the parts that need no database: who a new fact is about,
what the model's keep/update/retire/add/skip entries become (the code compares the dates, not
the model), the citation verdicts, the per-person page and its place in the token budget."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory_service.domain.beliefs import FACETS, Belief, person_name
from memory_service.domain.context_bundle import (
    ProfileBlockView,
    people_section,
    pinned_sections,
    profile_section,
)
from memory_service.domain.enums import Lifetime, MemoryType, ScopeLevel, Visibility
from memory_service.domain.evidence import EvidenceRef, EvidenceSource
from memory_service.domain.memory import CanonicalMemory, Scope, TemporalState
from memory_service.domain.profile import PROFILE_BLOCK_MAX_CHARS
from memory_service.modules.context.sections import Pinned, within_budget
from memory_service.modules.ingestion.hierarchy import estimate_tokens
from memory_service.modules.memory.consolidation import (
    checked,
    fact_scopes,
    plan,
    unit_scope,
    units_touched,
)
from memory_service.modules.profile.service import PersonPage, person_page

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


# -- the per-person page --------------------------------------------------------------------


def test_the_page_groups_lines_by_facet_and_shows_earlier_values_apart() -> None:
    work = belief("Works at Google (since Mar 2026).", day=30, sources=3)
    hobby = belief("Paints sunsets (May 2023).", facet="hobbies")
    old = belief("Works at Microsoft (Jan 2025).").model_copy(
        update={"superseded_by": work.belief_id, "valid_to": work.first_seen_at}
    )
    page = person_page([hobby, work], [old])
    assert page.splitlines() == [
        "- Works at Google (since Mar 2026).",
        "- Paints sunsets (May 2023).",
        "Earlier values (since replaced):",
        "- Works at Microsoft (Jan 2025). (until 2023-06-07)",
    ]


def test_a_version_a_refinement_grew_from_is_not_an_earlier_value() -> None:
    grown = belief("Paints lakes; sold two paintings at the fair (Jul 2023).", day=60)
    seed = belief("Paints lakes").model_copy(update={"superseded_by": grown.belief_id})
    assert person_page([grown], [seed]) == "- " + grown.text


# -- the page in the token budget -----------------------------------------------------------


def test_without_people_the_pinned_profile_is_what_it_was() -> None:
    profile = [ProfileBlockView(block="user", text="- likes tea", version=1)]
    kept, used = within_budget(Pinned(profile=list(profile)), 8000)
    assert kept.profile == profile and used == estimate_tokens(profile_section(profile) or "")


def _page(name: str, beliefs: list[Belief], earlier: list[Belief] | None = None) -> PersonPage:
    return PersonPage(name, tuple(beliefs), tuple(earlier or ()))


def _many(n: int, *, chars: int = 150) -> list[Belief]:
    """``n`` lines spread over every facet, values written last as the model often does."""
    return [
        belief(
            f"{FACETS[i % len(FACETS)]} line {i} " + "z" * chars,
            facet=FACETS[i % len(FACETS)],
            day=i,
            sources=1 + i % 3,
        )
        for i in range(n)
    ]


def test_a_page_over_4000_characters_is_shown_whole_when_the_budget_allows() -> None:
    lines = _many(40)
    assert sum(len(b.text) for b in lines) > PROFILE_BLOCK_MAX_CHARS
    kept, used = within_budget(Pinned(people=[_page("caroline", lines)]), 8000)
    [page] = kept.pages
    assert len(page.text.splitlines()) == 40 and len(page.text) > PROFILE_BLOCK_MAX_CHARS
    facets = [line.split()[1] for line in page.text.splitlines()]
    assert list(dict.fromkeys(facets)) == list(FACETS)  # identity, values, then the rest
    assert used == 0  # not charged to the memories (``PEOPLE_CHARGED``)


def test_over_the_budget_the_lines_relevant_to_the_question_stay_and_every_facet_keeps_one() -> (
    None
):
    lines = _many(60)
    relevant = [lines[7], lines[23], lines[41]]  # what the question's search ranked
    old = belief("An earlier value.")
    kept, used = within_budget(
        Pinned(people=[_page("caroline", lines, [old])]),
        3000,
        relevance={b.belief_id: rank for rank, b in enumerate(relevant)},
    )
    [page] = kept.pages
    shown = page.text.splitlines()
    assert used <= 1500 and len(shown) < 60
    assert all(f"- {b.text}" in shown for b in relevant)
    assert "Earlier values" not in page.text  # earlier values go before any current line
    assert {line.split()[1] for line in shown} == set(FACETS)  # no facet dropped whole
    worth = lambda b: (b.proof_count, b.last_seen_at)  # noqa: E731
    others = [b for b in lines if b not in relevant]
    dropped = [b for b in others if f"- {b.text}" not in shown]
    kept_ = [b for b in others if f"- {b.text}" in shown]
    # the least-supported go first, never the end of the page: what stays is worth at least
    # what went, but for a facet's last line
    lone = {b.facet for b in kept_ if sum(x.facet == b.facet for x in kept_) == 1}
    assert dropped and all(
        worth(k) >= worth(d) for k in kept_ if k.facet not in lone for d in dropped
    )


def test_people_come_after_the_profile_and_only_while_they_fit() -> None:
    profile = [ProfileBlockView(block="user", text="- likes tea", version=1)]
    big = _page("caroline", [belief("y" * 380, facet="work") for _ in range(8)])
    small = _page("melanie", [belief("paints", facet="hobbies")])
    kept, used = within_budget(Pinned(profile=list(profile), people=[big, small]), 1000)
    assert kept.profile == profile and "melanie" in [b.block for b in kept.pages]
    assert used == estimate_tokens(profile_section(profile) or "")  # the pages ride on top
    assert estimate_tokens(people_section(kept.pages) or "") <= 500  # within their own share


@pytest.mark.parametrize("charged", [False, True])
def test_a_big_page_never_displaces_or_trims_the_explicit_blocks_or_the_standing_ones(
    monkeypatch, charged: bool
) -> None:
    """Personalisation as before: every hand-written block and every implicit preference or
    rule the profile holds stays whole; the pages get what is left of the pinned share after
    them when charged, else a share of their own."""
    from memory_service.modules.context import sections

    monkeypatch.setattr(sections, "PEOPLE_CHARGED", charged)
    profile = [
        ProfileBlockView(block=name, text=f"- {name} " + "w" * 900, version=1)
        for name in ("user", "agent", "workspace", "standing_rules", "preferences", "about_user")
    ]
    alone, cost = within_budget(Pinned(profile=list(profile)), 4000)
    assert alone.profile == profile  # the whole profile fits the share on its own
    page = _page("caroline", _many(60))
    kept, used = within_budget(Pinned(profile=list(profile), people=[page]), 4000)
    assert kept.profile == profile  # nothing displaced, nothing trimmed
    assert kept.pages and len(kept.pages[0].text.splitlines()) < 60  # the page gave way
    pages = estimate_tokens(people_section(kept.pages) or "")
    if charged:
        assert used == cost + pages <= 2000  # the page in what the profile left of the share
    else:
        assert used == cost and pages <= 2000  # the page in a share of its own
    # a share the whole profile does not fit: the standing blocks, exactly as without a page
    squeezed, _ = within_budget(Pinned(profile=list(profile), people=[page]), 1600)
    standing = [b for b in profile if b.block in ("standing_rules", "preferences", "about_user")]
    assert (
        squeezed.profile
        == standing
        == within_budget(Pinned(profile=list(profile)), 1600)[0].profile
    )


def test_a_context_with_only_pages_has_no_profile_heading() -> None:
    """The harness adds its note on remembered rules when the context has ``## Profile``:
    a person's page alone must not switch it on."""
    kept, _ = within_budget(Pinned(people=[_page("caroline", _many(3))]), 8000)
    rendered = "\n\n".join(pinned_sections(kept.profile, None, [], None, kept.pages))
    assert rendered.startswith("## People\n### caroline\n") and "## Profile\n" not in rendered
    profile = [ProfileBlockView(block="user", text="- likes tea", version=1)]
    both = "\n\n".join(pinned_sections(profile, None, [], None, kept.pages))
    assert both.index("## Profile\n") < both.index("## People\n")


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


def test_the_people_pages_cost_is_decided_in_one_place(monkeypatch) -> None:
    from memory_service.modules.context import sections

    profile = [ProfileBlockView(block="user", text="- likes tea", version=1)]
    page = _page("caroline", _many(10))
    on_top, free = within_budget(Pinned(profile=list(profile), people=[page]), 8000)
    shown = estimate_tokens(profile_section(profile) or "")
    assert free == shown  # the default: the memories keep the rest of the budget
    monkeypatch.setattr(sections, "PEOPLE_CHARGED", True)
    charged, used = within_budget(Pinned(profile=list(profile), people=[page]), 8000)
    assert charged.pages == on_top.pages  # the same page, shown whole
    assert used == shown + estimate_tokens(people_section(charged.pages) or "")  # memories pay
