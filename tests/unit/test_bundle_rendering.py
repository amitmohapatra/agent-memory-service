"""Unit tests: how a ContextBundle renders its memories.

Two things are asserted here and nowhere else: a memory line names its date, its weekday
and its speaker exactly once, and the highest-ranked memories are repeated above the
chronological timeline without the body being paid for twice.
"""

from __future__ import annotations

import pytest

from memory_service.domain.context_bundle import (
    MOST_RELEVANT_MAX,
    SHOWN_ABOVE,
    ContextBundle,
    ContextItem,
    ConversationWindow,
    EvidenceReport,
)
from memory_service.domain.enums import EvidenceStatus, QueryType, Representation

pytestmark = pytest.mark.unit


def _memory(
    item_id: str,
    text: str,
    *,
    day: str | None = None,
    who: str | None = None,
    predicate: str | None = None,
    **extra: str,
):
    attributes: dict[str, str] = {}
    if day:
        attributes["observed_at"] = f"{day}T09:30:00+00:00"
    if who:
        attributes["subject"] = f"user:{who}"
    if predicate:
        attributes["predicate"] = predicate
    attributes.update(extra)
    return ContextItem(
        item_id=item_id,
        representation=Representation.MEMORY,
        text=text,
        citation=f"memory_id:{item_id}",
        attributes=attributes,
    )


def _bundle(memories: list[ContextItem]) -> ContextBundle:
    return ContextBundle(
        query="q",
        query_type=QueryType.GENERAL_SEMANTIC,
        conversation=ConversationWindow(),
        memories=memories,
        evidence=EvidenceReport(status=EvidenceStatus.COMPLETE),
        token_budget=8000,
        token_estimate=0,
    )


def _section(rendered: str, header: str) -> list[str]:
    body = rendered.split(f"## {header}\n", 1)[1]
    return body.split("\n\n", 1)[0].splitlines()


def test_memory_line_names_date_weekday_and_speaker_once() -> None:
    rendered = _bundle([_memory("mem_1", "I moved to Paris.", day="2023-05-08", who="caroline")])
    line = _section(rendered.render(), "Memories")[0]
    # 2023-05-08 was a Monday; the weekday is what the model derives worst
    assert line == "- [memory_id:mem_1] 2023-05-08 Mon caroline: I moved to Paris."
    assert line.count("2023-05-08") == 1 and line.count("caroline") == 1
    # and no trace of the older doubled form: "[2023-05-08] caroline:" plus the same date
    # and speaker stamped into the text by the ingesting harness
    assert "[2023-05-08]" not in line


def test_memory_line_drops_what_the_memory_does_not_carry() -> None:
    plain = _bundle([_memory("mem_1", "A fact with no date and no subject.")])
    assert _section(plain.render(), "Memories") == [
        "- [memory_id:mem_1] A fact with no date and no subject."
    ]
    dated = _bundle([_memory("mem_1", "Dated but unattributed.", day="2023-05-08")])
    assert _section(dated.render(), "Memories") == [
        "- [memory_id:mem_1] 2023-05-08 Mon Dated but unattributed."
    ]


def test_derived_creation_date_is_not_presented_as_the_date_of_its_events() -> None:
    item = _memory("insight", "Ari opened the workshop in June 2023.", day="2026-09-26")
    item = item.model_copy(update={"attributes": {**item.attributes, "derived": True}})
    line = _section(_bundle([item]).render(), "Memories")[0]
    assert "summary created 2026-09-26 Sat" in line
    assert "opened the workshop in June 2023" in line
    assert line.count(item.text) == 1


def test_memory_line_tolerates_an_unparseable_observed_at() -> None:
    item = _memory("mem_1", "Body.", who="mel")
    broken = item.model_copy(update={"attributes": {**item.attributes, "observed_at": "last May"}})
    assert _section(_bundle([broken]).render(), "Memories") == [
        "- [memory_id:mem_1] last May mel: Body."
    ]


def test_top_ranked_memories_are_repeated_above_the_chronological_timeline() -> None:
    # rank order is the list order; the dates run backwards, so the best-ranked memory is
    # last in the timeline and the strictly chronological rendering buried it
    memories = [
        _memory(f"mem_{i}", f"body {i}.", day=f"2023-05-{30 - i:02d}", who="caroline")
        for i in range(MOST_RELEVANT_MAX + 2)
    ]
    rendered = _bundle(memories).render()
    assert rendered.index("## Most relevant") < rendered.index("## Memories")

    relevant = _section(rendered, "Most relevant")
    assert len(relevant) == MOST_RELEVANT_MAX
    assert relevant[0] == "- [memory_id:mem_0] 2023-05-30 Tue caroline: body 0."
    assert [line.split("]")[0] for line in relevant] == [
        f"- [memory_id:mem_{i}" for i in range(MOST_RELEVANT_MAX)
    ]

    timeline = _section(rendered, "Memories")
    assert len(timeline) == len(memories)
    assert [line.split()[2] for line in timeline] == sorted(line.split()[2] for line in timeline)
    # every memory keeps its place, its date and its speaker in the timeline, but the ten
    # printed in full above appear there as a pointer instead of a second copy
    assert sum(SHOWN_ABOVE in line for line in timeline) == MOST_RELEVANT_MAX
    # the head is 30 now, so the first chronological line that is not a pointer moves
    assert timeline[0].startswith("- [memory_id:mem_")
    assert sum(rendered.count(f"body {i}.") for i in range(len(memories))) == len(memories)


def test_no_most_relevant_block_when_it_would_be_the_whole_timeline() -> None:
    memories = [
        _memory(f"mem_{i}", f"body {i}.", day=f"2023-05-{10 + i:02d}", who="mel")
        for i in range(MOST_RELEVANT_MAX)
    ]
    rendered = _bundle(memories).render()
    assert "## Most relevant" not in rendered and SHOWN_ABOVE not in rendered
    assert len(_section(rendered, "Memories")) == MOST_RELEVANT_MAX


# --------------------------------------------------------------------------------------
# Gathering the several values of one multi-valued slot (D6 step 4)
# --------------------------------------------------------------------------------------


def test_several_values_of_one_multi_valued_slot_render_as_one_dated_block() -> None:
    # the fragmentation bucket: four separate lines read as four unrelated facts, and a
    # question needing all of them is answered from whichever was read last
    memories = [
        _memory(
            "mem_1",
            "Jon ran the charity race.",
            day="2023-05-02",
            who="jon",
            predicate="participated_in",
        ),
        _memory(
            "mem_2",
            "Jon joined the hackathon.",
            day="2023-06-11",
            who="jon",
            predicate="participated_in",
        ),
    ]
    timeline = _section(_bundle(memories).render(), "Memories")
    assert len(timeline) == 3  # one heading line plus one statement per value
    assert timeline[0] == (
        "- [memory_id:mem_1; memory_id:mem_2] user:jon — participated in (source statements):"
    )
    assert timeline[1:] == [
        "[observed 2023-05-02] Jon ran the charity race.",
        "[observed 2023-06-11] Jon joined the hackathon.",
    ]


def test_a_single_valued_slot_is_never_gathered() -> None:
    # the newest value is the answer and the older one is superseded history; printing them
    # as a set of current values would say the person lives in two cities
    memories = [
        _memory("mem_1", "Jon lives in Berlin.", day="2023-05-02", who="jon", predicate="lives_in"),
        _memory("mem_2", "Jon lives in Lisbon.", day="2023-06-11", who="jon", predicate="lives_in"),
    ]
    timeline = _section(_bundle(memories).render(), "Memories")
    assert len(timeline) == 2
    assert all(line.startswith("- [memory_id:mem_") for line in timeline)


def test_one_value_alone_keeps_its_own_line() -> None:
    memories = [
        _memory(
            "mem_1", "Jon ran the race.", day="2023-05-02", who="jon", predicate="participated_in"
        ),
        _memory("mem_2", "Jon likes rain.", day="2023-06-11", who="jon", predicate="likes"),
    ]
    timeline = _section(_bundle(memories).render(), "Memories")
    assert timeline == [
        "- [memory_id:mem_1] 2023-05-02 Tue jon: Jon ran the race.",
        "- [memory_id:mem_2] 2023-06-11 Sun jon: Jon likes rain.",
    ]


def test_the_same_slot_on_two_subjects_stays_two_blocks() -> None:
    memories = [
        _memory("mem_1", "Jon ran.", day="2023-05-02", who="jon", predicate="participated_in"),
        _memory("mem_2", "Mel ran.", day="2023-05-03", who="mel", predicate="participated_in"),
        _memory("mem_3", "Jon swam.", day="2023-05-04", who="jon", predicate="participated_in"),
    ]
    rendered = _bundle(memories).render()
    timeline = _section(rendered, "Memories")
    # Jon's two are gathered at his oldest place; Mel's single value keeps its own line
    assert timeline[0].startswith("- [memory_id:mem_1; memory_id:mem_3] user:jon")
    assert "user:mel —" not in rendered
    assert sum("Mel ran." in line for line in timeline) == 1


def test_a_gathered_block_keeps_the_unverified_warning_of_any_member() -> None:
    # an aggregate is no more trustworthy than its least trustworthy source
    memories = [
        _memory("mem_1", "Jon ran.", day="2023-05-02", who="jon", predicate="participated_in"),
        _memory(
            "mem_2",
            "Jon swam.",
            day="2023-05-04",
            who="jon",
            predicate="participated_in",
            provider="llm",
        ),
    ]
    timeline = _section(_bundle(memories).render(), "Memories")
    assert "model-extracted, unverified" in timeline[0]


def test_a_derived_memory_is_already_an_aggregate_and_is_not_gathered() -> None:
    memories = [
        _memory("mem_1", "Jon ran.", day="2023-05-02", who="jon", predicate="participated_in"),
        _memory(
            "mem_2",
            "Jon — participated in …",
            day="2023-05-04",
            who="jon",
            predicate="participated_in",
            derived="true",
        ),
    ]
    timeline = _section(_bundle(memories).render(), "Memories")
    assert len(timeline) == 2
    assert all(line.startswith("- [memory_id:mem_") for line in timeline)


def test_a_promoted_memory_is_not_gathered_so_every_body_appears_once() -> None:
    # enough memories that the "Most relevant" block exists; the promoted ones stand in the
    # timeline as a pointer, and gathering them there would print their body a second time
    memories = [
        _memory(
            f"mem_{i}",
            f"body {i}.",
            day=f"2023-05-{30 - i:02d}",
            who="jon",
            predicate="participated_in",
        )
        for i in range(MOST_RELEVANT_MAX + 2)
    ]
    rendered = _bundle(memories).render()
    assert "## Most relevant" in rendered
    timeline = _section(rendered, "Memories")
    assert sum(SHOWN_ABOVE in line for line in timeline) == MOST_RELEVANT_MAX
    # the two that were not promoted are gathered into one block; every body still once
    assert sum(rendered.count(f"body {i}.") for i in range(len(memories))) == len(memories)
