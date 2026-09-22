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


def _memory(item_id: str, text: str, *, day: str | None = None, who: str | None = None):
    attributes: dict[str, str] = {}
    if day:
        attributes["observed_at"] = f"{day}T09:30:00+00:00"
    if who:
        attributes["subject"] = f"user:{who}"
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
    assert timeline[0] == "- [memory_id:mem_11] 2023-05-19 Fri caroline: body 11."
    assert sum(rendered.count(f"body {i}.") for i in range(len(memories))) == len(memories)


def test_no_most_relevant_block_when_it_would_be_the_whole_timeline() -> None:
    memories = [
        _memory(f"mem_{i}", f"body {i}.", day=f"2023-05-{10 + i:02d}", who="mel")
        for i in range(MOST_RELEVANT_MAX)
    ]
    rendered = _bundle(memories).render()
    assert "## Most relevant" not in rendered and SHOWN_ABOVE not in rendered
    assert len(_section(rendered, "Memories")) == MOST_RELEVANT_MAX
