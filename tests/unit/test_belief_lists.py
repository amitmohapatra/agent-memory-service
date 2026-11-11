"""List beliefs, the parts that need no database: one belief per recurring or countable
thing of a person, every instance named by the model with the fact it comes from and dated
by the code from that fact. The second instance turns a belief into a list, later ones
extend it with its sources carried over, a list past a line's size continues in a next part
of the same title, and the page shows lists first."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from memory_service.domain.beliefs import LIST_FACET, Belief
from memory_service.domain.memory import CanonicalMemory
from memory_service.modules.memory.consolidation import MAX_LINE_CHARS, listed, plan
from memory_service.modules.profile.service import person_page
from tests.unit.test_consolidation import DAY, belief, fact

TRIPS = "Trips taken"


def item(text: str, ref: str) -> dict:
    return {"text": text, "fact": ref}


def entry(op: str, facts: list[str], items: list[dict], *, text: str = TRIPS, on: str = "") -> dict:
    return {"op": op, "facts": facts, "belief": on, "text": text, "facet": "places", "items": items}


def held(text: str, sources: list[CanonicalMemory], *, facet: str = "places") -> Belief:
    """A current belief resting on ``sources``."""
    return belief(text, facet=facet).model_copy(
        update={"source_ids": [m.memory_id for m in sources]}
    )


def written(entries: list[dict], beliefs: list[Belief], facts: list[CanonicalMemory], *old):
    """The lines the entries make, written out with every cited memory at hand."""
    sources = {m.memory_id: m for m in (*facts, *old)}
    return [part for line in plan(entries, beliefs, facts).lines for part in listed(line, sources)]


# -- recognised at the second instance, dated by the code ------------------------------------


def test_the_second_instance_turns_a_belief_into_a_list_dated_from_its_facts() -> None:
    paris = fact("I went to Paris.")
    rome = fact(
        "I flew to Rome last week.",
        day=40,
        dated_mentions=[{"text": "last week", "date": "2023-06-05..2023-06-11"}],
    )
    was = held("Went to Paris (May 2023).", [paris])
    turned = entry("update", ["F1"], [item("Paris", "B1"), item("Rome", "F1")], on="B1")
    [line] = written([turned], [was], [rome], paris)
    # the first instance has only the day it was said; the second the date its words resolve to
    assert line.text == "Trips taken: Paris (said 2023-05-08); Rome (2023-06-05..2023-06-11)"
    assert line.facet == LIST_FACET
    assert line.source_ids == (paris.memory_id, rome.memory_id)
    assert line.folds == (was,) and line.replaces is None  # the belief is now an item of it


def test_an_instance_is_dated_by_the_date_its_fact_names_never_by_the_models_words() -> None:
    named = fact("The concert was great.", day=3)
    named = named.model_copy(
        update={
            "temporal": named.temporal.model_copy(
                update={"valid_from": datetime(2023, 4, 30, tzinfo=UTC)}
            )
        }
    )
    two = fact(
        "Yesterday I saw a play, and last month an opera.",
        day=5,
        dated_mentions=[
            {"text": "yesterday", "date": "2023-05-12"},
            {"text": "last month", "date": "2023-04-01..2023-04-30"},
        ],
    )
    plain = fact("We watched a musical.", day=7)  # holds from the day it was said, as stored
    plain = plain.model_copy(
        update={
            "temporal": plain.temporal.model_copy(update={"valid_from": DAY + timedelta(days=7)})
        }
    )
    made = entry(
        "add",
        ["F1", "F2", "F3"],
        [item("concert", "F1"), item("play", "F2"), item("opera", "F2"), item("musical", "F3")],
        text="Shows seen",
    )
    [line] = written([made], [], [named, two, plain])
    assert line.text == (
        "Shows seen: concert (2023-04-30); play (said 2023-05-13); opera (said 2023-05-13); "
        "musical (said 2023-05-15)"
    )  # two dates in one fact: which is whose is not the code's to guess, so the day said


# -- later instances extend it ---------------------------------------------------------------


def test_a_later_instance_extends_the_list_with_its_sources_carried_over() -> None:
    paris, rome = fact("I went to Paris."), fact("I went to Rome.", day=40)
    berlin = fact("Back from Berlin!", day=80)
    trips = held(
        "Trips taken: Paris (said 2023-05-08); Rome (said 2023-06-17)",
        [paris, rome],
        facet=LIST_FACET,
    )
    more = entry("update", ["F1"], [item("Berlin", "F1")], text="", on="B1")
    [line] = written([more], [trips], [berlin], paris, rome)
    assert line.text == trips.text + "; Berlin (said 2023-07-27)"  # what it held, untouched
    assert line.replaces == trips and line.facet == LIST_FACET
    assert line.source_ids == (paris.memory_id, rome.memory_id, berlin.memory_id)


def test_an_older_instance_said_late_joins_the_list_instead_of_becoming_an_earlier_value() -> None:
    rome = fact("I went to Rome.", day=40)
    late = fact("Oh, and Paris, before that.", day=2)  # said before the list's newest fact
    trips = held("Trips taken: Rome (said 2023-06-17)", [rome], facet=LIST_FACET).model_copy(
        update={"last_seen_at": DAY + timedelta(days=40)}
    )
    [line] = written(
        [entry("update", ["F1"], [item("Paris", "F1")], text="", on="B1")], [trips], [late], rome
    )
    assert line.replaces == trips and line.history_of is None
    assert line.text.endswith("; Paris (said 2023-05-10)")


def test_two_entries_for_one_list_in_a_run_are_one_line() -> None:
    paris, rome, oslo = fact("Paris."), fact("Rome.", day=1), fact("Oslo.", day=2)
    trips = held("Trips taken: Paris (said 2023-05-08)", [paris], facet=LIST_FACET)
    lines = written(
        [
            entry("update", ["F1"], [item("Rome", "F1")], text="", on="B1"),
            entry("update", ["F2"], [item("Oslo", "F2")], text="", on="B1"),
        ],
        [trips],
        [rome, oslo],
        paris,
    )
    assert [x.text for x in lines] == [
        "Trips taken: Paris (said 2023-05-08); Rome (said 2023-05-09); Oslo (said 2023-05-10)"
    ]


def test_an_item_that_cites_nothing_known_is_left_out_and_a_list_needs_a_new_fact() -> None:
    paris, rome = fact("Paris."), fact("Rome.", day=1)
    was = held("Went to Paris.", [paris])
    stray = entry("add", ["F1"], [item("Rome", "F1"), item("Mars", "F9"), item("", "F1")])
    [line] = written([stray], [was], [rome], paris)
    assert line.text == "Trips taken: Rome (said 2023-05-09)"
    # only beliefs, no fact of this run: nothing to fold a list from
    assert written([entry("add", [], [item("Paris", "B1")])], [was], [rome], paris) == []


# -- a list that grows long ------------------------------------------------------------------


def test_a_list_past_a_lines_size_continues_in_a_next_part_of_the_same_title() -> None:
    seen = [fact(f"Museum {n}.", day=n) for n in range(12)]
    full = "Museums visited: " + "; ".join(
        f"the long-named museum number {n} (said x)" for n in range(9)
    )
    assert MAX_LINE_CHARS - 60 < len(full) <= MAX_LINE_CHARS
    first = held(full, seen[:9], facet=LIST_FACET)
    new = fact("Another museum today.", day=20)
    more = entry("update", ["F1"], [item("the long-named museum number 9", "F1")], text="", on="B1")
    [part] = written([more], [first], [new], *seen)
    # decided by size alone: the full part is left as it is, the list goes on beside it
    assert part.replaces is None and part.source_ids == (new.memory_id,)
    assert part.text == "Museums visited: the long-named museum number 9 (said 2023-05-28)"
    assert len(part.text) <= MAX_LINE_CHARS
    # the next instance goes to the part with room, whichever part the model names
    second = held(part.text, [new], facet=LIST_FACET)
    newer = fact("One more museum.", day=30)
    [grown] = written(
        [entry("update", ["F1"], [item("museum 10", "F1")], text="", on="B1")],
        [first, second],
        [newer],
        *seen,
        new,
    )
    assert grown.replaces == second and grown.text == second.text + "; museum 10 (said 2023-06-07)"


def test_a_first_list_too_long_for_a_line_is_written_as_parts() -> None:
    facts = [fact(f"Recipe {n}.", day=n) for n in range(14)]
    made = entry(
        "add",
        [f"F{n + 1}" for n in range(14)],
        [item(f"a recipe with a rather long name, number {n}", f"F{n + 1}") for n in range(14)],
        text="Recipes made",
    )
    parts = written([made], [], facts)
    assert len(parts) > 1 and all(len(p.text) <= MAX_LINE_CHARS for p in parts)
    assert all(p.text.startswith("Recipes made: ") and p.facet == LIST_FACET for p in parts)
    assert sum(p.text.count("(said ") for p in parts) == 14  # every instance, once
    assert [s for p in parts for s in p.source_ids] == [m.memory_id for m in facts]


# -- the page --------------------------------------------------------------------------------


def test_the_page_shows_lists_first() -> None:
    who = belief("Is a counselor.", facet="identity")
    trips = belief("Trips taken: Paris (said 2023-05-08); Rome (said 2023-06-17)", facet=LIST_FACET)
    hobby = belief("Paints sunsets.", facet="hobbies", day=1)
    assert person_page([hobby, who, trips], []).splitlines() == [
        f"- {trips.text}",
        "- Is a counselor.",
        "- Paints sunsets.",
    ]
