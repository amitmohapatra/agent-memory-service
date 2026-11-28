"""``demand.said_once``: each statement once, however it was stored - a profile line, the memory
read from a message, and the message kept as said are one statement."""

from __future__ import annotations

from datetime import UTC, datetime

from memory_service.domain.context_bundle import (
    ContextBundle,
    ContextItem,
    ConversationWindow,
    EvidenceReport,
    ProfileBlockView,
)
from memory_service.domain.enums import EvidenceStatus, QueryType, Representation
from memory_service.domain.evidence import EvidenceRef, EvidenceSource
from memory_service.modules.context.demand import said_once

ROLE = "Priya runs allocation for the Northeast district."
TABLE = "Priya wants stock numbers always shown as a table."


def _item(item_id: str, text: str, message: str, **attributes: object) -> ContextItem:
    return ContextItem(
        item_id=item_id,
        representation=Representation.MEMORY,
        text=text,
        citation=f"memory_id:{item_id}",
        token_estimate=10,
        evidence=[
            EvidenceRef(
                source_type=EvidenceSource.MESSAGE,
                source_id=message,
                message_id=message,
                observed_at=datetime(2026, 11, 28, tzinfo=UTC),
            )
        ],
        attributes=attributes,
    )


def _bundle(memories: list[ContextItem], *lines: tuple[str, str]) -> ContextBundle:
    return ContextBundle(
        query="Should we move units to Boston?",
        query_type=QueryType.GENERAL_SEMANTIC,
        conversation=ConversationWindow(),
        evidence=EvidenceReport(status=EvidenceStatus.COMPLETE),
        token_budget=1000,
        token_estimate=100,
        profile=[ProfileBlockView(block=block, text=text, version=1) for block, text in lines],
        memories=memories,
    )


def test_a_profile_field_states_its_value() -> None:
    """The ``user`` block's lines are ``field: value``: the value is the statement."""
    bundle = said_once(_bundle([_item("mem_1", ROLE, "msg_1")], ("user", f"role: {ROLE}")))
    assert bundle.memories == []


def test_what_a_model_read_from_a_message_goes_when_the_message_is_shown_as_said() -> None:
    """A message kept as said and what a model read from it say one thing: the message is
    shown as said, the reading is not (the turn holds every reading of it whole, as ranking
    lets it stand for the statement, ``memory_ranking._standing_for``; LongMemEval: the
    round, found by its facts, is what is read) - and a reading of a message not shown
    stays."""
    read = _item("mem_read", ROLE, "msg_1", category="attribute", provider="llm")
    turn = _item("mem_turn", "Hi, I'm Priya. I run allocation for the Northeast district.",
                 "msg_1", category="verbatim_turn")  # fmt: skip
    other = _item("mem_other", "Boston sells nine a week.", "msg_2", category="source_fact",
                  provider="llm")  # fmt: skip
    bundle = said_once(_bundle([read, turn, other]))
    assert [m.item_id for m in bundle.memories] == ["mem_turn", "mem_other"]
    assert bundle.token_estimate == 90


def test_a_statement_listed_in_the_profile_takes_its_message_with_it() -> None:
    """A preference listed with its date in the profile is the statement: neither the memory
    nor the message it was read from is shown again."""
    pref = _item("mem_pref", TABLE, "msg_3", category="rule", provider="llm")
    turn = _item("mem_turn", "From now on, always show stock numbers as a table.", "msg_3",
                 category="verbatim_turn")  # fmt: skip
    bundle = said_once(_bundle([pref, turn], ("preferences", f"- {TABLE} (said 2026-11-28)")))
    assert bundle.memories == []


def test_a_fragment_word_patterns_cut_from_a_message_goes_and_the_message_stays() -> None:
    """Word patterns cut a sentence out of a message with no reading of the rest: the message
    says it, with what was said around it."""
    turn = _item("mem_turn", "Thanks! That's why I want to pass that support on. It brings me joy.",
                 "msg_4", category="verbatim_turn", provider="native")  # fmt: skip
    cut = _item("mem_cut", "That's why I want to pass that support on.", "msg_4",
                category="preference", provider="native")  # fmt: skip
    bundle = said_once(_bundle([cut, turn]))
    assert [m.item_id for m in bundle.memories] == ["mem_turn"]
