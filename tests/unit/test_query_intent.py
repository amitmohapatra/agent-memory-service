"""Lever 3 of ADR 0041: a question about now reads a slot's latest value, a question about
what was keeps the earlier values where they rank."""

from __future__ import annotations

import pytest

from memory_service.modules.retrieval.engine import Candidate, current_first, latest_per_slot
from memory_service.modules.retrieval.router import QueryRouter

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "question",
    [
        "Who was our pallet supplier before we switched?",
        "What did I use to drive?",
        "Wer war früher unser Lieferant?",
        "¿Dónde vivía antes?",
        "من كان المورد سابقا؟",
        "हमारा पुराना सप्लायर कौन था?",
    ],
)
def test_a_question_about_what_was_is_a_history_question(question: str) -> None:
    assert QueryRouter().route(question).signals["history"] is True


@pytest.mark.parametrize(
    "question",
    [
        "Who is our pallet supplier?",
        "Wo wohne ich?",
        "¿Cuál es el precio?",
        "What did Caroline paint last week?",
        "Which tool is used to log tickets?",
        "What do we use to track returns?",
    ],
)
def test_a_question_about_now_is_not(question: str) -> None:
    assert QueryRouter().route(question).signals["history"] is False


def _fact(rid: str, *, subject: str, obj: str, day: str, owner: str = "user:u1", **extra):
    return Candidate(
        record_id=rid,
        kind="memory",
        text=f"{subject} is {obj}",
        score=1.0,
        retrievers=[],
        payload={
            "subject": subject,
            "predicate": "is",
            "object": obj,
            "owner_principal": owner,
            "observed_at": f"{day}T09:00:00+00:00",
            **extra,
        },
    )


@pytest.mark.parametrize(
    "question",
    [
        "What was that reorder number you worked out on Monday for the Merino Hoodie?",
        "Last time you worked out a north reorder for the Merino Hoodie. Redo it.",
        "A couple of weeks back I gave you the master lock code. What was it again?",
        "Given the region changes we talked about, does Store 58 get the cereal BOGO?",
        "In my last two chats I asked about August revenue for two regions. Which were they?",
        "Vor ein paar Wochen habe ich dir den Generalcode gegeben. Wie lautete er?",
        "Mencionaste hace un tiempo un cambio de depósito; ¿a dónde van ahora?",
        "الشهر الماضي أخبرتك برقم حسابنا. هل تذكّرني؟",
        "पिछले महीने मैंने तुम्हें हमारा अकाउंट नंबर बताया था। याद दिलाओगे?",
    ],
)
def test_a_question_about_an_earlier_chat_points_to_one(question: str) -> None:
    """ADR 0044: what was said or worked out in an earlier chat, in each language's words."""
    assert QueryRouter().route(question).signals["earlier_chat"] is True


@pytest.mark.parametrize(
    "question",
    [
        "What were total sales last week?",
        "What is the reorder quantity for the Merino Hoodie in the north?",
        "Who was our pallet supplier before we switched?",
        "Wie hoch war der Umsatz im letzten Monat?",
    ],
)
def test_a_question_about_the_past_of_the_world_does_not(question: str) -> None:
    assert QueryRouter().route(question).signals["earlier_chat"] is False


def test_the_newest_value_of_a_slot_wins_and_the_older_becomes_an_earlier_value() -> None:
    old = _fact("old", subject="our pallet supplier", obj="Uline", day="2026-10-01")
    new = _fact("new", subject="our pallet supplier", obj="PackagingCorp", day="2026-10-09")
    got = current_first(latest_per_slot([old, new]))
    assert [c.record_id for c in got] == ["new", "old"]
    assert got[1].payload["earlier_value"] and got[1].payload["replaced_by"] == "new"
    assert "earlier_value" not in old.payload  # the hit itself is not changed


@pytest.mark.parametrize(
    "other",
    [
        # the same value said again
        _fact("b", subject="our pallet supplier", obj="uline", day="2026-10-09"),
        # another person's statement of the slot
        _fact("b", subject="our pallet supplier", obj="Acme", day="2026-10-09", owner="user:u2"),
        # a pronoun is no slot
        _fact("b", subject="it", obj="Acme", day="2026-10-09"),
        # what a turn's rule writes is no slot
        Candidate(
            record_id="b",
            kind="memory",
            text="x",
            score=1.0,
            retrievers=[],
            payload={
                "subject": "our pallet supplier",
                "predicate": "said",
                "object": "Acme",
                "owner_principal": "user:u1",
                "observed_at": "2026-10-09T09:00:00+00:00",
            },
        ),
    ],
)
def test_no_slot_no_change(other: Candidate) -> None:
    old = _fact("a", subject="our pallet supplier", obj="Uline", day="2026-10-01")
    got = latest_per_slot([old, other])
    assert not any(c.payload.get("earlier_value") for c in got)
