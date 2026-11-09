"""The extraction eval's scorer and the Mem0 J ruler of the LoCoMo harness."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from benchmark.extraction import score_case, summarise
from benchmark.locomo import MEM0_J_PROMPT, _judge

pytestmark = pytest.mark.unit


def mem(content, *, category="fact", status="CURRENT", marked=False):
    return {
        "content": content,
        "category": category,
        "temporal_status": status,
        "marked": marked,
        "memory_type": "SEMANTIC",
        "confidence": 0.9,
    }


CASE = {
    "id": "c1",
    "kind": "update",
    "expect": [["PackagingCorp"], ["Tacoma"]],
    "forbid": [["Mumbai"]],
    "stale": [["supplier is Uline"]],
    "keep": [["Rex"]],
}


def test_a_held_fact_is_a_reading_not_the_turn_as_said():
    row = score_case(
        CASE,
        [
            mem("We moved to Tacoma.", category="verbatim_turn"),
            mem("Our supplier is PackagingCorp."),
            mem("The supplier is PackagingCorp now."),
            mem("My dog is Rex."),
        ],
        [True, False],
    )
    assert (row["held"], row["expected"], row["duplicated"]) == (1, 2, 1)
    assert (row["false"], row["stale_current"], row["keep_lost"]) == (0, 0, 0)
    assert (row["asks_hit"], row["asks"]) == (1, 2)


def test_a_closed_or_marked_value_is_not_current():
    row = score_case(
        CASE,
        [
            mem("Our supplier is Uline.", status="SUPERSEDED"),
            mem("The supplier is Uline.", marked=True),
            mem("Mumbai is the hub."),
        ],
        [],
    )
    assert row["stale_current"] == 0 and row["false"] == 1 and row["keep_lost"] == 1


def test_the_summary_rates():
    rows = [
        score_case(CASE, [mem("PackagingCorp"), mem("Our supplier is Uline.")], []),
        score_case({"id": "c2", "kind": "chitchat", "negative": True}, [mem("Hi there.")], []),
    ]
    out = summarise(rows)
    assert out["recall"] == 0.5
    assert out["false_fact_rate"] == 0.5  # no forbidden fact; the chit-chat stored one
    assert out["wrong_update_rate"] == 1.0  # the stale value current, the kept one gone


@pytest.mark.parametrize(
    ("reply", "correct"),
    [
        ('It names the same date. {"label": "CORRECT"}', True),
        ('Different city. {"label": "WRONG"}', False),
        ("no json at all", False),
    ],
)
async def test_the_mem0_ruler_reads_the_label_of_their_prompt(reply, correct):
    sent = []

    class LLM:
        async def complete(self, messages, **kwargs):
            sent.append(messages)
            return SimpleNamespace(text=reply)

    verdict = await _judge(LLM(), "When?", "7 May 2023", "On May 7th", ruler="mem0")
    assert verdict["correct"] is correct
    (message,) = sent[0]
    assert message.role == "user"
    assert message.content == MEM0_J_PROMPT.format(
        question="When?", gold_answer="7 May 2023", generated_answer="On May 7th"
    )


def test_each_public_qa_configuration_pins_its_ingestion_uses():
    from benchmark.public.memory_qa import config_uses

    assert config_uses("native") == ()
    assert config_uses("extraction") == ("contextual_extraction",)
    assert "contextual_extraction" in config_uses("bifrost")
    with pytest.raises(ValueError, match="unknown configuration"):
        config_uses("other")


def test_a_word_is_held_where_a_word_starts_not_inside_another():
    """coref-4 was counted a duplicate: "art" (the art corner) inside "started"
    (kindergarten). A stem still matches its forms ("gat" in "gata", "renew" in
    "renewal"), in any script."""
    from benchmark.extraction import _holds

    assert not _holds("Zoe started kindergarten.", ["Zoe", "art"])
    assert _holds("Zoe loves the art corner.", ["Zoe", "art"])
    assert _holds("Pimienta es una gata siamesa.", ["Pimienta", "gat"])
    assert _holds("The Halvorsen Foods renewal is for three years.", ["Halvorsen", "renew"])
    assert _holds("hi-update अब बेंगलुरु में रहता है।", ["बेंगलुरु"])
