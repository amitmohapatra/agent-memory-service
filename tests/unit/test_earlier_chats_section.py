"""The earlier conversations' lines in a context (ADR 0044): how they read, what they cost, and
the full form's shape."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memory_service.api.schemas.context import ContextResponse
from memory_service.domain.context_bundle import EarlierChatLine, earlier_chats_section
from memory_service.domain.enums import MessageRole
from memory_service.modules.context.builder import _within
from memory_service.modules.retrieval.search import _ends, _passage, _weights

pytestmark = pytest.mark.unit


def _line(text: str, day: int, *, corrected: bool = False) -> EarlierChatLine:
    return EarlierChatLine(
        message_id=f"msg_{day}",
        role=MessageRole.ASSISTANT,
        text=text,
        thread_id=f"thr_{day}",
        said_at=datetime(2026, 10, day, 10, tzinfo=UTC),
        corrected=corrected,
    )


def test_each_line_is_dated_labelled_and_marked_when_corrected_oldest_first() -> None:
    [section] = earlier_chats_section(
        [_line("Reorder 137\nunits.", 27, corrected=True), _line("West: 182,340.", 26)]
    )
    heading, note, first, second = section.split("\n")
    assert heading == "## Earlier conversations" and note.endswith("that day's data):")
    assert first == "- 2026-10-26 Monday, conversation thr_26, the assistant said: West: 182,340."
    assert second.startswith("- 2026-10-27 Tuesday, conversation thr_27, the assistant said: ")
    assert "Reorder 137 units." in second, "one line each"
    assert second.endswith("(corrected by the user's next message: use the correction)")


def test_no_line_costs_nothing_and_lines_keep_to_half_the_budget() -> None:
    assert earlier_chats_section([]) == []
    assert _within([], 4000) == ([], 0), "a question with no earlier answer pays no token"
    long = [_line("x " * 450, day) for day in (20, 21, 22)]
    kept, tokens = _within(long, 1000)
    assert 0 < len(kept) < 3 and 0 < tokens <= 500


def test_a_long_answer_keeps_its_start_and_its_end() -> None:
    text = "Start of the working. " + "middle " * 400 + "Result: 137 units."
    clipped = _ends(text, 200)
    assert len(clipped) <= 200 and clipped.startswith("Start") and clipped.endswith("137 units.")
    assert _ends("short", 200) == "short"


def test_a_word_every_message_shares_weighs_least() -> None:
    weight = _weights({"the", "merino"}, [{"the", "merino"}, {"the"}, {"the", "scarf"}])
    assert weight["merino"] > weight["the"] > 0


def test_the_full_form_carries_them() -> None:
    body = {
        "bundle_id": "b",
        "evidence_status": "COMPLETE",
        "token_estimate": 1,
        "earlier_chats": [
            {
                "id": "msg_1",
                "role": "ASSISTANT",
                "text": "137 units",
                "thread_id": "thr_1",
                "said_at": "2026-10-26T10:00:00+00:00",
                "corrected": True,
            }
        ],
    }
    assert ContextResponse.model_validate(body).earlier_chats[0].corrected is True


def test_an_answer_comes_with_the_steps_that_made_it() -> None:
    """An answer whose turn made calls carries its steps list under it: which tools, with
    what, and each result's message, dated by its line."""
    steps = 'For "Top supplier?" (1 call):\n1. sales_summary(group_by="supplier"): result msg_9'
    line = _line("SUP-24 leads.", 26).model_copy(update={"steps": steps})
    [section] = earlier_chats_section([line, _line("West: 182,340.", 27)])
    shown = section.split("\n")[2:]
    assert shown == [
        "- 2026-10-26 Monday, conversation thr_26, the assistant said: SUP-24 leads.",
        '  Its steps: For "Top supplier?" (1 call):',
        '  1. sales_summary(group_by="supplier"): result msg_9',
        "- 2026-10-27 Tuesday, conversation thr_27, the assistant said: West: 182,340.",
    ]


def test_a_long_answer_keeps_the_sentences_the_question_falls_in() -> None:
    """Whole sentences, the ones holding the question's heaviest words first, in the answer's
    own order, within the cap, with what was left out marked."""
    text = (
        "Here is the review. "
        + "The Cotton Tee needs no change. " * 30
        + "The Merino Hoodie reorder is 137 units. "
        + "The Linen Shirt can wait. " * 30
        + "Ask for the south too."
    )
    weight = {"merino": 2.0, "hoodie": 2.0, "reorder": 1.0, "review": 0.2}
    passage = _passage(text, weight, 200)
    assert len(passage) <= 200
    assert "The Merino Hoodie reorder is 137 units." in passage
    assert passage.index("Here is the review.") < passage.index("The Merino Hoodie")
    assert " … " in passage and passage.count("The Cotton Tee") < 30
    assert _passage("short", weight, 200) == "short"
    assert _passage("one sentence with no stop " * 40, weight, 200) == _ends(
        "one sentence with no stop " * 40, 200
    )
