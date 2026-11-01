"""The referring turn's rule (``modules.memory.referred``), without a database."""

from __future__ import annotations

from datetime import UTC, datetime

from memory_service.domain.enums import Lifetime, MemoryType, ObservationKind
from memory_service.domain.ids import content_hash
from memory_service.domain.observation import Observation, ProcessingHints
from memory_service.modules.memory.referred import MAX_CHARS, headline, referred_turn
from memory_service.ports.intelligence import MemoryCandidate


def _said(text: str, role: str) -> Observation:
    return Observation(
        tenant_id="acme",
        user_id="ada",
        principal_id="user:ada",
        kind=ObservationKind.MESSAGE,
        content=text,
        content_hash=content_hash(text),
        hints=ProcessingHints(),
        occurred_at=datetime(2026, 10, 30, tzinfo=UTC),
        custom_metadata={"role": role},
    )


def _verbatim(text: str) -> MemoryCandidate:
    return MemoryCandidate(
        content=text,
        memory_type=MemoryType.OBSERVATION,
        lifetime=Lifetime.LONG_TERM,
        importance=0.25,
        confidence=0.99,
        category="verbatim_turn",
    )


ANSWER = _said("\n## \n**Reorder qty, north: 19 units**\n- inputs", "ASSISTANT")


def test_the_headline_is_the_first_line_with_text_without_markup() -> None:
    assert headline(ANSWER.content) == "Reorder qty, north: 19 units"
    assert headline("   \n**  **\n") == ""


def test_only_a_short_referring_turn_right_after_the_agents_answer() -> None:
    turn = _said("Yes, go with that.", "USER")
    [kept] = referred_turn(turn, [_verbatim(turn.content)], ANSWER)
    assert kept.content == "Yes, go with that. (in reply to: Reorder qty, north: 19 units)"
    assert kept.category == "referred_turn" and kept.importance == 0.5
    long = _said("Approve it " + "x" * MAX_CHARS, "USER")
    assert referred_turn(long, [_verbatim(long.content)], ANSWER) == []
    after_user = _said("I said 19 units.", "USER")
    assert referred_turn(turn, [_verbatim(turn.content)], after_user) == []
    assert referred_turn(turn, [], ANSWER) == []  # nothing kept as said: nothing beside it
    assert referred_turn(turn, [_verbatim(turn.content)], _said("**  **", "ASSISTANT")) == []
