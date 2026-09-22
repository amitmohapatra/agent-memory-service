"""Worthiness: the transient-phrasing penalty and who is exempt from it."""

from __future__ import annotations

from memory_service.domain.enums import Lifetime, MemoryType
from memory_service.modules.memory.admission import worthiness_of
from memory_service.ports.intelligence import MemoryCandidate

TURN = "Off to lunch now, but the Berlin office moved to a four-day week in May."


def _candidate(**overrides) -> MemoryCandidate:
    base = {
        "content": TURN,
        "memory_type": MemoryType.OBSERVATION,
        "lifetime": Lifetime.LONG_TERM,
        "subject": "user:melanie",
        "confidence": 0.6,
    }
    return MemoryCandidate(**{**base, **overrides})


def test_transient_phrasing_halves_an_extracted_fact() -> None:
    value, reasons = worthiness_of(_candidate())
    assert "transient phrasing" in reasons
    assert value < 0.5


def test_a_verbatim_turn_is_kept_for_its_wording_and_not_penalised_for_it() -> None:
    """The raw turn is stored *because* it is the raw wording. Halving it for "off to lunch"
    dropped the only record of the turn that also said where the office moved."""
    value, reasons = worthiness_of(_candidate(category="verbatim_turn"))
    assert "transient phrasing" not in reasons
    assert value == worthiness_of(_candidate(content="The Berlin office moved in May."))[0]


def test_a_decision_is_exempt_too() -> None:
    _, reasons = worthiness_of(_candidate(category="decision"))
    assert "transient phrasing" not in reasons
