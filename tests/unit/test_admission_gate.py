"""The admission gate's verdicts: which candidate is admitted, deferred or rejected, and the
inputs (worthiness, novelty, confidence, expected utility) stored with the decision."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from memory_service.config.constants import MEMORY_INTELLIGENCE, MemoryIntelligenceSettings
from memory_service.domain.enums import AdmissionVerdict, DedupDecision, Lifetime, MemoryType
from memory_service.domain.evidence import EvidenceRef
from memory_service.modules.memory.admission import (
    AdmissionGate,
    expected_utility,
    worthiness_of,
)
from memory_service.ports.intelligence import ConsolidationOutcome, MemoryCandidate

pytestmark = pytest.mark.unit

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
GATE = AdmissionGate(MEMORY_INTELLIGENCE)


def _candidate(**overrides: Any) -> MemoryCandidate:
    base: dict[str, Any] = {
        "content": "The Berlin office moved to Hamburg.",
        "memory_type": MemoryType.USER,
        "lifetime": Lifetime.LONG_TERM,
        "subject": "user:melanie",
        "confidence": 0.9,
        "importance": 0.8,
    }
    return MemoryCandidate(**{**base, **overrides})


def _outcome(candidate: MemoryCandidate, decision: DedupDecision) -> ConsolidationOutcome:
    return ConsolidationOutcome(decision=decision, candidate=candidate)


def _seen(at: datetime) -> EvidenceRef:
    return EvidenceRef(source_type="message", source_id="m1", observed_at=at)


# --- worthiness ------------------------------------------------------------------------


def test_worthiness_blends_the_type_prior_with_extraction_confidence() -> None:
    value, reasons = worthiness_of(_candidate(memory_type=MemoryType.USER, confidence=0.5))
    assert value == pytest.approx(0.6 * 0.9 + 0.4 * 0.5)
    assert reasons == []


def test_an_explicit_hint_lifts_worthiness_to_at_least_point_eight() -> None:
    weak = _candidate(memory_type=MemoryType.CONVERSATION, confidence=0.1)
    plain, _ = worthiness_of(weak)
    hinted, reasons = worthiness_of(weak, hinted=True)
    assert plain < 0.8
    assert hinted == 0.8
    assert "explicit hint" in reasons


def test_an_explicit_hint_never_lowers_a_worthiness_already_above_it() -> None:
    strong = _candidate(memory_type=MemoryType.USER, confidence=1.0)
    assert worthiness_of(strong, hinted=True)[0] == worthiness_of(strong)[0] == 0.94


def test_transient_phrasing_is_not_penalised_when_the_candidate_is_dated() -> None:
    dated = _candidate(content="I am on my way to the Berlin office.", valid_from=NOW)
    undated = _candidate(content="I am on my way to the Berlin office.")
    assert "transient phrasing" not in worthiness_of(dated)[1]
    assert "transient phrasing" in worthiness_of(undated)[1]
    assert worthiness_of(undated)[0] == pytest.approx(worthiness_of(dated)[0] * 0.4)


def test_a_candidate_with_only_an_end_date_counts_as_dated() -> None:
    _, reasons = worthiness_of(_candidate(content="Hold on, the office is shut.", valid_to=NOW))
    assert "transient phrasing" not in reasons


def test_a_generic_statement_without_a_subject_is_halved() -> None:
    generic, reasons = worthiness_of(_candidate(content="It is fine.", subject=None))
    specific, _ = worthiness_of(_candidate(content="The office is fine.", subject=None))
    assert "generic statement" in reasons
    assert generic == pytest.approx(specific * 0.5)


def test_a_generic_statement_about_a_named_subject_keeps_its_worthiness() -> None:
    _, reasons = worthiness_of(_candidate(content="It is fine.", subject="user:melanie"))
    assert "generic statement" not in reasons


def test_transient_and_generic_penalties_compound() -> None:
    value, reasons = worthiness_of(_candidate(content="That is it for now.", subject=None))
    assert reasons == ["transient phrasing", "generic statement"]
    assert value == pytest.approx((0.6 * 0.9 + 0.4 * 0.9) * 0.4 * 0.5)


# --- expected utility ------------------------------------------------------------------


def test_expected_utility_is_lifetime_times_importance_without_evidence() -> None:
    assert expected_utility(_candidate(importance=0.8), now=NOW) == 0.8
    assert expected_utility(_candidate(lifetime=Lifetime.SHORT_TERM, importance=1.0), now=NOW) == (
        0.65
    )
    assert expected_utility(_candidate(lifetime=Lifetime.ARCHIVAL, importance=1.0), now=NOW) == 0.4
    assert expected_utility(_candidate(lifetime=Lifetime.EPHEMERAL, importance=1.0), now=NOW) == 0.2


def test_importance_has_a_floor_so_utility_is_never_zero() -> None:
    assert expected_utility(_candidate(importance=0.0), now=NOW) == 0.05


def test_utility_decays_linearly_to_half_over_a_year_of_age() -> None:
    fresh = _candidate(importance=1.0, evidence=[_seen(NOW)])
    half_year = _candidate(importance=1.0, evidence=[_seen(NOW - timedelta(days=182.5))])
    a_year = _candidate(importance=1.0, evidence=[_seen(NOW - timedelta(days=365))])
    assert expected_utility(fresh, now=NOW) == 1.0
    assert expected_utility(half_year, now=NOW) == 0.75
    assert expected_utility(a_year, now=NOW) == 0.5


def test_utility_never_decays_below_the_recency_floor() -> None:
    ancient = _candidate(importance=1.0, evidence=[_seen(NOW - timedelta(days=3650))])
    assert expected_utility(ancient, now=NOW) == 0.5


def test_evidence_dated_in_the_future_counts_as_fresh() -> None:
    ahead = _candidate(importance=1.0, evidence=[_seen(NOW + timedelta(days=30))])
    assert expected_utility(ahead, now=NOW) == 1.0


def test_recency_is_read_from_the_first_evidence() -> None:
    old_first = _candidate(importance=1.0, evidence=[_seen(NOW - timedelta(days=365)), _seen(NOW)])
    assert expected_utility(old_first, now=NOW) == 0.5


# --- verdicts --------------------------------------------------------------------------


def test_a_worthy_novel_confident_candidate_is_admitted_with_every_input_recorded() -> None:
    decision = GATE.evaluate(_candidate(), now=NOW)
    assert decision.verdict is AdmissionVerdict.ADMIT
    assert decision.worthiness == 0.9
    assert decision.novelty == 1.0  # no consolidation outcome is a CREATE
    assert decision.confidence == 0.9
    assert decision.expected_utility == 0.8
    assert decision.score == 0.9
    assert decision.reasons == ["score 0.90 >= 0.4"]
    assert decision.decided_at == NOW


def test_the_decision_is_timestamped_now_when_no_clock_is_given() -> None:
    before = datetime.now(UTC)
    decision = GATE.evaluate(_candidate())
    assert before <= decision.decided_at <= datetime.now(UTC)


def test_an_unworthy_candidate_is_rejected_before_anything_else_is_weighed() -> None:
    decision = GATE.evaluate(
        _candidate(memory_type=MemoryType.CONVERSATION, confidence=0.3), now=NOW
    )
    assert decision.verdict is AdmissionVerdict.REJECT
    assert decision.worthiness == 0.3
    assert decision.reasons == ["worthiness 0.30 < 0.35"]


def test_a_hinted_but_unconfident_candidate_is_rejected_for_its_confidence() -> None:
    decision = GATE.evaluate(_candidate(confidence=0.2), hinted=True, now=NOW)
    assert decision.verdict is AdmissionVerdict.REJECT
    assert decision.reasons == ["explicit hint", "confidence 0.20 < 0.3"]


def test_a_low_scoring_candidate_inside_the_band_is_deferred_for_corroboration() -> None:
    weak = _candidate(memory_type=MemoryType.AGENT, confidence=0.35, importance=0.5)
    decision = GATE.evaluate(weak, _outcome(weak, DedupDecision.IGNORE), now=NOW)
    assert decision.verdict is AdmissionVerdict.DEFER
    assert decision.novelty == 0.0
    assert decision.score == 0.334
    assert decision.reasons == ["score 0.33 < 0.4: needs corroboration"]


def test_a_candidate_scoring_below_the_band_is_rejected() -> None:
    weak = _candidate(
        memory_type=MemoryType.TOOL, confidence=0.5, lifetime=Lifetime.EPHEMERAL, importance=0.05
    )
    decision = GATE.evaluate(weak, _outcome(weak, DedupDecision.IGNORE), now=NOW)
    assert decision.verdict is AdmissionVerdict.REJECT
    assert decision.score == 0.266
    assert decision.reasons == ["score 0.27 below admission band"]


@pytest.mark.parametrize(
    ("decision", "novelty"), [(DedupDecision.REINFORCE, 0.25), (DedupDecision.MERGE, 0.5)]
)
def test_a_repeat_of_something_kept_is_admitted_to_strengthen_it_whatever_its_score(
    decision: DedupDecision, novelty: float
) -> None:
    weak = _candidate(
        memory_type=MemoryType.TOOL, confidence=0.5, lifetime=Lifetime.EPHEMERAL, importance=0.05
    )
    verdict = GATE.evaluate(weak, _outcome(weak, decision), now=NOW)
    assert verdict.verdict is AdmissionVerdict.ADMIT
    assert verdict.novelty == novelty
    assert verdict.score < MEMORY_INTELLIGENCE.admission_score_min
    assert verdict.reasons == [f"strengthens existing ({decision.value.lower()})"]


def test_an_unworthy_repeat_is_still_rejected() -> None:
    weak = _candidate(memory_type=MemoryType.CONVERSATION, confidence=0.1)
    decision = GATE.evaluate(weak, _outcome(weak, DedupDecision.REINFORCE), now=NOW)
    assert decision.verdict is AdmissionVerdict.REJECT


@pytest.mark.parametrize(
    ("decision", "novelty"),
    [
        (DedupDecision.CREATE, 1.0),
        (DedupDecision.SUPERSEDE, 0.9),
        (DedupDecision.UPDATE, 0.85),
        (DedupDecision.CONTRADICT, 0.9),
        (DedupDecision.IGNORE, 0.0),
    ],
)
def test_novelty_follows_the_consolidators_decision(
    decision: DedupDecision, novelty: float
) -> None:
    candidate = _candidate()
    assert GATE.evaluate(candidate, _outcome(candidate, decision), now=NOW).novelty == novelty


def test_the_thresholds_are_the_gates_settings() -> None:
    strict = AdmissionGate(MemoryIntelligenceSettings(admission_score_min=0.99))
    decision = strict.evaluate(_candidate(), now=NOW)
    assert decision.verdict is AdmissionVerdict.REJECT
    assert decision.reasons == ["score 0.90 below admission band"]
    lenient = AdmissionGate(MemoryIntelligenceSettings(admission_worthiness_min=0.0))
    chatter = _candidate(memory_type=MemoryType.CONVERSATION, confidence=0.3)
    assert GATE.evaluate(chatter, now=NOW).verdict is AdmissionVerdict.REJECT
    assert lenient.evaluate(chatter, now=NOW).verdict is AdmissionVerdict.ADMIT
