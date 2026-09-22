"""Admission gate: an explicit, stored decision per memory candidate.

Extraction already drops what no rule recognises; the gate is the second, measurable step
that decides whether a recognised candidate is *worth keeping*: worthiness (type prior
blended with extraction confidence, penalised for transient phrasing), novelty (what the
consolidator decided), confidence and expected utility (lifetime x importance x recency).
The verdict and every input are stored on the memory (``system_metadata.admission``) so the
memory gate can report admission precision/recall against a labelled set. Deferred
candidates are parked in working memory until they are seen again.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from memory_service.config.constants import MemoryIntelligenceSettings
from memory_service.domain.enums import AdmissionVerdict, DedupDecision, Lifetime, MemoryType
from memory_service.domain.memory import AdmissionDecision
from memory_service.ports.intelligence import ConsolidationOutcome, MemoryCandidate

_TYPE_PRIOR: dict[MemoryType, float] = {
    MemoryType.USER: 0.9,
    MemoryType.PREFERENCE: 0.9,
    MemoryType.SEMANTIC: 0.85,
    MemoryType.DECISION: 0.9,
    MemoryType.PROCEDURAL: 0.8,
    MemoryType.POLICY: 0.85,
    MemoryType.SHARED: 0.75,
    MemoryType.BELIEF: 0.8,
    MemoryType.ENTITY_SUMMARY: 0.8,
    MemoryType.OBSERVATION: 0.7,
    MemoryType.KNOWLEDGE_RAG: 0.7,
    MemoryType.SKILL: 0.75,
    MemoryType.TASK: 0.6,
    MemoryType.WORK: 0.6,
    MemoryType.EPISODIC: 0.55,
    MemoryType.OUTCOME: 0.6,
    MemoryType.FAILURE: 0.65,
    MemoryType.ARTIFACT: 0.5,
    MemoryType.AGENT: 0.45,
    MemoryType.TOOL: 0.35,
    MemoryType.CONVERSATION: 0.3,
    MemoryType.WORKING: 0.2,
    MemoryType.SUMMARY: 0.6,
    MemoryType.DERIVED: 0.6,
    MemoryType.CUSTOM: 0.6,
}
_NOVELTY: dict[DedupDecision, float] = {
    DedupDecision.CREATE: 1.0,
    DedupDecision.SUPERSEDE: 0.9,
    DedupDecision.UPDATE: 0.85,
    DedupDecision.CONTRADICT: 0.9,
    DedupDecision.MERGE: 0.5,
    DedupDecision.REINFORCE: 0.25,
    DedupDecision.IGNORE: 0.0,
}
_LIFETIME_WEIGHT: dict[Lifetime, float] = {
    Lifetime.LONG_TERM: 1.0,
    Lifetime.SHORT_TERM: 0.65,
    Lifetime.ARCHIVAL: 0.4,
    Lifetime.EPHEMERAL: 0.2,
}
_TRANSIENT = re.compile(
    r"\b(right now|at the moment|for now|for the moment|today only|just now|one sec|"
    r"one second|a moment|brb|be right back|hold on|hang on|running late|on my way|"
    r"this (?:morning|afternoon|evening)|tonight|later today|in a minute|in a bit|"
    r"currently (?:in|at|on) a (?:call|meeting)|off to lunch|back in)\b",
    re.IGNORECASE,
)
_GENERIC = re.compile(
    r"^(?:it|this|that|there|here|everything|nothing|something)\b.{0,40}$", re.IGNORECASE
)
_RECENCY_FLOOR = 0.5
_RECENCY_YEAR_DAYS = 365.0


def worthiness_of(candidate: MemoryCandidate, *, hinted: bool = False) -> tuple[float, list[str]]:
    """Type prior blended with extraction confidence; transient or generic phrasing halves it.
    An explicit application hint (memory_type / importance) is a strong signal on its own."""
    reasons: list[str] = []
    prior = _TYPE_PRIOR.get(candidate.memory_type, 0.5)
    value = 0.6 * prior + 0.4 * candidate.confidence
    if hinted:
        value = max(value, 0.8)
        reasons.append("explicit hint")
    text = candidate.content
    dated = candidate.valid_from is not None or candidate.valid_to is not None
    # A verbatim turn is kept *because* it is the raw wording; "I'm off to lunch, but the
    # Berlin office moved in May" is a memory because of the second clause, and halving it
    # for the first dropped the only record of the turn.
    if (
        _TRANSIENT.search(text)
        and not dated
        and candidate.category not in ("decision", "verbatim_turn")
    ):
        value *= 0.4
        reasons.append("transient phrasing")
    if _GENERIC.match(text.strip()) and not candidate.subject:
        value *= 0.5
        reasons.append("generic statement")
    return round(min(1.0, value), 4), reasons


def expected_utility(candidate: MemoryCandidate, *, now: datetime) -> float:
    """lifetime weight x importance x recency (linear decay to half over a year)."""
    observed = next((e.observed_at for e in candidate.evidence if e.observed_at), None)
    recency = 1.0
    if observed is not None:
        age_days = max(0.0, (now - observed).total_seconds() / 86_400)
        recency = max(_RECENCY_FLOOR, 1.0 - (1.0 - _RECENCY_FLOOR) * age_days / _RECENCY_YEAR_DAYS)
    weight = _LIFETIME_WEIGHT.get(candidate.lifetime, 0.5)
    return round(min(1.0, weight * max(candidate.importance, 0.05) * recency), 4)


class AdmissionGate:
    def __init__(self, settings: MemoryIntelligenceSettings) -> None:
        self.cfg = settings

    def evaluate(
        self,
        candidate: MemoryCandidate,
        outcome: ConsolidationOutcome | None = None,
        *,
        hinted: bool = False,
        now: datetime | None = None,
    ) -> AdmissionDecision:
        now = now or datetime.now(UTC)
        worthiness, reasons = worthiness_of(candidate, hinted=hinted)
        decision = outcome.decision if outcome is not None else DedupDecision.CREATE
        novelty = _NOVELTY.get(decision, 0.5)
        confidence = round(candidate.confidence, 4)
        utility = expected_utility(candidate, now=now)
        score = round(0.4 * worthiness + 0.2 * novelty + 0.2 * confidence + 0.2 * utility, 4)
        verdict = AdmissionVerdict.ADMIT
        strengthens = decision in (DedupDecision.REINFORCE, DedupDecision.MERGE)
        if worthiness < self.cfg.admission_worthiness_min:
            verdict = AdmissionVerdict.REJECT
            reasons.append(f"worthiness {worthiness:.2f} < {self.cfg.admission_worthiness_min}")
        elif confidence < self.cfg.admission_confidence_min:
            verdict = AdmissionVerdict.REJECT
            reasons.append(f"confidence {confidence:.2f} < {self.cfg.admission_confidence_min}")
        elif strengthens:
            # a repeat of something already kept never needs its own novelty: it strengthens
            reasons.append(f"strengthens existing ({decision.value.lower()})")
        elif score < self.cfg.admission_score_min - self.cfg.admission_defer_band:
            verdict = AdmissionVerdict.REJECT
            reasons.append(f"score {score:.2f} below admission band")
        elif score < self.cfg.admission_score_min:
            verdict = AdmissionVerdict.DEFER
            reasons.append(
                f"score {score:.2f} < {self.cfg.admission_score_min}: needs corroboration"
            )
        else:
            reasons.append(f"score {score:.2f} >= {self.cfg.admission_score_min}")
        return AdmissionDecision(
            verdict=verdict,
            worthiness=worthiness,
            novelty=novelty,
            confidence=confidence,
            expected_utility=utility,
            score=min(1.0, score),
            reasons=reasons,
            decided_at=now,
        )
