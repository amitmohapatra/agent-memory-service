"""The memories' ranking: reciprocal-rank fusion of every arm, and four rules (ADR 0026).

A memory search reads eight ranked lists from the store in one round trip - BM25 and both
dense spaces over each of the two keys (the memory alone, and the memory read with the turn
it answers), and the late-interaction arm over their union, once per key. They are fused by
reciprocal rank, ``1 / (K + rank)``, every arm at weight one except the late-interaction
arms, ``LATE`` over the memory's own key and ``LATE_CONTEXT`` over its context key:
token-level matching is what separates the right turn from its paraphrases, and both corpora
it was checked on chose those weights on their own.

Then four rules, each a fact about conversations rather than about one benchmark:

* **session** - evidence comes in runs: every memory gains ``SESSION`` times the best fused
  score in its session (the day it was said on);
* **speaker** - a question that names a person is about that person's memories;
* **time** - a "when" question is answered by a memory that names a time;
* **period** - a question that names a period ("in June", "on 1 February, 2023", "last
  month") is answered by a memory said in it or about a day in it (``periods``).

Nothing here is fitted. ``K``, ``LATE``, ``LATE_CONTEXT`` and ``SESSION`` were chosen on
LongMemEval and scored on LoCoMo, and the other way round, and kept only where both
agreed; ``SPEAKER`` and ``TIME`` are round values neither corpus tuned (LongMemEval names
no speakers to test one on). ``PERIOD`` is the middle of the range LongMemEval is indifferent
to (0 to 5; it costs LongMemEval at 8), offline +0.9 on LoCoMo overall and +6.8 on its
questions that name a period. The learned ranking this replaces (ADR 0025) read 0.841 on
LoCoMo, the corpus it was fitted on, and 0.852 on LongMemEval, which it had not seen, where
plain fusion reads 0.868.

Pure functions over hits: the store and the models are the engine's business.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

from memory_service.modules.retrieval import periods
from memory_service.ports.search import SearchHit, VectorName

#: the reciprocal-rank constant: a rank-1 hit scores 1/11, a rank-10 one 1/20
K = 10
#: the late-interaction arm's weight over the memory's own key, and over its context key;
#: every other arm weighs one
LATE = 6.0
LATE_CONTEXT = 2.0
#: share of its session's best fused score every memory gains
SESSION = 0.3
#: the sessions' best scores are read from the fused top this-many
SESSION_DEPTH = 50
#: what the speaker and time rules add, in units of a rank-1 hit in one arm
SPEAKER = 1.0
TIME = 1.0
#: what a memory in the period the question names gains (``periods``), in the same units
PERIOD = 3.0

#: "when ...", "what year ...", "how long ...", "before"/"after": a question about time
WHEN = re.compile(
    r"^\s*(when|what (year|month|date|day|time)|how long|how many (days|weeks|months|years)"
    r"|since when)\b|\b(before|after)\b",
    re.IGNORECASE,
)
#: a memory that names a time
TIMED = re.compile(
    r"\b(yesterday|today|tonight|tomorrow|last (night|week|weekend|month|year|summer|winter"
    r"|spring|fall|time)|ago|recently|next (week|month|year)|this (week|weekend|month|year"
    r"|morning)|(19|20)\d\d|january|february|march|april|may|june|july|august|september"
    r"|october|november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|weeks?|months?|years?|days?)\b",
    re.IGNORECASE,
)


@dataclass
class ArmPool:
    """Everything the arms returned, by record: the hit and its rank in each arm."""

    hits: dict[str, SearchHit]
    #: arm -> record -> one-based rank in that arm
    ranks: dict[VectorName, dict[str, int]]

    @classmethod
    def of(cls, arms: Mapping[VectorName, Sequence[SearchHit]]) -> ArmPool:
        hits: dict[str, SearchHit] = {}
        ranks: dict[VectorName, dict[str, int]] = {}
        for name, listed in arms.items():
            ranks[name] = {}
            for rank, hit in enumerate(listed, start=1):
                hits.setdefault(hit.record_id, hit)
                ranks[name].setdefault(hit.record_id, rank)
        return cls(hits=hits, ranks=ranks)

    def speaker(self, rid: str) -> str:
        """Who the memory is about: its subject's name, lower-cased (``user:caroline``)."""
        return str(self.hits[rid].payload.get("subject") or "").split(":", 1)[-1].lower()

    def session(self, rid: str) -> str:
        """The day the memory was observed on: a conversation's sessions are its days."""
        return str(self.hits[rid].payload.get("observed_at") or "")[:10]


def fused(pool: ArmPool) -> dict[str, float]:
    """Weighted reciprocal-rank fusion of every arm."""
    out = dict.fromkeys(pool.hits, 0.0)
    weights = {VectorName.COLBERT: LATE, VectorName.COLBERT_CTX: LATE_CONTEXT}
    for name, ranks in pool.ranks.items():
        weight = weights.get(name, 1.0)
        for rid, rank in ranks.items():
            out[rid] += weight / (K + rank)
    return out


def ranked(pool: ArmPool, query: str, now: date | None = None) -> list[tuple[str, float]]:
    """Every record the arms found, best first, with its final score. ``now`` anchors a
    question's relative periods ("last month"); today when not given."""
    base = fused(pool)
    unit = 1.0 / (K + 1)
    best = _session_best(pool, base)
    named = _named(pool, query)
    when = bool(WHEN.search(query))
    named_periods = periods.query_periods(query, now or datetime.now(UTC).date())
    scores = {}
    for rid, score in base.items():
        payload = pool.hits[rid].payload
        speaks = bool(named) and pool.speaker(rid) in named
        timed = when and bool(TIMED.search(str(payload.get("text", ""))))
        dated = bool(named_periods) and periods.within(named_periods, periods.memory_days(payload))
        scores[rid] = (
            score
            + SESSION * best.get(pool.session(rid), 0.0)
            + (SPEAKER * unit if speaks else 0.0)
            + (TIME * unit if timed else 0.0)
            + (PERIOD * unit if dated else 0.0)
        )
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _session_best(pool: ArmPool, base: Mapping[str, float]) -> dict[str, float]:
    best: dict[str, float] = {}
    for rid in sorted(base, key=lambda r: (-base[r], r))[:SESSION_DEPTH]:
        best.setdefault(pool.session(rid), base[rid])
    return best


def _named(pool: ArmPool, query: str) -> set[str]:
    """The people the question names, among the subjects of everything the arms found."""
    lowered = query.lower()
    speakers = {pool.speaker(rid) for rid in pool.hits} - {""}
    return {s for s in speakers if re.search(rf"\b{re.escape(s)}\b", lowered)}
