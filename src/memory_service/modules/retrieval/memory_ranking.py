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
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

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
#: the same question in the other pack languages (``periods``): "Wann ...", "¿Cuándo ...",
#: "متى ...", "... कब ..."
WHEN_FOREIGN = re.compile(r"(?<!\w)(wann|cuándo|متى|कब)(?!\w)", re.IGNORECASE)
#: a memory that names a time
TIMED = re.compile(
    r"\b(yesterday|today|tonight|tomorrow|last (night|week|weekend|month|year|summer|winter"
    r"|spring|fall|time)|ago|recently|next (week|month|year)|this (week|weekend|month|year"
    r"|morning)|(19|20)\d\d|january|february|march|april|may|june|july|august|september"
    r"|october|november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|weeks?|months?|years?|days?)\b",
    re.IGNORECASE,
)
#: the same in the other pack languages: yesterday, today, last week/month/year, a month name
TIMED_FOREIGN = re.compile(
    r"(?<!\w)(gestern|heute|morgen|letzte[nrs]?|vor\s+\d+|ayer|hoy|mañana|pasad[oa]|hace"
    r"|أمس|اليوم|غدا|الماضي|الماضية|منذ|कल|आज|पिछले|पहले"
    r"|(?:19|20)\d\d|"
    + "|".join(sorted({m for months in periods.FOREIGN_MONTHS.values() for m in months}))
    + r")(?!\w)",
    re.IGNORECASE,
)


@dataclass
class ArmPool:
    """Everything the arms returned, by record: the hit and its rank in each arm."""

    hits: dict[str, SearchHit]
    #: arm -> record -> one-based rank in that arm
    ranks: dict[VectorName, dict[str, int]]
    #: record -> every hit of the statement it stands for, itself first (``statements``);
    #: a record that is its statement's only form is absent
    forms: dict[str, list[SearchHit]] = field(default_factory=dict)

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
        return _subject_name(self.hits[rid])

    def speakers(self, rid: str) -> set[str]:
        """Who the statement is about, in any of its forms: the turn names its speaker, a
        reading of it the thing it is about."""
        return {_subject_name(hit) for hit in self.forms.get(rid) or (self.hits[rid],)}

    def payloads(self, rid: str) -> list[dict]:
        """The payload of every form of the statement ``rid`` stands for."""
        return [hit.payload for hit in self.forms.get(rid) or (self.hits[rid],)]

    def session(self, rid: str) -> str:
        """The day the memory was observed on: a conversation's sessions are its days."""
        return str(self.hits[rid].payload.get("observed_at") or "")[:10]


def _subject_name(hit: SearchHit) -> str:
    return str(hit.payload.get("subject") or "").split(":", 1)[-1].lower()


#: One statement, one document. A turn is kept verbatim beside what the rules read from it
#: (a fact, an event, a preference: ADR 0036), and each was a candidate of its own: two of the
#: ten places for one thing said, the reading's words ranked against the turn's. LongMemEval's
#: authors measured exactly this choice (Wu et al., ICLR 2025, Table 10): the extracted facts
#: merged into the key of the turn they came from raised Recall@5 to 0.644, ranked as separate
#: entries and fused it fell to 0.478, below the 0.582 of the turns alone. Here every form of
#: one statement - the same speaker's words of the same instant from the same source turns,
#: current or earlier value alike - is one document in every arm: its rank in an arm is its
#: best form's, the rules read all of its forms, and the form shown is the turn itself when it
#: is among them (it holds every reading whole), else the one that ranked best on its own
#: (``statements``). The other forms ride along as its ``duplicates``.
GROUP_STATEMENTS = True
#: the category of a turn kept verbatim (ADR 0036)
TURN = "verbatim_turn"


def _speaker_key(principal: object) -> object:
    """The person a principal speaks for: an agent bound to a user relays that user's words
    (as ``engine._speaker``)."""
    if isinstance(principal, str) and principal.startswith("agent:") and "/" in principal:
        return "user:" + principal.removeprefix("agent:").split("/", 1)[0]
    return principal


def statement_key(payload: Mapping[str, Any]) -> tuple | None:
    """What makes two memories forms of one statement: the speaker, the instant, the source
    turns (observations only: a memory derived from other memories states something of its
    own), and whether it is a current or an earlier value. None for a memory that cannot
    show all of it."""
    if payload.get("derived"):
        return None
    refs = payload.get("source_refs") or []
    sources = tuple(
        sorted(
            (str(ref.get("source_type")), str(ref.get("source_id")))
            for ref in refs
            if isinstance(ref, Mapping) and ref.get("source_type") and ref.get("source_id")
        )
    )
    owner, observed = payload.get("owner_principal"), payload.get("observed_at")
    if (
        not sources
        or len(sources) != len(refs)
        or any(kind == "memory" for kind, _ in sources)
        or not owner
        or not observed
    ):
        return None
    return (_speaker_key(owner), str(observed), sources, bool(payload.get("earlier_value")))


def statements(pool: ArmPool) -> ArmPool:
    """``pool`` with every statement one record (``GROUP_STATEMENTS``): the turn as said, or
    else the form that fused best on its own, stands for it, at its best form's rank in every
    arm, the arms' lists closed up behind it."""
    keys = {rid: statement_key(hit.payload) for rid, hit in pool.hits.items()}
    members: dict[tuple, list[str]] = {}
    for rid, key in keys.items():
        if key is not None:
            members.setdefault(key, []).append(rid)
    groups = [rids for rids in members.values() if len(rids) > 1]
    if not groups:
        return pool
    base = fused(pool)
    shown: dict[str, str] = {}
    forms: dict[str, list[SearchHit]] = {}
    for rids in groups:
        ordered = sorted(rids, key=lambda r: (-base[r], r))
        # the turn as said stands for its statement when it is in hand: it holds every
        # reading of it whole (a list of codes, a format's every column)
        turn = next((r for r in ordered if pool.hits[r].payload.get("category") == TURN), None)
        if turn is not None:
            ordered.remove(turn)
            ordered.insert(0, turn)
        forms[ordered[0]] = [pool.hits[r] for r in ordered]
        shown.update(dict.fromkeys(ordered, ordered[0]))
    ranks: dict[VectorName, dict[str, int]] = {}
    for name, listed in pool.ranks.items():
        closed: dict[str, int] = {}
        for rid, _ in sorted(listed.items(), key=lambda item: item[1]):
            closed.setdefault(shown.get(rid, rid), len(closed) + 1)
        ranks[name] = closed
    hits = {rid: hit for rid, hit in pool.hits.items() if shown.get(rid, rid) == rid}
    return ArmPool(hits=hits, ranks=ranks, forms=forms)


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
    when = bool(WHEN.search(query) or WHEN_FOREIGN.search(query))
    named_periods = periods.query_periods(query, now or datetime.now(UTC).date())
    scores = {}
    for rid, score in base.items():
        payloads = pool.payloads(rid)
        speaks = bool(named) and bool(pool.speakers(rid) & named)
        timed = when and any(_timed(str(p.get("text", ""))) for p in payloads)
        dated = bool(named_periods) and any(
            periods.within(named_periods, periods.memory_days(p)) for p in payloads
        )
        scores[rid] = (
            score
            + SESSION * best.get(pool.session(rid), 0.0)
            + (SPEAKER * unit if speaks else 0.0)
            + (TIME * unit if timed else 0.0)
            + (PERIOD * unit if dated else 0.0)
        )
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _timed(text: str) -> bool:
    return bool(TIMED.search(text) or TIMED_FOREIGN.search(text))


def _session_best(pool: ArmPool, base: Mapping[str, float]) -> dict[str, float]:
    best: dict[str, float] = {}
    for rid in sorted(base, key=lambda r: (-base[r], r))[:SESSION_DEPTH]:
        best.setdefault(pool.session(rid), base[rid])
    return best


def _named(pool: ArmPool, query: str) -> set[str]:
    """The people the question names, among the subjects of everything the arms found."""
    lowered = query.lower()
    speakers = set().union(*(pool.speakers(rid) for rid in pool.hits)) - {""}
    return {s for s in speakers if re.search(rf"\b{re.escape(s)}\b", lowered)}
