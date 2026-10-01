"""The memories' ranking: every arm read unfused, two first stages, and a logistic regression
over both (ADR 0025).

A memory search reads seven ranked lists from the store in one round trip - BM25 and both
dense spaces over each of the two keys (the memory alone, and the memory read with the turn
it answers), and the late-interaction arm over their union. Two first stages fuse them by
weighted reciprocal rank, each with a neighbour lift (a turn gains a share of the score of
the turns either side of it in the same conversation):

* ``FIRST_A``: the dense-and-lexical weights fitted for the hybrid search, the late arm
  added after the lift;
* ``FIRST_B``: BM25 and the multilingual space only, the late arm three times heavier, the
  lift smaller and over the late arm too.

Each catches evidence the other ranks low, so the top ``pool_k`` of each, together, is the
pool the learned score orders. Every candidate in it is described by twenty-six numbers
(``FEATURES``), all read from what the arms returned:

* its reciprocal rank in each arm and each first stage, and each arm's own score, min-max
  normalised within the pool - a rank says "first", a score says by how much;
* its neighbours: the best first-stage score and the best late-interaction score of the
  turns either side of it;
* how many of the seven arms put it in their top ten;
* its session (the day it was said on): the best first-stage score in that session, and how
  many of the first stage's top 30 that session holds - evidence comes in runs;
* whether the question names the person it is about (any subject among the arms' hits) and
  whether this memory is that person's;
* whether a "when" question meets a candidate that names a time, and the candidate's length.

``COEFFICIENTS`` turn them into one score. The pool is ordered by it; everything else the
arms found follows, in ``FIRST_B`` order.

No cross-encoder. Two were measured as further features (``mmarco-mMiniLMv2``,
``mxbai-rerank-xsmall``): with the rank features alone they lifted recall@10 from 0.815 to
0.841, at 2.4 CPU-seconds a query - forty-eight cores at the 20 requests a second the
service is sized for. The scores, neighbours, sessions and speakers above reach the same
0.841 from what the arms already returned, at no model cost.

``COEFFICIENTS`` were fitted on LoCoMo's ten conversations (58,193 candidate rows, 1,913 of
them evidence) on standardised features, the standardisation folded into the weights.
Evaluated leave-one-conversation-out, so no conversation is scored by a model that saw it,
recall@10 is 0.841 against 0.724 for the hybrid search before this ADR. The same
coefficients, unchanged, are what LongMemEval is scored with.

The model assumes the late-interaction arm; every container wires one (the hash stand-in in
the hermetic suite).

Pure functions over hits and scores: the store and the models are the engine's business.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from memory_service.ports.search import SearchHit, VectorName

#: "when ...", "what year ...", "how long ...", "before"/"after": a question about time
WHEN = re.compile(
    r"^\s*(when|what (year|month|date|day|time)|how long|how many (days|weeks|months|years)"
    r"|since when)\b|\b(before|after)\b",
    re.IGNORECASE,
)
#: a candidate that names a time
TIME = re.compile(
    r"\b(yesterday|today|tonight|tomorrow|last (night|week|weekend|month|year|summer|winter"
    r"|spring|fall|time)|ago|recently|next (week|month|year)|this (week|weekend|month|year"
    r"|morning)|(19|20)\d\d|january|february|march|april|may|june|july|august|september"
    r"|october|november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|weeks?|months?|years?|days?)\b",
    re.IGNORECASE,
)

#: A key fusion keeps only its top this-many; below it a turn scores nothing from that key.
KEY_DEPTH = 100

#: Every arm a memory search reads, in the order the features name them.
ARMS: tuple[VectorName, ...] = (
    VectorName.DENSE_ML_CTX,
    VectorName.DENSE_ML,
    VectorName.BM25_CTX,
    VectorName.BM25,
    VectorName.COLBERT,
    VectorName.DENSE_EN_CTX,
    VectorName.DENSE_EN,
)
#: The session statistics read the first stage's top this-many, and count its top this-many.
SESSION_DEPTH = 100
SESSION_TOP = 30


@dataclass(frozen=True)
class FirstStage:
    """Weighted reciprocal-rank fusion of both keys, a neighbour lift, and the late arm."""

    weights: Mapping[VectorName, float]
    late: float
    lift: float
    #: whether the late arm is added before the lift (and so lifted with the rest)
    late_lifted: bool


FIRST_A = FirstStage(
    weights={VectorName.BM25: 2.0, VectorName.DENSE_EN: 0.5, VectorName.DENSE_ML: 2.0},
    late=2.0,
    lift=0.25,
    late_lifted=False,
)
FIRST_B = FirstStage(
    weights={VectorName.BM25: 1.0, VectorName.DENSE_ML: 1.0},
    late=6.0,
    lift=0.1,
    late_lifted=True,
)

#: The features, in the order ``COEFFICIENTS`` reads them.
FEATURES: tuple[str, ...] = (
    "rank_dense_ml_ctx",
    "rank_dense_ml",
    "rank_bm25_ctx",
    "rank_bm25",
    "rank_colbert",
    "rank_dense_en_ctx",
    "rank_dense_en",
    "rank_first_a",
    "rank_first_b",
    "score_dense_ml_ctx",
    "score_dense_ml",
    "score_bm25_ctx",
    "score_bm25",
    "score_colbert",
    "score_dense_en_ctx",
    "score_dense_en",
    "neighbour",
    "neighbour_colbert",
    "when",
    "when_time",
    "arms_top10",
    "session_best",
    "session_top",
    "speaker_named",
    "any_named",
    "length",
)
#: Fitted with scikit-learn's ``LogisticRegression(C=1, class_weight="balanced")`` over
#: standardised features, the scaling folded in. The intercept moves every candidate of a
#: query alike and is kept only so ``probability`` is one.
COEFFICIENTS: Mapping[str, float] = {
    "rank_dense_ml_ctx": 0.5693,
    "rank_dense_ml": 0.140615,
    "rank_bm25_ctx": 0.400681,
    "rank_bm25": 0.669265,
    "rank_colbert": 4.337692,
    "rank_dense_en_ctx": 1.434214,
    "rank_dense_en": -1.299781,
    "rank_first_a": -0.49483,
    "rank_first_b": -2.324544,
    "score_dense_ml_ctx": 1.101677,
    "score_dense_ml": -0.179775,
    "score_bm25_ctx": 0.984314,
    "score_bm25": -0.527763,
    "score_colbert": 4.837351,
    "score_dense_en_ctx": -0.002655,
    "score_dense_en": -1.492302,
    "neighbour": 0.495776,
    "neighbour_colbert": -0.31659,
    "when": -1.16454,
    "when_time": 1.518581,
    "arms_top10": -0.346267,
    "session_best": 1.781182,
    "session_top": -0.374265,
    "speaker_named": 2.593249,
    "any_named": -2.281676,
    "length": 1.476828,
}
INTERCEPT = -6.177069


@dataclass
class ArmPool:
    """Everything the arms returned, by record: payloads, each arm's ranks, neighbours."""

    hits: dict[str, SearchHit]
    #: arm -> record -> one-based rank in that arm
    ranks: dict[VectorName, dict[str, int]]
    #: arm -> record -> that arm's own score (cosine, BM25, MaxSim)
    scores: dict[VectorName, dict[str, float]]
    previous: dict[str, str]
    following: dict[str, str]

    @classmethod
    def of(cls, arms: Mapping[VectorName, Sequence[SearchHit]]) -> ArmPool:
        hits: dict[str, SearchHit] = {}
        ranks: dict[VectorName, dict[str, int]] = {}
        scores: dict[VectorName, dict[str, float]] = {}
        for name, listed in arms.items():
            ranks[name], scores[name] = {}, {}
            for rank, hit in enumerate(listed, start=1):
                hits.setdefault(hit.record_id, hit)
                ranks[name].setdefault(hit.record_id, rank)
                scores[name].setdefault(hit.record_id, hit.score)
        by_source: dict[str, str] = {}
        for rid, hit in hits.items():
            if source := _source_id(hit):
                by_source.setdefault(source, rid)
        previous: dict[str, str] = {}
        following: dict[str, str] = {}
        for rid, hit in hits.items():
            before = hit.payload.get("preceding_source_id")
            if before and (prior := by_source.get(str(before))) and prior != rid:
                previous[rid] = prior
                following.setdefault(prior, rid)
        return cls(hits=hits, ranks=ranks, scores=scores, previous=previous, following=following)

    def reciprocal(self, name: VectorName, rid: str) -> float:
        rank = self.ranks.get(name, {}).get(rid)
        return 0.0 if rank is None else 1.0 / (1 + rank)

    def score(self, name: VectorName, rid: str) -> float:
        """The arm's score for ``rid``; below the arm's depth, the lowest score it returned
        (the record scores no more than that, and the true value was never read)."""
        listed = self.scores.get(name) or {}
        if rid in listed:
            return listed[rid]
        return min(listed.values(), default=0.0)

    def speaker(self, rid: str) -> str:
        """Who the memory is about: its subject's name, lower-cased (``user:caroline``)."""
        return str(self.hits[rid].payload.get("subject") or "").split(":", 1)[-1].lower()

    def session(self, rid: str) -> str:
        """The day the memory was observed on: a conversation's sessions are its days."""
        return str(self.hits[rid].payload.get("observed_at") or "")[:10]


def _source_id(hit: SearchHit) -> str | None:
    refs = hit.payload.get("source_refs") or []
    first = refs[0] if refs and isinstance(refs[0], dict) else {}
    return str(first["source_id"]) if first.get("source_id") else None


def _ranked(scores: Mapping[str, float]) -> list[str]:
    return sorted(scores, key=lambda rid: (-scores[rid], rid))


def first_stage(pool: ArmPool, stage: FirstStage) -> dict[str, float]:
    """One first stage's score for every record the arms found."""
    total = dict.fromkeys(pool.hits, 0.0)
    for context in (False, True):
        key = {
            rid: sum(
                weight * pool.reciprocal(name.context if context else name, rid)
                for name, weight in stage.weights.items()
            )
            for rid in pool.hits
        }
        for rid in _ranked(key)[:KEY_DEPTH]:
            total[rid] += key[rid]
    late = {rid: stage.late * pool.reciprocal(VectorName.COLBERT, rid) for rid in pool.hits}
    if stage.late_lifted:
        total = {rid: total[rid] + late[rid] for rid in total}
    lifted = {
        rid: score
        + stage.lift * total.get(pool.previous.get(rid, ""), 0.0)
        + stage.lift * total.get(pool.following.get(rid, ""), 0.0)
        for rid, score in total.items()
    }
    if not stage.late_lifted:
        lifted = {rid: lifted[rid] + late[rid] for rid in lifted}
    return lifted


def scoring_pool(a: Mapping[str, float], b: Mapping[str, float], k: int) -> list[str]:
    """The top ``k`` of each first stage, the first stage's order kept, without repeats."""
    return list(dict.fromkeys([*_ranked(a)[:k], *_ranked(b)[:k]]))


def features(
    pool: ArmPool,
    candidates: Sequence[str],
    *,
    a: Mapping[str, float],
    b: Mapping[str, float],
    query: str,
) -> list[list[float]]:
    """``FEATURES`` for each candidate."""
    ranked_a = _ranked(a)
    rank_a = {rid: n for n, rid in enumerate(ranked_a, start=1)}
    rank_b = {rid: n for n, rid in enumerate(_ranked(b), start=1)}
    when = 1.0 if WHEN.search(query) else 0.0
    lowered = query.lower()
    speakers = {pool.speaker(rid) for rid in pool.hits} - {""}
    named = {s for s in speakers if re.search(rf"\b{re.escape(s)}\b", lowered)}
    best: dict[str, float] = defaultdict(float)
    counted: dict[str, int] = defaultdict(int)
    top = set(ranked_a[:SESSION_TOP])
    for rid in ranked_a[:SESSION_DEPTH]:
        day = pool.session(rid)
        best[day] = max(best[day], a[rid])
        counted[day] += rid in top
    peak = max(a.values(), default=0.0) or 1.0
    raw = {name: [pool.score(name, rid) for rid in candidates] for name in ARMS}
    span = {name: (min(v), max(v)) for name, v in raw.items() if v}
    rows = []
    for j, rid in enumerate(candidates):
        around = [n for n in (pool.previous.get(rid), pool.following.get(rid)) if n]
        text = str(pool.hits[rid].payload.get("text", ""))
        row = {
            **{f"rank_{name.value}": pool.reciprocal(name, rid) for name in ARMS},
            "rank_first_a": 1.0 / (1 + rank_a[rid]),
            "rank_first_b": 1.0 / (1 + rank_b[rid]),
            **{f"score_{name.value}": _scaled(raw[name][j], span[name]) for name in ARMS},
            "neighbour": max((a[n] for n in around), default=0.0),
            "neighbour_colbert": _scaled(
                max(pool.score(VectorName.COLBERT, n) for n in around), span[VectorName.COLBERT]
            )
            if around
            else 0.0,
            "when": when,
            "when_time": when * (1.0 if TIME.search(text) else 0.0),
            "arms_top10": sum(1 for name in ARMS if pool.ranks.get(name, {}).get(rid, 11) <= 10)
            / len(ARMS),
            "session_best": best.get(pool.session(rid), 0.0) / peak,
            "session_top": counted.get(pool.session(rid), 0) / SESSION_TOP,
            "speaker_named": 1.0 if pool.speaker(rid) in named else 0.0,
            "any_named": 1.0 if named else 0.0,
            "length": math.log1p(len(text)) / 6,
        }
        rows.append([row[name] for name in FEATURES])
    return rows


def _scaled(value: float, span: tuple[float, float]) -> float:
    low, high = span
    return (value - low) / (high - low + 1e-9)


def probability(row: Sequence[float]) -> float:
    z = INTERCEPT + sum(COEFFICIENTS[name] * x for name, x in zip(FEATURES, row, strict=True))
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))


def order(
    pool: ArmPool,
    candidates: Sequence[str],
    rows: Sequence[Sequence[float]],
    b: Mapping[str, float],
) -> list[tuple[str, float]]:
    """The final ranking with a score each: the pool by probability, then every other
    record the arms found in ``FIRST_B`` order, each scored below the whole pool."""
    scored = sorted(
        ((rid, probability(row)) for rid, row in zip(candidates, rows, strict=True)),
        key=lambda item: (-item[1], item[0]),
    )
    floor = min((p for _, p in scored), default=1.0)
    chosen = set(candidates)
    rest = [rid for rid in _ranked(b) if rid not in chosen]
    top = max((b[rid] for rid in rest), default=0.0) or 1.0
    tail = [(rid, floor * 0.5 * b[rid] / top) for rid in rest]
    return [*scored, *tail]
