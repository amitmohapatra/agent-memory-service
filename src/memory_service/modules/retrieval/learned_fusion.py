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
pool the learned score orders. Every candidate in it is described by twelve numbers - its
reciprocal rank in each arm and each first stage, the best first-stage score of its
neighbours, and whether a "when" question meets a candidate that names a time - and
``COEFFICIENTS`` turns them into one score. The pool is ordered by that score; everything
else the arms found follows it, in ``FIRST_B`` order.

No cross-encoder: two were measured (``mmarco-mMiniLMv2``, ``mxbai-rerank-xsmall``) and lift
recall@10 by about two and a half points, at 2.4 CPU-seconds a query - forty-eight cores at
the 20 requests a second the service is sized for. Without them a memory search costs
0.17 CPU-seconds.

``COEFFICIENTS`` were fitted on LoCoMo's ten conversations (58,193 candidate rows, 1,913 of
them evidence). Evaluated leave-one-conversation-out, so no conversation is scored by a
model that saw it, recall@10 is 0.815 against 0.724 for the hybrid search before this ADR.
The same coefficients, unchanged, are what LongMemEval is scored with.

The model assumes the late-interaction arm; every container wires one (the hash stand-in in
the hermetic suite).

Pure functions over hits and scores: the store and the models are the engine's business.
"""

from __future__ import annotations

import math
import re
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

_OWN = (VectorName.BM25, VectorName.DENSE_EN, VectorName.DENSE_ML)


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
    "dense_ml_ctx",
    "dense_ml",
    "bm25_ctx",
    "bm25",
    "colbert",
    "dense_en_ctx",
    "dense_en",
    "first_a",
    "first_b",
    "neighbour",
    "when",
    "when_time",
)
#: Fitted with scikit-learn's ``LogisticRegression(C=1, class_weight="balanced")``. The
#: intercept moves every candidate of a query alike and is kept only so ``probability`` is one.
COEFFICIENTS: Mapping[str, float] = {
    "dense_ml_ctx": 2.022573,
    "dense_ml": 1.32985,
    "bm25_ctx": 2.323338,
    "bm25": -0.368609,
    "colbert": 13.997494,
    "dense_en_ctx": -1.82581,
    "dense_en": -2.027476,
    "first_a": 1.352318,
    "first_b": -6.065822,
    "neighbour": 0.484646,
    "when": -1.483238,
    "when_time": 1.794904,
}
INTERCEPT = -2.234446


@dataclass
class ArmPool:
    """Everything the arms returned, by record: payloads, each arm's ranks, neighbours."""

    hits: dict[str, SearchHit]
    #: arm -> record -> one-based rank in that arm
    ranks: dict[VectorName, dict[str, int]]
    previous: dict[str, str]
    following: dict[str, str]

    @classmethod
    def of(cls, arms: Mapping[VectorName, Sequence[SearchHit]]) -> ArmPool:
        hits: dict[str, SearchHit] = {}
        ranks: dict[VectorName, dict[str, int]] = {}
        for name, listed in arms.items():
            ranks[name] = {}
            for rank, hit in enumerate(listed, start=1):
                hits.setdefault(hit.record_id, hit)
                ranks[name].setdefault(hit.record_id, rank)
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
        return cls(hits=hits, ranks=ranks, previous=previous, following=following)

    def reciprocal(self, name: VectorName, rid: str) -> float:
        rank = self.ranks.get(name, {}).get(rid)
        return 0.0 if rank is None else 1.0 / (1 + rank)


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
    rank_a = {rid: n for n, rid in enumerate(_ranked(a), start=1)}
    rank_b = {rid: n for n, rid in enumerate(_ranked(b), start=1)}
    when = 1.0 if WHEN.search(query) else 0.0
    rows = []
    for rid in candidates:
        neighbours = [a[n] for n in (pool.previous.get(rid), pool.following.get(rid)) if n]
        text = str(pool.hits[rid].payload.get("text", ""))
        row = {
            "dense_ml_ctx": pool.reciprocal(VectorName.DENSE_ML_CTX, rid),
            "dense_ml": pool.reciprocal(VectorName.DENSE_ML, rid),
            "bm25_ctx": pool.reciprocal(VectorName.BM25_CTX, rid),
            "bm25": pool.reciprocal(VectorName.BM25, rid),
            "colbert": pool.reciprocal(VectorName.COLBERT, rid),
            "dense_en_ctx": pool.reciprocal(VectorName.DENSE_EN_CTX, rid),
            "dense_en": pool.reciprocal(VectorName.DENSE_EN, rid),
            "first_a": 1.0 / (1 + rank_a[rid]),
            "first_b": 1.0 / (1 + rank_b[rid]),
            "neighbour": max(neighbours, default=0.0),
            "when": when,
            "when_time": when * (1.0 if TIME.search(text) else 0.0),
        }
        rows.append([row[name] for name in FEATURES])
    return rows


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
