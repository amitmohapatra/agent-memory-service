"""Deduplication preserves evidence and aggregates every dropped representation.

The former oracle copied the pre-provenance implementation, including its attribution
loss and discarded-alias score bug. Output identity to that implementation is no longer
a valid contract. These randomized checks assert coverage, provenance, order, score and
retriever conservation independently of the optimized grouping implementation.
"""

from __future__ import annotations

import copy
import random

import pytest

from memory_service.modules.retrieval import engine as eng
from memory_service.modules.retrieval.engine import SUBSUMPTION_MIN_CHARS, Candidate, _dedup

pytestmark = pytest.mark.unit

TURNS = [
    "I prefer tea in the afternoon, and my name is Amit, and I moved to Berlin last year",
    "The build server runs Ubuntu 22.04 and it rebuilds the index every night at two",
    "Melanie adopted a cat called Oscar in the spring and she posts photographs of him",
]


def _pool(rng: random.Random, n: int) -> list[Candidate]:
    out = []
    for i in range(n):
        turn = rng.choice(TURNS)
        shape = rng.random()
        if shape < 0.35:
            words = turn.split()
            text = " ".join(words[: rng.randint(4, max(5, len(words) - 2))])
        elif shape < 0.5:
            text = turn
        elif shape < 0.62:
            text = out[rng.randrange(len(out))].text if out else turn
        else:
            text = f"unrelated observation number {i} about an entirely different subject"
        actor = f"user:{rng.randrange(2)}"
        payload = {
            "subject": actor,
            "owner_principal": actor,
            "observed_at": f"2026-09-{25 + rng.randrange(2)}T12:00:00Z",
            "source_refs": [{"source_type": "message", "source_id": f"source-{rng.randrange(2)}"}],
        }
        if i % 7 == 0:
            payload = {}  # legacy hits do not prove common attribution
        out.append(
            Candidate(
                record_id=f"r{i}",
                kind="memory",
                text=text,
                score=round(rng.random(), 6),
                retrievers=[rng.choice(["dense", "bm25", "graph"])],
                payload=payload,
            )
        )
    return out


@pytest.mark.parametrize("collapse", [True, False])
@pytest.mark.parametrize("seed", range(40))
def test_every_input_keeps_its_evidence_or_a_valid_representative(seed, collapse, monkeypatch):
    monkeypatch.setattr(eng, "COLLAPSE_SUBSUMED", collapse)
    rng = random.Random(seed)
    pool = _pool(rng, rng.randint(2, 80))
    original = {c.record_id: c for c in pool}
    got = _dedup(copy.deepcopy(pool))
    assert [int(c.record_id[1:]) for c in got] == sorted(int(c.record_id[1:]) for c in got)
    represented = []
    for survivor in got:
        ids = [survivor.record_id, *survivor.payload.get("duplicates", [])]
        assert len(ids) == len(set(ids))
        represented.extend(ids)
        group = [original[identity] for identity in ids]
        assert survivor.score == max(c.score for c in group)
        assert set(survivor.retrievers) == {r for c in group for r in c.retrievers}
        assert survivor.text == original[survivor.record_id].text
        for dropped in group[1:]:
            kept = original[survivor.record_id]
            assert dropped.payload and dropped.payload == kept.payload, "attribution changed"
            short, full = (
                " ".join(dropped.text.lower().split()),
                " ".join(kept.text.lower().split()),
            )
            assert dropped.text == kept.text or (
                collapse and len(short) >= SUBSUMPTION_MIN_CHARS and short in full
            )
    assert sorted(represented) == sorted(original), "a source disappeared or was represented twice"


def test_the_stage_is_timed_so_it_cannot_hide_from_the_budget_again():
    from memory_service.observability.metrics import stage_seconds

    before = stage_seconds.labels("retrieval.dedup")._sum.get()
    _dedup(_pool(random.Random(1), 30))
    assert stage_seconds.labels("retrieval.dedup")._sum.get() > before
