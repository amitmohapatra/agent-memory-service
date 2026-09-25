"""``_dedup`` was made linear-ish. This proves it did not change what it returns.

The old pass re-normalised every kept candidate's body on every comparison - O(n^2) string
allocations for a scan needing n of them - and called ``_subsumed_by`` twice per candidate,
once to test and once to fetch what the test had already found. Measured at 9.6-26 ms per
query and sitting inside no timing stage, which is how it stayed invisible in a p99 already
over its 300 ms budget.

A latency change to a ranking function is only safe if the ranking is untouched, so the
acceptance criterion is output identity rather than "the tests still pass". The reference
below is the OLD algorithm, written out; the property is that both agree on randomised pools
built from the shapes this service actually produces - verbatim turns, propositions extracted
from them, exact twins, and unrelated text.
"""

from __future__ import annotations

import copy
import random

import pytest

from memory_service.modules.retrieval import engine as eng
from memory_service.modules.retrieval.engine import (
    SUBSUMPTION_MIN_CHARS,
    Candidate,
    _dedup,
    _normalised,
    content_hash,
)

pytestmark = pytest.mark.unit

TURNS = [
    "I prefer tea in the afternoon, and my name is Amit, and I moved to Berlin last year",
    "The build server runs Ubuntu 22.04 and it rebuilds the index every night at two",
    "Melanie adopted a cat called Oscar in the spring and she posts photographs of him",
]


def _reference(candidates: list[Candidate]) -> list[Candidate]:
    """The algorithm as it was before the change, normalising inside the inner loop."""
    seen: dict[str, Candidate] = {}
    by_hash: dict[str, Candidate] = {}
    for c in candidates:
        if c.record_id in seen:
            existing = seen[c.record_id]
            existing.retrievers = sorted(set(existing.retrievers) | set(c.retrievers))
            existing.score = max(existing.score, c.score)
            continue
        h = c.payload.get("text_hash") or (content_hash(c.text) if c.text else None)
        if h:
            twin = by_hash.get(h)
            if twin is not None:
                twin.payload.setdefault("duplicates", []).append(c.record_id)
                twin.score = max(twin.score, c.score)
                twin.retrievers = sorted(set(twin.retrievers) | set(c.retrievers))
                continue
            by_hash[h] = c
        if eng.COLLAPSE_SUBSUMED:
            text = _normalised(c.text)
            found = None
            if len(text) >= SUBSUMPTION_MIN_CHARS:
                for other in seen.values():
                    if other.record_id == c.record_id:
                        continue
                    body = _normalised(other.text)
                    if len(body) > len(text) and text in body:
                        found = other
                        break
            if found is not None:
                found.payload.setdefault("duplicates", []).append(c.record_id)
                found.retrievers = sorted(set(found.retrievers) | set(c.retrievers))
                continue
        seen[c.record_id] = c
    return list(seen.values())


def _pool(rng: random.Random, n: int) -> list[Candidate]:
    out: list[Candidate] = []
    for i in range(n):
        turn = rng.choice(TURNS)
        shape = rng.random()
        if shape < 0.35:  # a proposition carved out of a turn - the subsumption case
            words = turn.split()
            cut = rng.randint(4, max(5, len(words) - 2))
            text = " ".join(words[:cut])
        elif shape < 0.5:  # the verbatim turn
            text = turn
        elif shape < 0.62:  # an exact twin of something already emitted
            text = out[rng.randrange(len(out))].text if out else turn
        else:
            text = f"unrelated observation number {i} about an entirely different subject"
        out.append(
            Candidate(
                record_id=f"r{i}",
                kind="memory",
                text=text,
                score=round(rng.random(), 6),
                retrievers=[rng.choice(["dense", "bm25", "graph"])],
                payload={},
            )
        )
    return out


@pytest.mark.parametrize("collapse", [True, False])
@pytest.mark.parametrize("seed", range(40))
def test_output_is_identical_to_the_previous_algorithm(
    seed: int, collapse: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(eng, "COLLAPSE_SUBSUMED", collapse)
    rng = random.Random(seed)
    pool = _pool(rng, rng.randint(2, 60))

    got = _dedup(copy.deepcopy(pool))
    want = _reference(copy.deepcopy(pool))

    assert [c.record_id for c in got] == [c.record_id for c in want], "kept set or order moved"
    assert [c.score for c in got] == [c.score for c in want]
    assert [sorted(c.retrievers) for c in got] == [sorted(c.retrievers) for c in want]
    assert [c.payload.get("duplicates") for c in got] == [
        c.payload.get("duplicates") for c in want
    ], "a collapse was recorded against a different survivor"


def test_the_stage_is_timed_so_it_cannot_hide_from_the_budget_again() -> None:
    """The cost was real and invisible: no stage covered it, so it never appeared in a p99
    breakdown that was already over budget."""
    from memory_service.observability.metrics import stage_seconds

    before = stage_seconds.labels("retrieval.dedup")._sum.get()
    _dedup(_pool(random.Random(1), 30))
    assert stage_seconds.labels("retrieval.dedup")._sum.get() > before
