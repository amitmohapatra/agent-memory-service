"""Text equality does not imply equal speakers, events or source evidence."""

from copy import deepcopy

import pytest

from memory_service.modules.retrieval.engine import Candidate, _dedup

pytestmark = pytest.mark.unit


def memory(identity: str, **payload) -> Candidate:
    return Candidate(
        identity,
        "memory",
        "I visited the museum last weekend.",
        0.5,
        payload={
            "subject": "user:alex",
            "owner_principal": "user:alex",
            "observed_at": "2026-01-12T10:00:00Z",
            "source_refs": [{"source_type": "observation", "source_id": "visit-1"}],
            **payload,
        },
    )


@pytest.mark.parametrize(
    "change",
    [
        {"subject": "user:sam"},
        {"owner_principal": "user:sam"},
        {"observed_at": "2026-02-12T10:00:00Z"},
        {"source_refs": [{"source_type": "observation", "source_id": "visit-2"}]},
        {"source_refs": []},
    ],
)
def test_distinct_events_and_attributions_survive(change):
    assert len(_dedup([memory("first"), memory("second", **change)])) == 2


def test_source_fact_collapses_into_its_own_complete_turn_only():
    full = memory("turn")
    full.text += " I went with Jo, who lives in Paris."
    same = memory("fact")
    other = memory("other", subject="user:jo")
    assert [c.record_id for c in _dedup([full, same, other])] == ["turn", "other"]


@pytest.mark.parametrize("subject", [None, "thread:work-thread", "workspace:project"])
def test_unlabelled_first_person_excerpt_keeps_its_source_author_attribution(subject):
    full = memory("turn")
    full.text += " I went with Jo, who lives in Paris."
    excerpt = memory("excerpt", subject=subject)
    assert [c.record_id for c in _dedup([full, excerpt])] == ["turn"]


def test_scope_placeholder_without_known_author_cannot_establish_attribution():
    assert (
        len(
            _dedup(
                [
                    memory("a", subject="thread:one", owner_principal=None),
                    memory("b", subject="thread:one", owner_principal=None),
                ]
            )
        )
        == 2
    )


def test_repeated_retriever_hits_update_the_surviving_representative():
    full = memory("turn")
    full.text += " I went with Jo, who lives in Paris."
    fact = memory("fact")
    repeated = deepcopy(fact)
    repeated.score = 0.9
    repeated.retrievers = ["graph"]
    out = _dedup([full, fact, repeated])
    assert len(out) == 1
    assert out[0].score == 0.9
    assert out[0].retrievers == ["graph"]
    assert out[0].payload["duplicates"] == ["fact"]


def test_unknown_sources_are_not_assumed_to_be_the_same_event():
    assert len(_dedup([memory("a", source_refs=[]), memory("b", source_refs=[])])) == 2


def test_document_and_memory_with_equal_text_are_not_interchangeable():
    item = memory("m")
    chunk = Candidate("c", "chunk", item.text, 0.5)
    duplicate = Candidate("d", "chunk", item.text, 0.4)
    assert [c.record_id for c in _dedup([chunk, item, duplicate])] == ["c", "m"]
