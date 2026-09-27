"""Source promotion must preserve facts, scope-filtered candidates and the fixed budget."""

import pytest

from memory_service.modules.retrieval.engine import Candidate
from memory_service.modules.retrieval.source_turns import promote_source_turns

pytestmark = pytest.mark.unit


def candidate(identifier, text, *, source="obs_1", predicate="likes", kind="memory"):
    return Candidate(
        identifier,
        kind,
        text,
        0.4,
        retrievers=["fusion"],
        payload={
            "predicate": predicate,
            "source_refs": [{"source_type": "message", "source_id": source}],
        },
    )


def test_source_turn_moves_to_seed_rank_and_frees_a_slot():
    fact = candidate("fact", "I like pottery.")
    unrelated = candidate("other", "I run on Sundays.", source="obs_2")
    turn = candidate("turn", "I like pottery. I also swim.", predicate="said")
    turn.score = 0.1
    result, count = promote_source_turns([fact, unrelated, turn])
    assert count == 1
    assert [c.record_id for c in result] == ["turn", "other"]
    assert result[0].score == fact.score
    assert result[0].text == turn.text
    assert result[0].expanded_from == "fact"
    assert result[0].retrievers == ["fusion", "source_turn"]
    assert turn.score == 0.1 and turn.expanded_from is None
    assert turn.retrievers == ["fusion"]


def test_truncated_or_different_source_cannot_replace_a_fact():
    fact = candidate("fact", "I also swim.")
    truncated = candidate("turn", "I like pottery.", predicate="said")
    other = candidate("other", "I also swim. I run.", predicate="said", source="obs_2")
    original = [fact, truncated, other]
    result, count = promote_source_turns(original)
    assert result == original
    assert count == 0


def test_legacy_multisource_documents_and_exact_lookups_are_not_promoted():
    turn = candidate("turn", "I like pottery. I also swim.", predicate="said")
    exact = candidate("exact", "I like pottery.")
    exact.retrievers = ["exact"]
    legacy = candidate("legacy", "I like pottery.")
    legacy.payload = {}
    multiple = candidate("multiple", "I like pottery.")
    multiple.payload["source_refs"].append({"source_type": "message", "source_id": "obs_2"})
    document = candidate("chunk", "I like pottery.", kind="chunk")
    original = [exact, legacy, multiple, document, turn]
    result, count = promote_source_turns(original)
    assert result == original
    assert count == 0


def test_identical_source_id_in_a_different_namespace_is_not_a_parent():
    fact = candidate("fact", "I like pottery.")
    turn = candidate("turn", "I like pottery. I also swim.", predicate="said")
    turn.payload["source_refs"][0]["source_type"] = "file"
    assert promote_source_turns([fact, turn]) == ([fact, turn], 0)
