"""Raw-source provenance survives retrieval without changing memory citations."""

from datetime import UTC, datetime

import pytest

from memory_service.domain.context_bundle import _head_with_siblings, _source_ids
from memory_service.domain.evidence import EvidenceRef
from memory_service.modules.context.builder import candidate_to_item
from memory_service.modules.retrieval.engine import Candidate

pytestmark = pytest.mark.unit
WHEN = datetime(2023, 5, 8, tzinfo=UTC)


def item(identifier, *sources):
    return candidate_to_item(
        Candidate(
            record_id=identifier,
            kind="memory",
            text=identifier,
            score=0.5,
            payload={
                "source_refs": [ref.model_dump(mode="json", exclude_none=True) for ref in sources]
            },
        )
    )


def ref(identifier, source_type="message"):
    return EvidenceRef(source_id=identifier, source_type=source_type, observed_at=WHEN)


def test_preserves_all_source_locators_and_original_time():
    original = ref("msg_1").model_copy(
        update={
            "message_id": "msg_1",
            "span_start": 3,
            "span_end": 12,
            "source_hash": "abc",
        }
    )
    found = item("mem_1", original, ref("obs_2"))
    assert found.evidence == [original, ref("obs_2")]
    assert found.citation == "memory_id:mem_1"


def test_legacy_and_working_memories_keep_record_pointer():
    assert item("mem_old").evidence[0].source_id == "mem_old"
    assert item("wm_0").evidence[0].source_id == "wm_0"


def test_source_siblings_join_head_without_changing_ranked_candidates():
    memories = [item("a", ref("turn1")), item("b", ref("turn2")), item("c", ref("turn1"))]
    assert [m.item_id for m in _head_with_siblings(memories, 2)] == ["a", "c"]
    assert [m.item_id for m in memories] == ["a", "b", "c"]


def test_source_namespace_and_order_are_preserved():
    seed = item("a", ref("z"), ref("a"), ref("z"))
    assert _source_ids(seed) == [("message", "z"), ("message", "a")]
    memories = [seed, item("b", ref("z", "file")), item("c", ref("a")), item("d", ref("z"))]
    assert [m.item_id for m in _head_with_siblings(memories, 3)] == ["a", "d", "c"]
