from datetime import UTC, datetime, timedelta

from memory_service.domain.conversation import Message
from memory_service.domain.enums import MessageRole
from memory_service.domain.ids import content_hash
from memory_service.modules.archive.segments import (
    build_segment,
    plan_segments,
    read_segment,
    segment_key,
    tenant_shard,
)


def _msgs(n: int, size: int = 100) -> list[Message]:
    base = datetime(2026, 9, 14, tzinfo=UTC)
    return [
        Message(
            thread_id="thr_1",
            session_id="ses_1",
            turn_id="trn_1",
            tenant_id="acme",
            role=MessageRole.USER,
            sequence=i + 1,
            content=("x" * size) + str(i),
            content_hash=content_hash("x" * size + str(i)),
            author_principal="user:u1",
            occurred_at=base + timedelta(seconds=i),
        )
        for i in range(n)
    ]


def test_segment_roundtrip_preserves_raw_evidence() -> None:
    messages = _msgs(5)
    built = build_segment(messages, tenant_id="acme", thread_id="thr_1", shards=64)
    assert built.key.startswith(
        f"tenant-shard={tenant_shard('acme', 64):03d}/tenant=acme/year=2026/month=09/thread=thr_1/segment="
    )
    assert built.key.endswith(".jsonl.zst")
    records = list(read_segment(built.data))
    assert [r["message_id"] for r in records] == [m.message_id for m in messages]
    assert all(r["content_hash"] == content_hash(r["content"]) for r in records)
    assert built.manifest["message_count"] == 5 and built.manifest["first_sequence"] == 1
    assert built.manifest["compressed_bytes"] < built.manifest["raw_bytes"]


def test_plan_segments_respects_limits() -> None:
    plans = plan_segments(
        _msgs(10, size=1000), target_compressed_bytes=1500, max_messages=100, expected_ratio=1.0
    )
    assert len(plans) > 1 and sum(len(p.messages) for p in plans) == 10
    plans = plan_segments(_msgs(10), target_compressed_bytes=10**9, max_messages=4)
    assert [len(p.messages) for p in plans] == [4, 4, 2]


def test_shard_is_stable_and_bounded() -> None:
    assert tenant_shard("acme", 64) == tenant_shard("acme", 64)
    assert 0 <= tenant_shard("globex", 16) < 16
    assert (
        segment_key("t", "thr", "seg", shards=8, at=datetime(2026, 1, 5, tzinfo=UTC)).count("/")
        == 5
    )
