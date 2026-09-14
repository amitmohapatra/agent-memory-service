"""Chat segment format: JSONL + zstd, one record per message, sized by policy.

Object naming avoids purely sequential hot prefixes::

    tenant-shard=<hash % shards>/tenant=<t>/year=YYYY/month=MM/thread=<thr>/segment=<id>.jsonl.zst
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import zstandard

from memory_service.domain.conversation import Message
from memory_service.domain.ids import content_hash, new_id

SEGMENT_FORMAT = "jsonl+zstd/v1"


def message_record(m: Message) -> dict:
    """The archival representation of a message (raw evidence, never summarised)."""
    return {
        "message_id": m.message_id,
        "thread_id": m.thread_id,
        "session_id": m.session_id,
        "turn_id": m.turn_id,
        "tenant_id": m.tenant_id,
        "sequence": m.sequence,
        "role": m.role.value,
        "kind": m.kind.value,
        "version": m.version,
        "author_principal": m.author_principal,
        "agent_run_id": m.agent_run_id,
        "parent_message_id": m.parent_message_id,
        "occurred_at": m.occurred_at.isoformat(),
        "created_at": m.created_at.isoformat(),
        "content": m.content,
        "content_hash": m.content_hash,
        "attachments": [a.model_dump(mode="json") for a in m.attachments],
        "source_system": m.source_system,
        "source_message_id": m.source_message_id,
        "custom_metadata": m.custom_metadata,
    }


@dataclass(frozen=True)
class SegmentPlan:
    messages: list[Message]

    @property
    def raw_bytes(self) -> int:
        return sum(len(m.content.encode("utf-8")) for m in self.messages)


def plan_segments(
    messages: Sequence[Message],
    *,
    target_compressed_bytes: int,
    max_messages: int,
    expected_ratio: float = 5.0,  # measured 5.5-6.1x (benchmark/results/storage.json)
) -> list[SegmentPlan]:
    """Group messages (already ordered by sequence) into segments.

    The compressed size is only known after compression, so planning uses an expected
    compression ratio; the benchmark (``make bench-storage``) measures the real ratio and the
    resulting segment sizes so the target can be tuned rather than guessed.
    """
    plans: list[SegmentPlan] = []
    current: list[Message] = []
    current_bytes = 0
    budget = int(target_compressed_bytes * expected_ratio)
    for m in messages:
        size = len(m.content.encode("utf-8")) + 256
        if current and (current_bytes + size > budget or len(current) >= max_messages):
            plans.append(SegmentPlan(current))
            current, current_bytes = [], 0
        current.append(m)
        current_bytes += size
    if current:
        plans.append(SegmentPlan(current))
    return plans


@dataclass(frozen=True)
class BuiltSegment:
    segment_id: str
    key: str
    data: bytes
    checksum_sha256: str
    raw_bytes: int
    message_ids: list[str]
    first_sequence: int
    last_sequence: int
    first_at: datetime
    last_at: datetime
    manifest: dict


def tenant_shard(tenant_id: str, shards: int) -> int:
    return int(hashlib.sha256(tenant_id.encode()).hexdigest()[:8], 16) % shards


def segment_key(
    tenant_id: str, thread_id: str, segment_id: str, *, shards: int, at: datetime
) -> str:
    return (
        f"tenant-shard={tenant_shard(tenant_id, shards):03d}/tenant={tenant_id}/"
        f"year={at:%Y}/month={at:%m}/thread={thread_id}/segment={segment_id}.jsonl.zst"
    )


def build_segment(
    messages: Sequence[Message], *, tenant_id: str, thread_id: str, shards: int, zstd_level: int = 6
) -> BuiltSegment:
    if not messages:
        raise ValueError("cannot build an empty segment")
    segment_id = new_id("segment")
    lines = [
        json.dumps(message_record(m), separators=(",", ":"), ensure_ascii=False, default=str)
        for m in messages
    ]
    raw = ("\n".join(lines) + "\n").encode("utf-8")
    data = zstandard.ZstdCompressor(level=zstd_level).compress(raw)
    first_at = min(m.occurred_at for m in messages)
    manifest = {
        "format": SEGMENT_FORMAT,
        "segment_id": segment_id,
        "tenant_id": tenant_id,
        "thread_id": thread_id,
        "message_count": len(messages),
        "message_ids": [m.message_id for m in messages],
        "sessions": sorted({m.session_id for m in messages}),
        "turns": sorted({m.turn_id for m in messages}),
        "agent_runs": sorted({m.agent_run_id for m in messages if m.agent_run_id}),
        "first_sequence": messages[0].sequence,
        "last_sequence": messages[-1].sequence,
        "first_at": first_at.isoformat(),
        "last_at": max(m.occurred_at for m in messages).isoformat(),
        "raw_bytes": len(raw),
        "compressed_bytes": len(data),
        "source_hashes": {m.message_id: m.content_hash for m in messages},
        "built_at": datetime.now(UTC).isoformat(),
    }
    return BuiltSegment(
        segment_id=segment_id,
        key=segment_key(tenant_id, thread_id, segment_id, shards=shards, at=first_at),
        data=data,
        checksum_sha256=content_hash(data),
        raw_bytes=len(raw),
        message_ids=[m.message_id for m in messages],
        first_sequence=messages[0].sequence,
        last_sequence=messages[-1].sequence,
        first_at=first_at,
        last_at=max(m.occurred_at for m in messages),
        manifest=manifest,
    )


def read_segment(data: bytes) -> Iterable[dict]:
    raw = zstandard.ZstdDecompressor().decompress(data, max_output_size=512 * 1024 * 1024)
    for line in raw.decode("utf-8").splitlines():
        if line:
            yield json.loads(line)
