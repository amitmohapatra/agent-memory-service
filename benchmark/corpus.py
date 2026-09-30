"""One ingested LoCoMo corpus, reused across the arms that leave the write path alone.

Ingesting the ten conversations is the expensive half of a LoCoMo run - forty minutes on an
idle box, hours on a busy one - and every arm of the accuracy programme that changes only
what happens at query time (fusion weights, depth, the entity prefetch, the render) was
paying it again. So each conversation goes into its own tenant once, and a ledger beside the
results records what was ingested: the dataset's hash, the index fingerprint, the ingestion
settings that shape the corpus, and per conversation the observation-id -> turn-id map the
scoring needs. An arm that asks for ``--reuse-corpus`` gets the corpus back when the ledger
says it is the same one and the store still holds it; anything else ingests afresh.

Arms that change the write path (LLM-assisted extraction) get their own
database, so their ledger never matches another arm's.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark.common import RESULTS
from memory_service.modules.rag.indexer import MEMORIES
from memory_service.ports.search import SearchFilter

#: every conversation's tenant, so the arm's questions read one conversation and nothing else
#: The conversation tenants' prefix. An arm that ingests its own corpus into the shared
#: isolated Qdrant names its own (``BENCH_CORPUS_TENANT_PREFIX``): reset_store deletes by
#: tenant, so two corpora under one prefix would delete each other's vectors.
TENANT_PREFIX = os.environ.get("BENCH_CORPUS_TENANT_PREFIX") or "bench_conv"


def conversation_tenant(index: int) -> str:
    return f"{TENANT_PREFIX}_{index}"


def turn_texts(conversation: dict[str, Any]) -> dict[str, str]:
    """``dia_id -> the text a turn was submitted as``: a pure function of the dataset, so an
    arm that reuses the corpus recovers it without ingesting."""
    from benchmark.locomo import _sessions

    turns: dict[str, str] = {}
    for _, session in _sessions(conversation):
        for turn in session:
            dia_id = turn.get("dia_id")
            body = turn.get("text") or turn.get("clean_text") or ""
            if caption := (turn.get("blip_caption") or "").strip():
                query = (turn.get("query") or "").strip()
                body = (
                    f"{body} [Shared an image{f' of {query}' if query else ''}: {caption}]".strip()
                )
            if dia_id and body:
                turns[dia_id] = body
    return turns


@dataclass(frozen=True)
class CorpusKey:
    """What must be identical for two arms to share a corpus."""

    dataset_sha256: str
    index_fingerprint: str
    ingestion_sha256: str

    @staticmethod
    def ingestion_digest(settings: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(settings, sort_keys=True, default=str).encode()
        ).hexdigest()


@dataclass
class ConversationRecord:
    tenant_id: str
    turns: int
    source_ids: dict[str, str]
    ingested_at: str


class CorpusLedger:
    """The corpus a database holds, per conversation, keyed by what made it."""

    def __init__(self, database: str, *, root: Path = RESULTS / "phase7") -> None:
        self.path = root / f"corpus-{database}.json"
        self.data: dict[str, Any] = (
            json.loads(self.path.read_text())
            if self.path.is_file()
            else {"key": None, "conversations": {}}
        )

    def matches(self, key: CorpusKey) -> bool:
        return self.data.get("key") == asdict(key)

    def conversation(self, index: int) -> ConversationRecord | None:
        raw = self.data["conversations"].get(str(index))
        return ConversationRecord(**raw) if raw else None

    def adopt(self, key: CorpusKey) -> dict[str, Any]:
        """Take the held corpus as ``key``'s when only the ingestion settings differ (the same
        dataset under the same index); returns the key it held, for the artifact to show."""
        held = dict(self.data["key"])
        wanted = asdict(key)
        if {f: v for f, v in held.items() if f != "ingestion_sha256"} != {
            f: v for f, v in wanted.items() if f != "ingestion_sha256"
        }:
            raise SystemExit(f"{self.path}: only an ingestion-settings difference can be adopted")
        self.data["key"] = wanted
        self._write()
        return held

    def begin(self, key: CorpusKey) -> None:
        """A fresh corpus under ``key``; whatever the ledger said before is gone with it."""
        self.data = {"key": asdict(key), "conversations": {}}
        self._write()

    def record(self, index: int, record: ConversationRecord) -> None:
        self.data["conversations"][str(index)] = asdict(record)
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")


async def holds(container: Any, tenant_id: str) -> bool:
    """Whether the store still carries memories for ``tenant_id``: the ledger can be right
    about what was ingested and the store can still have been wiped underneath it."""
    indexer = container.services["indexer"]
    try:
        count = await container.search.count(
            indexer.collection(MEMORIES), SearchFilter(tenant_id=tenant_id)
        )
    except Exception:  # noqa: BLE001 - a missing collection is "does not hold"
        return False
    return count > 0


async def ensure_conversation(
    container: Any,
    ctx: Any,
    conversation: dict[str, Any],
    *,
    index: int,
    key: CorpusKey,
    ledger: CorpusLedger,
    reuse: bool,
) -> tuple[dict[str, str], dict[str, str], bool]:
    """The conversation's turns and its observation-id -> turn-id map, ingesting only when
    the ledger cannot vouch for what the store holds. Returns ``(turns, source_ids, reused)``."""
    from benchmark.locomo import _ingest_conversation

    known = ledger.conversation(index) if reuse and ledger.matches(key) else None
    if (
        known is not None
        and known.tenant_id == ctx.tenant_id
        and await holds(container, ctx.tenant_id)
    ):
        return turn_texts(conversation), dict(known.source_ids), True
    if not ledger.matches(key):
        ledger.begin(key)
    source_ids: dict[str, str] = {}
    turns = await _ingest_conversation(container, ctx, conversation, source_ids=source_ids)
    ledger.record(
        index,
        ConversationRecord(
            tenant_id=ctx.tenant_id,
            turns=len(turns),
            source_ids=source_ids,
            ingested_at=datetime.now(UTC).isoformat(),
        ),
    )
    return turns, source_ids, False
