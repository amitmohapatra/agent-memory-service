"""Who may read a thread's searchable episode: the exact audience key it is indexed with."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from memory_service.domain.conversation import Thread
from memory_service.modules.rag.indexer import Indexer
from memory_service.ports.search import SearchRecord, SparseVector

WHEN = datetime(2026, 9, 30, tzinfo=UTC)


class _Store:
    def __init__(self) -> None:
        self.upserted: list[SearchRecord] = []

    async def upsert(self, records: list[SearchRecord]) -> None:
        self.upserted.extend(records)


class _Sparse:
    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1], values=[1.0]) for _ in texts]


class _Indexer(Indexer):
    def __init__(self, thread: Thread) -> None:
        self.store: Any = _Store()
        self.sparse: Any = _Sparse()
        self.late = None
        self._thread = thread

    def collection(self, base: str) -> str:
        return base

    async def ensure_collections(self) -> None:
        return None

    async def embed_cached(self, texts: Any, hashes: Any) -> dict[Any, list[list[float]]]:
        return {}

    async def _episode_source(self, tenant_id: str, thread_id: str) -> Any:
        return self._thread, "we agreed on 137 units", WHEN, WHEN


@pytest.mark.parametrize(
    ("owner", "keys"),
    [
        ("ann", ["user:acme/ann"]),
        (None, ["thread:acme/thr_1"]),
        # an owner that is no owner keeps the thread's own audience
        ("", ["thread:acme/thr_1"]),
    ],
)
async def test_an_episode_is_read_by_its_threads_owner_or_inside_the_thread(
    owner: str | None, keys: list[str]
) -> None:
    indexer = _Indexer(Thread(thread_id="thr_1", tenant_id="acme", owner_user_id=owner))
    assert await indexer.index_episode("acme", "thr_1") is True
    [record] = indexer.store.upserted
    assert record.payload["visibility_keys"] == keys
