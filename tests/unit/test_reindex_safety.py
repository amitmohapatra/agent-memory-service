"""The reindex tool must not be able to destroy what it was not asked to rebuild.

Two separate hazards live in one function. A collection is shared by every tenant, and the
rebuild is filtered by tenant - so dropping the collection while rebuilding one tenant
removed every other tenant's vectors and then reported success, which was documented as an
instruction. And a collection's name carries the model fingerprints, so changing a model
leaves the previous generation behind holding points nothing will ever query: eighteen had
accumulated on the development store with no way to see or remove them.
"""

from __future__ import annotations

from typing import Any

import pytest

from memory_service.modules.rag.indexer import MEMORIES
from memory_service.ports.search import SearchFilter
from memory_service.tools.reindex import prune_retired_collections, rebuild_search_index

pytestmark = pytest.mark.unit

LIVE_MEMORIES = "memories_st-model-torch-d384__bm25-v1"
LIVE_KNOWLEDGE = "knowledge_st-model-torch-d384__bm25-v1"


class _Search:
    def __init__(self, collections: list[str]) -> None:
        self.collections = collections
        self.dropped: list[str] = []
        self.deleted: list[tuple[str, str | None]] = []

    async def list_collections(self) -> list[str]:
        return list(self.collections)

    async def drop_collection(self, name: str) -> bool:
        self.dropped.append(name)
        return True

    async def delete_by_filter(self, name: str, flt: SearchFilter) -> int:
        self.deleted.append((name, flt.tenant_id))
        return 7


class _Indexer:
    def collection(self, base: str) -> str:
        return LIVE_MEMORIES if base == MEMORIES else LIVE_KNOWLEDGE

    async def ensure_collections(self) -> None:
        return None


class _Container:
    def __init__(self, search: _Search) -> None:
        self.search = search
        self.services: dict[str, Any] = {"indexer": _Indexer()}
        self.database = None


async def _rebuild(container: _Container, **kwargs: Any) -> Any:
    """``rebuild_search_index`` past the drop, without a database to read from."""
    with pytest.raises((AttributeError, TypeError)):
        await rebuild_search_index(container, **kwargs)  # type: ignore[arg-type]


async def test_a_tenant_scoped_rebuild_never_drops_a_shared_collection() -> None:
    search = _Search([LIVE_MEMORIES, LIVE_KNOWLEDGE])
    await _rebuild(_Container(search), tenant_id="acme", drop=True)
    assert search.dropped == [], "dropping the collection destroys every other tenant"
    assert {name for name, _ in search.deleted} == {LIVE_MEMORIES, LIVE_KNOWLEDGE}
    assert {tenant for _, tenant in search.deleted} == {"acme"}


async def test_a_whole_store_rebuild_may_still_drop() -> None:
    search = _Search([LIVE_MEMORIES, LIVE_KNOWLEDGE])
    await _rebuild(_Container(search), tenant_id=None, drop=True)
    assert set(search.dropped) == {LIVE_MEMORIES, LIVE_KNOWLEDGE}
    assert search.deleted == []


async def test_pruning_reports_retired_collections_and_keeps_the_live_ones() -> None:
    retired = ["memories_st-old-torch-d768__bm25-v1", "knowledge_hash-v1-d64__bm25-v1"]
    search = _Search([LIVE_MEMORIES, LIVE_KNOWLEDGE, *retired])
    found = await prune_retired_collections(_Container(search), dry_run=True)  # type: ignore[arg-type]
    assert found == sorted(retired)
    assert search.dropped == [], "a dry run changes nothing"


async def test_pruning_drops_only_what_the_current_fingerprint_does_not_name() -> None:
    retired = ["memories_st-old-torch-d768__bm25-v1"]
    search = _Search([LIVE_MEMORIES, LIVE_KNOWLEDGE, *retired])
    await prune_retired_collections(_Container(search), dry_run=False)  # type: ignore[arg-type]
    assert search.dropped == retired
    assert LIVE_MEMORIES not in search.dropped and LIVE_KNOWLEDGE not in search.dropped


async def test_an_empty_store_prunes_nothing() -> None:
    search = _Search([])
    assert await prune_retired_collections(_Container(search), dry_run=False) == []  # type: ignore[arg-type]
    assert search.dropped == []
