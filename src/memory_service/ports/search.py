"""SearchStore port. Qdrant by default; rebuildable from canonical PostgreSQL data.

The port speaks in terms of *records* with dense/sparse vectors and filterable payloads.
Fusion (RRF) is performed by the store when it supports it natively; the fallback is a
bounded client-side RRF over per-retriever candidate lists.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class SparseVector(BaseModel):
    """Term-id -> weight. BM25 term frequencies with server-side IDF, or SPLADE weights."""

    model_config = ConfigDict(frozen=True)

    indices: list[int]
    values: list[float]


class SearchRecord(BaseModel):
    """One indexable object. ``payload`` holds only filterable/security metadata + display text."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    collection: str
    tenant_id: str
    dense: list[float] | None = None
    sparse: SparseVector | None = None
    late_interaction: list[list[float]] | None = Field(
        default=None, description="ColBERT multivectors"
    )
    payload: dict[str, Any] = Field(default_factory=dict)


class SearchFilter(BaseModel):
    """Scope filter applied *inside* the store, before any candidate is returned.

    ``must`` entries are equality matches; ``must_any`` entries match any value of a list.
    ``tenant_id`` is mandatory so no query can be issued without a tenant boundary.
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    must: dict[str, str | int | bool] = Field(default_factory=dict)
    must_any: dict[str, list[str]] = Field(default_factory=dict)
    must_not: dict[str, str | int | bool] = Field(default_factory=dict)


class SearchHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    record_id: str
    score: float
    retriever: Literal["dense", "sparse", "bm25", "late_interaction", "exact", "fusion"]
    payload: dict[str, Any] = Field(default_factory=dict)


class CollectionSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    dense_dim: int | None = None
    sparse: bool = True
    sparse_idf: bool = Field(default=True, description="server-side IDF modifier (BM25)")
    late_interaction_dim: int | None = None
    on_disk: bool = False


@runtime_checkable
class SearchStore(Protocol):
    async def ensure_collection(self, spec: CollectionSpec) -> None: ...

    async def upsert(self, records: Sequence[SearchRecord]) -> None: ...

    async def delete(self, collection: str, record_ids: Sequence[str]) -> None: ...

    async def delete_by_filter(self, collection: str, flt: SearchFilter) -> int: ...

    async def search_dense(
        self, collection: str, vector: Sequence[float], flt: SearchFilter, *, limit: int
    ) -> list[SearchHit]: ...

    async def search_sparse(
        self, collection: str, vector: SparseVector, flt: SearchFilter, *, limit: int
    ) -> list[SearchHit]: ...

    async def search_hybrid(
        self,
        collection: str,
        *,
        dense: Sequence[float] | None,
        sparse: SparseVector | None,
        flt: SearchFilter,
        limit: int,
        prefetch_limit: int,
    ) -> list[SearchHit]:
        """Native RRF fusion when supported; otherwise client-side RRF over bounded prefetch."""
        ...

    async def get(self, collection: str, record_ids: Sequence[str]) -> list[SearchRecord]: ...

    async def count(self, collection: str, flt: SearchFilter) -> int: ...

    async def ping(self) -> bool: ...

    async def close(self) -> None: ...
