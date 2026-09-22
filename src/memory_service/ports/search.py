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


Retriever = Literal["dense", "sparse", "bm25", "exact", "fusion"]


class SearchHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    record_id: str
    score: float
    retriever: Retriever
    payload: dict[str, Any] = Field(default_factory=dict)


class CollectionSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    dense_dim: int | None = None
    sparse: bool = True
    sparse_idf: bool = Field(default=True, description="server-side IDF modifier (BM25)")
    on_disk: bool = False
    on_disk_payload: bool = Field(
        default=True,
        description="keep payloads on disk; False for a collection small enough to hold in RAM",
    )


#: The payload keys a reader is allowed to rely on, and therefore the only ones a store has
#: to return. Every key here is read somewhere in the retrieval path (the retrieval engine,
#: the context builder, evidence, expansion, the graph stage) or by the store itself; a
#: unit test re-derives the set from those files and fails when the two drift apart.
#:
#: What is *not* here matters as much: a hit used to arrive with its whole payload, so the
#: security metadata a filter had already applied inside the store (tenant_id aside) and the
#: indexing bookkeeping travelled back over the wire and were parsed under the GIL for every
#: candidate of every query.
PAYLOAD_FIELDS: tuple[str, ...] = (
    "attributes",
    "chunk_id",
    "confidence",
    "contradicts",
    "contributors",
    "document_id",
    "kind",
    "memory_type",
    "node_id",
    "object",
    "observed_at",
    "owner_principal",
    "page",
    "predicate",
    "record_id",
    "section_path",
    "status",
    "subject",
    "tenant_id",
    "text",
    "text_hash",
    "visibility",
)


@runtime_checkable
class SearchStore(Protocol):
    async def ensure_collection(self, spec: CollectionSpec) -> None: ...

    async def upsert(self, records: Sequence[SearchRecord]) -> None: ...

    async def delete(self, collection: str, record_ids: Sequence[str]) -> None: ...

    async def delete_by_filter(self, collection: str, flt: SearchFilter) -> int: ...

    async def record_ids(self, collection: str, flt: SearchFilter) -> list[str]:
        """Every record id matching ``flt``. Used to find index entries whose source row is
        gone, which is the only way to notice a superseded generation of a document."""
        ...

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

    async def drop_collection(self, collection: str) -> bool:
        """Delete a whole collection (index loss / rebuild). Returns True if it existed."""
        ...

    async def ping(self) -> bool: ...

    async def close(self) -> None: ...
