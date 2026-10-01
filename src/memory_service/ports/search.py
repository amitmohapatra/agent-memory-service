"""SearchStore port. Qdrant by default; rebuildable from canonical PostgreSQL data.

The port speaks in terms of *records* with named dense vectors, a sparse vector and
filterable payloads. A collection carries one dense vector per *space* (``VectorName``): the
English specialist and the multilingual encoder each own a space, and a query searches the
spaces its script calls for. Fusion (RRF) is performed by the store when it supports it
natively; the fallback is a bounded client-side RRF over per-retriever candidate lists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class VectorName(StrEnum):
    """The named vectors a collection carries: the names on the wire."""

    #: the English specialist encoder (Granite); searched for Latin-script queries only
    DENSE_EN = "dense_en"
    #: the multilingual encoder (Bekko); searched for every query
    DENSE_ML = "dense_ml"
    #: client-side BM25 term frequencies with the store's IDF modifier
    BM25 = "bm25"


class Retriever(StrEnum):
    """Which arm produced a hit. The vector arms share their wire names."""

    DENSE_EN = "dense_en"
    DENSE_ML = "dense_ml"
    BM25 = "bm25"
    EXACT = "exact"
    FUSION = "fusion"

    @classmethod
    def for_vector(cls, name: VectorName | str) -> Retriever:
        return cls(str(name))


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
    #: one vector per dense space the collection carries, keyed by its wire name
    dense: dict[VectorName, list[float]] = Field(default_factory=dict)
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
    #: field -> (from, to) instants, either bound open; a record without the field is out
    within: dict[str, tuple[datetime | None, datetime | None]] = Field(default_factory=dict)


class SearchHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    record_id: str
    score: float
    retriever: Retriever
    payload: dict[str, Any] = Field(default_factory=dict)


class CollectionSpec(BaseModel):
    #: ``extra="forbid"`` because this spec decides what a collection physically is. When
    #: ``dense_dim: int`` became ``dense: dict[VectorName, int]``, two callers kept passing
    #: the old name and pydantic dropped it in silence: they went on asking for a collection
    #: with no dense vector at all and still read as if they had asked for a 384-wide one. A
    #: field this port removes must fail loudly at the call site, not become a default.
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    #: the dense spaces and their widths
    dense: dict[VectorName, int] = Field(default_factory=dict)
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
#: candidate of every query. ``script`` and ``entities`` are filter fields, indexed and
#: matched inside the store, and never come back.
#:
#: Declared, because the plan said otherwise: ``contributors``, ``owner_principal``,
#: ``visibility`` and ``status`` are projected, and the Phase 2 plan listed them among the
#: fields to never return. They are kept because they are *read* - the context builder
#: copies them into ``ContextItem.attributes``, so they are part of the API response today
#: and have been since before this projection existed. Dropping them here would not be a
#: wire optimisation, it would be an API change made silently, and the field these three
#: protect (visibility) is enforced by the store-side filter, not by what comes back. The
#: cost is four short scalars per hit. Removing them is a decision about the response
#: schema and belongs where that is versioned, not here.
PAYLOAD_FIELDS: tuple[str, ...] = (
    "category",
    "provider",
    "attributes",
    "chunk_id",
    "confidence",
    "connections",
    "contradicts",
    "contributors",
    "current",
    "dated_mentions",
    "derived",
    "document_id",
    "kind",
    "memory_type",
    "node_id",
    "object",
    "observed_at",
    "owner_principal",
    "page",
    "preceding_source_id",
    "predicate",
    "record_id",
    "reinforcement",
    "section_path",
    "source_refs",
    "source_observed_to",
    "status",
    "subject",
    "tenant_id",
    "text",
    "text_hash",
    "thread_id",
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
        self,
        collection: str,
        name: VectorName,
        vector: Sequence[float],
        flt: SearchFilter,
        *,
        limit: int,
    ) -> list[SearchHit]: ...

    async def search_sparse(
        self, collection: str, vector: SparseVector, flt: SearchFilter, *, limit: int
    ) -> list[SearchHit]: ...

    async def similarity(
        self,
        collection: str,
        name: VectorName,
        vector: Sequence[float],
        record_ids: Sequence[str],
    ) -> dict[str, float]:
        """The dense similarity of each named record to ``vector`` in space ``name`` (cosine,
        so an absolute number, unlike a fusion score). Records without that vector are
        absent. ``record_ids`` are already authorized: no filter beyond the ids."""
        ...

    async def search_hybrid(
        self,
        collection: str,
        *,
        dense: Mapping[VectorName, Sequence[float]],
        sparse: SparseVector | None,
        flt: SearchFilter,
        limit: int,
        prefetch_limit: int,
        rrf_k: int = 1,
        weights: Mapping[VectorName, float] | None = None,
    ) -> list[SearchHit]:
        """Bounded hybrid fusion over every dense space given plus the sparse arm, each arm
        scoring ``weight / (rrf_k + one-based rank)``; a missing weight is 1.0. Hits carry
        ``PAYLOAD_FIELDS`` (a two-phase read was measured slower: MEASUREMENTS.md 8.7)."""
        ...

    async def get(self, collection: str, record_ids: Sequence[str]) -> list[SearchRecord]: ...

    async def count(self, collection: str, flt: SearchFilter) -> int: ...

    async def list_collections(self) -> list[str]:
        """Every collection this deployment owns, unprefixed.

        A collection's name carries the embedding and sparse fingerprints, so changing a
        model does not migrate the old one - it creates a new one beside it and leaves the
        previous generation holding points nothing will ever read. Measured on the dev
        store: eighteen such collections. Something has to be able to see them to remove
        them.
        """
        ...

    async def drop_collection(self, collection: str) -> bool:
        """Delete a whole collection (index loss / rebuild). Returns True if it existed."""
        ...

    async def ping(self) -> bool: ...

    async def close(self) -> None: ...
