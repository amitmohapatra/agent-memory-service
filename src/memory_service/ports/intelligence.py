"""Memory-intelligence, graph, document-parser and policy ports.

``MemoryIntelligenceProvider`` is the seam behind which Native / Mem0 / Cognee / LangMem
sit. None of them is the public contract; they produce candidates that the service turns
into canonical memories with provenance.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.documents import Chunk, ContextEdge, DocumentNode, DocumentVersion
from memory_service.domain.enums import DedupDecision, Lifetime, MemoryType, Visibility
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.observation import Observation
from memory_service.ports.models import ProviderInfo

# --------------------------------------------------------------------------
# Memory intelligence
# --------------------------------------------------------------------------


class MemoryCandidate(BaseModel):
    """A possible memory extracted from an observation; not yet canonical."""

    model_config = ConfigDict(frozen=True)

    content: str
    memory_type: MemoryType
    lifetime: Lifetime
    visibility: Visibility | None = None
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)
    importance: float = Field(default=0.5, ge=0, le=1)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    provider: str = "native"
    negates_prior: bool = Field(
        default=False,
        description="the statement explicitly replaces an earlier one (no longer / now / instead)",
    )
    category: str | None = Field(
        default=None, description="finer label than memory_type, e.g. decision, attribute"
    )
    provider_ref: str | None = Field(
        default=None, description="the external provider's own id for this memory, if any"
    )


class ConsolidationOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: DedupDecision
    candidate: MemoryCandidate
    target_memory_id: str | None = Field(
        default=None, description="existing memory affected, if any"
    )
    score: float = 0.0
    reason: str = ""


@runtime_checkable
class MemoryIntelligenceProvider(Protocol):
    info: ProviderInfo

    async def extract(
        self, observation: Observation, ctx: MemoryExecutionContext
    ) -> list[MemoryCandidate]: ...

    async def classify(
        self, candidate: MemoryCandidate, ctx: MemoryExecutionContext
    ) -> MemoryCandidate:
        """Refine lifetime/type/visibility/importance."""
        ...

    async def consolidate(
        self,
        candidate: MemoryCandidate,
        existing: Sequence[CanonicalMemory],
        ctx: MemoryExecutionContext,
    ) -> ConsolidationOutcome: ...

    async def search_features(self, query: str, ctx: MemoryExecutionContext) -> dict[str, Any]:
        """Provider-specific retrieval hints (e.g. Mem0 filters). May return {}."""
        ...


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str
    tenant_id: str
    name: str
    canonical_name: str
    entity_type: str = "THING"
    aliases: list[str] = Field(default_factory=list)
    scope_key: str = ""
    evidence: list[EvidenceRef] = Field(default_factory=list)
    revision: int = 1


class Relation(BaseModel):
    """A temporal fact: subject -predicate-> object, valid over [valid_from, valid_to)."""

    model_config = ConfigDict(extra="forbid")

    relation_id: str
    tenant_id: str
    subject_id: str
    predicate: str
    object_id: str
    scope_key: str = ""
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    observed_at: datetime
    status: str = "CURRENT"
    confidence: float = 0.5
    evidence: list[EvidenceRef] = Field(..., min_length=1)
    memory_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class GraphNeighborhood(BaseModel):
    model_config = ConfigDict(frozen=True)

    entities: list[Entity]
    relations: list[Relation]
    visited: int


@runtime_checkable
class GraphStore(Protocol):
    async def upsert_entities(self, entities: Sequence[Entity]) -> None: ...

    async def upsert_relations(self, relations: Sequence[Relation]) -> None: ...

    async def find_entities(
        self, tenant_id: str, names: Sequence[str], *, scope_keys: Sequence[str]
    ) -> list[Entity]: ...

    async def neighborhood(
        self,
        tenant_id: str,
        entity_ids: Sequence[str],
        *,
        scope_keys: Sequence[str],
        hops: int = 1,
        max_visited: int = 200,
        as_of: datetime | None = None,
    ) -> GraphNeighborhood:
        """Bounded traversal: O(V+E) over at most ``max_visited`` nodes."""
        ...

    async def supersede(self, relation_id: str, *, by: str, at: datetime) -> None: ...

    async def ping(self) -> bool: ...


@runtime_checkable
class GraphEnrichmentProvider(Protocol):
    """Native (LLM-free) | Graphiti | DoclingGraph | Cognee."""

    info: ProviderInfo

    async def enrich_memory(
        self, memory: CanonicalMemory, ctx: MemoryExecutionContext
    ) -> tuple[list[Entity], list[Relation]]: ...

    async def enrich_document(
        self,
        version: DocumentVersion,
        nodes: Sequence[DocumentNode],
        chunks: Sequence[Chunk],
        ctx: MemoryExecutionContext,
    ) -> tuple[list[Entity], list[Relation]]: ...


# --------------------------------------------------------------------------
# Document parsing
# --------------------------------------------------------------------------


class ParsedDocument(BaseModel):
    """Output of a DocumentParser: hierarchy + deterministic structural edges."""

    model_config = ConfigDict(frozen=True)

    version: DocumentVersion
    nodes: list[DocumentNode]
    edges: list[ContextEdge]
    title: str
    page_count: int | None = None


@runtime_checkable
class DocumentParser(Protocol):
    info: ProviderInfo
    supported_media_types: frozenset[str]

    async def parse(
        self,
        *,
        document_id: str,
        tenant_id: str,
        filename: str,
        media_type: str,
        data: bytes,
    ) -> ParsedDocument: ...


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


class PolicyDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: bool
    reason: str = ""
    obligations: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class PolicyProvider(Protocol):
    """OPA optional. Answers 'is this operation permitted under policy?'."""

    async def evaluate(self, policy: str, input_data: dict[str, Any]) -> PolicyDecision: ...
