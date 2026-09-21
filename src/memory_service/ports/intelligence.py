"""Memory-intelligence, graph, document-parser and policy ports.

``MemoryIntelligenceProvider`` is the seam behind which Native / Mem0 / Cognee / LangMem
sit. None of them is the public contract; they produce candidates that the service turns
into canonical memories with provenance.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.documents import Chunk, ContextEdge, DocumentNode, DocumentVersion
from memory_service.domain.enums import DedupDecision, Lifetime, MemoryType, Visibility
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.graph import GraphLayer, layer_for
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
    #: Only meaningful with ``memory_type=CUSTOM``, which carries a caller-defined taxonomy
    #: and is rejected by CanonicalMemory without it. See ProcessingHints.custom_type.
    custom_type: str | None = None
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
    visibility_keys: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    mention_count: int = 1
    revision: int = 1
    summary: str = Field(
        default="",
        description="maintained one-paragraph summary of the entity's current facts",
    )


class Relation(BaseModel):
    """A bitemporal fact: subject -predicate-> object, true over [valid_from, valid_to)
    (valid time) and known from ``observed_at`` until ``invalidated_at`` (knowledge time).

    A relation is never deleted: a replaced or withdrawn fact keeps its row, gets a status
    (SUPERSEDED when it stopped being true, INVALIDATED when it was never right) and an
    ``invalidated_by`` edge that names the winning fact and the reason.
    """

    model_config = ConfigDict(extra="forbid")

    relation_id: str
    tenant_id: str
    subject_id: str
    predicate: str
    object_id: str
    layer: GraphLayer = Field(
        default="entity", description="entity | temporal | causal | structural (from predicate)"
    )
    scope_key: str = ""
    visibility_keys: list[str] = Field(default_factory=list)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    observed_at: datetime
    invalidated_at: datetime | None = Field(
        default=None, description="knowledge time at which the fact stopped being asserted"
    )
    status: str = "CURRENT"
    superseded_by: str | None = None
    confidence: float = 0.5
    evidence: list[EvidenceRef] = Field(..., min_length=1)
    memory_id: str | None = None
    document_id: str | None = None
    fact_text: str = Field(default="", description="human-readable statement of the fact")
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _layer_from_predicate(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("layer"):
            data = {**data, "layer": layer_for(str(data.get("predicate", "")))}
        return data


class EntityAlias(BaseModel):
    """Tenant-wide alias row: ``alias`` (canonical form) names ``entity_id`` with a
    confidence and the source that established it (canonical, lexicon, rule:acronym,
    rule:plural, rule:punctuation, embedding, llm)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    alias: str
    entity_id: str
    confidence: float = Field(default=1.0, ge=0, le=1)
    source: str = "canonical"


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

    async def list_entities(
        self, tenant_id: str, *, scope_keys: Sequence[str], limit: int = 200
    ) -> list[Entity]:
        """Bounded listing of visible entities, most mentioned first."""
        ...

    async def neighborhood(
        self,
        tenant_id: str,
        entity_ids: Sequence[str],
        *,
        scope_keys: Sequence[str],
        hops: int = 1,
        max_visited: int = 200,
        as_of: datetime | None = None,
        valid_at: datetime | None = None,
        layers: Sequence[GraphLayer] | None = None,
    ) -> GraphNeighborhood:
        """Bounded traversal: O(V+E) over at most ``max_visited`` nodes.

        ``as_of`` selects facts *true* at that instant (valid time); ``valid_at`` selects
        facts *asserted* by then and not yet invalidated (knowledge time); ``layers``
        restricts the walk to those layers.
        """
        ...

    async def supersede(self, relation_id: str, *, by: str, at: datetime) -> None: ...

    async def supersede_for_memory(self, tenant_id: str, memory_id: str, *, at: datetime) -> int:
        """Retire every CURRENT relation derived from a memory that is no longer current."""
        ...

    async def invalidate(
        self,
        relation_id: str,
        *,
        reason: str,
        at: datetime,
        by: str | None = None,
        status: str = "INVALIDATED",
        attributes: dict[str, Any] | None = None,
    ) -> Relation | None:
        """Close a relation without deleting it and record why: the row keeps its history,
        gets ``status`` and ``invalidated_at``, and — when ``by`` names the winning fact —
        an ``invalidated_by`` edge (relation -> relation) carries ``reason``, the winner,
        the loser and ``attributes`` (which fact won and why). Returns the edge."""
        ...

    async def delete_for_document(self, tenant_id: str, document_id: str) -> int: ...

    async def relations_for_document(
        self,
        tenant_id: str,
        document_id: str,
        *,
        scope_keys: Sequence[str],
        include_invalidated: bool = False,
    ) -> list[Relation]:
        """Every visible relation extracted from a document (audits, evals, exports)."""
        ...

    async def get_entities(
        self, tenant_id: str, entity_ids: Sequence[str], *, scope_keys: Sequence[str]
    ) -> list[Entity]: ...

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
