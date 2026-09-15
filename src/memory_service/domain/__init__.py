"""Domain layer: contracts the Memory Service owns.

Rules (enforced by ruff banned-api + architecture tests):
- no third-party provider SDK imports (qdrant, redis, google.cloud, mem0, graphiti, ...)
- no framework types (FastAPI, LangGraph)
- typed Pydantic models at every boundary; ``dict[str, Any]`` only for custom metadata
"""

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.context_bundle import (
    ContextBundle,
    ContextItem,
    ConversationWindow,
    EvidenceReport,
    UnusedEvidence,
)
from memory_service.domain.conversation import (
    AgentRun,
    Attachment,
    Message,
    Session,
    Thread,
    Turn,
)
from memory_service.domain.documents import (
    Chunk,
    ContextEdge,
    Document,
    DocumentNode,
    DocumentVersion,
)
from memory_service.domain.enums import (
    ArchiveStatus,
    ContextGraphEdge,
    DedupDecision,
    ErrorCode,
    EvidenceStatus,
    JobStatus,
    Lifetime,
    MemoryType,
    MessageKind,
    MessageRole,
    ObservationKind,
    QueryType,
    Representation,
    ScopeLevel,
    TemporalStatus,
    Visibility,
)
from memory_service.domain.evidence import EvidenceGroup, EvidenceRef
from memory_service.domain.grounding import ClaimReport, ClaimVerdict, GroundingReport
from memory_service.domain.memory import CanonicalMemory, MemoryResult, Scope, TemporalState
from memory_service.domain.observation import Observation, ProcessingHints

__all__ = [
    "AgentRun",
    "ArchiveStatus",
    "Attachment",
    "CanonicalMemory",
    "Chunk",
    "ClaimReport",
    "ClaimVerdict",
    "ContextBundle",
    "ContextEdge",
    "ContextGraphEdge",
    "ContextItem",
    "ConversationWindow",
    "DedupDecision",
    "Document",
    "DocumentNode",
    "DocumentVersion",
    "ErrorCode",
    "EvidenceGroup",
    "EvidenceRef",
    "EvidenceReport",
    "EvidenceStatus",
    "GroundingReport",
    "JobStatus",
    "Lifetime",
    "MemoryExecutionContext",
    "MemoryResult",
    "MemoryType",
    "Message",
    "MessageKind",
    "MessageRole",
    "Observation",
    "ObservationKind",
    "ProcessingHints",
    "QueryType",
    "Representation",
    "Scope",
    "ScopeLevel",
    "Session",
    "TemporalState",
    "TemporalStatus",
    "Thread",
    "Turn",
    "UnusedEvidence",
    "Visibility",
]
