"""universal-memory: Python SDK for the Enterprise Multi-Agent Memory Service."""

from universal_memory.client import ChatAPI, FilesAPI, MemoryClient, MemoryContext, current_context
from universal_memory.errors import (
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    DependencyUnavailableError,
    InsufficientEvidence,
    MemoryError,
    NotFoundError,
    RateLimitedError,
    ValidationError,
)
from universal_memory.models import (
    ContextBundle,
    ContextItem,
    EvidenceRef,
    FileHandle,
    JobHandle,
    MemoryResult,
    MessageAck,
    MessageInfo,
    ObservationAck,
    Scope,
    ThreadInfo,
)

__version__ = "0.1.0"

__all__ = [
    "AuthenticationError",
    "AuthorizationError",
    "ChatAPI",
    "ConflictError",
    "ContextBundle",
    "ContextItem",
    "DependencyUnavailableError",
    "EvidenceRef",
    "FileHandle",
    "FilesAPI",
    "InsufficientEvidence",
    "JobHandle",
    "MemoryClient",
    "MemoryContext",
    "MemoryError",
    "MemoryResult",
    "MessageAck",
    "MessageInfo",
    "NotFoundError",
    "ObservationAck",
    "RateLimitedError",
    "Scope",
    "ThreadInfo",
    "ValidationError",
    "current_context",
]
