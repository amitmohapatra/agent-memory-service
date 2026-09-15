"""Closed vocabularies used across the Memory Service domain.

Memory classification is deliberately multi-dimensional (lifetime, type, scope,
visibility, representation, temporal state, ...). No single enum describes a memory.
"""

from __future__ import annotations

from enum import StrEnum


class Lifetime(StrEnum):
    """How long a memory is expected to matter."""

    EPHEMERAL = "EPHEMERAL"  # Dragonfly TTL only
    SHORT_TERM = "SHORT_TERM"  # current session/task/active context
    LONG_TERM = "LONG_TERM"  # reusable durable intelligence
    ARCHIVAL = "ARCHIVAL"  # raw history/evidence and superseded versions


class MemoryType(StrEnum):
    """What kind of intelligence a memory carries. Custom plugin types use ``CUSTOM``."""

    WORKING = "WORKING"
    CONVERSATION = "CONVERSATION"
    EPISODIC = "EPISODIC"
    SEMANTIC = "SEMANTIC"
    PROCEDURAL = "PROCEDURAL"
    USER = "USER"
    PREFERENCE = "PREFERENCE"
    AGENT = "AGENT"
    SHARED = "SHARED"
    TASK = "TASK"
    WORK = "WORK"
    TOOL = "TOOL"
    SKILL = "SKILL"
    DECISION = "DECISION"
    FAILURE = "FAILURE"
    OUTCOME = "OUTCOME"
    ARTIFACT = "ARTIFACT"
    KNOWLEDGE_RAG = "KNOWLEDGE_RAG"
    SUMMARY = "SUMMARY"
    DERIVED = "DERIVED"
    POLICY = "POLICY"
    OBSERVATION = "OBSERVATION"  # dated, thread-scoped notes compressed from older turns
    BELIEF = "BELIEF"  # derived generalisation with support and a revision chain
    ENTITY_SUMMARY = "ENTITY_SUMMARY"  # one maintained summary per entity (subject)
    CUSTOM = "CUSTOM"


class Visibility(StrEnum):
    """Who may see a memory. Owner and visibility are separate concepts."""

    PRIVATE = "PRIVATE"  # only the owning principal (user or agent)
    USER = "USER"  # the owning user across threads
    GROUP = "GROUP"  # a user group
    AGENT_GROUP = "AGENT_GROUP"  # a group of cooperating agents
    RUN = "RUN"  # this agent run and the runs it spawns (hand-off context flows down)
    THREAD = "THREAD"  # everyone participating in the thread
    WORK = "WORK"  # everyone participating in a unit of work
    WORKSPACE = "WORKSPACE"
    TENANT = "TENANT"
    GLOBAL = "GLOBAL"


class ScopeLevel(StrEnum):
    """Where a memory is anchored. Distinct from visibility."""

    AGENT = "AGENT"
    AGENT_GROUP = "AGENT_GROUP"
    WORK = "WORK"
    THREAD = "THREAD"
    USER = "USER"
    GROUP = "GROUP"
    WORKSPACE = "WORKSPACE"
    TENANT = "TENANT"
    GLOBAL = "GLOBAL"


class Representation(StrEnum):
    """Which representation of knowledge an object is."""

    RAW_FILE = "RAW_FILE"
    DOCUMENT = "DOCUMENT"
    DOCUMENT_VERSION = "DOCUMENT_VERSION"
    SECTION = "SECTION"
    SUBSECTION = "SUBSECTION"
    PARAGRAPH = "PARAGRAPH"
    TABLE = "TABLE"
    CODE_BLOCK = "CODE_BLOCK"
    CHUNK = "CHUNK"
    SUMMARY = "SUMMARY"
    ENTITY = "ENTITY"
    RELATION = "RELATION"
    EMBEDDING = "EMBEDDING"
    MESSAGE = "MESSAGE"
    MEMORY = "MEMORY"


class TemporalStatus(StrEnum):
    """Validity state of a fact/memory in time."""

    CURRENT = "CURRENT"
    SUPERSEDED = "SUPERSEDED"
    EXPIRED = "EXPIRED"
    CONTRADICTED = "CONTRADICTED"
    RETRACTED = "RETRACTED"
    ARCHIVED = "ARCHIVED"  # forgotten by policy: kept, restorable, out of retrieval


class AdmissionVerdict(StrEnum):
    """What the admission gate decided for a memory candidate."""

    ADMIT = "ADMIT"
    REJECT = "REJECT"
    DEFER = "DEFER"  # parked in working memory until corroborated


class MessageRole(StrEnum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"
    TOOL = "TOOL"
    AGENT = "AGENT"


class MessageKind(StrEnum):
    """Visible UI history contains only VISIBLE messages; internal execution is INTERNAL."""

    VISIBLE = "VISIBLE"
    INTERNAL = "INTERNAL"


class ObservationKind(StrEnum):
    """What an application is telling the Memory Service happened."""

    MESSAGE = "MESSAGE"
    FILE = "FILE"
    AGENT_RESULT = "AGENT_RESULT"
    TOOL_RESULT = "TOOL_RESULT"
    DECISION = "DECISION"
    FEEDBACK = "FEEDBACK"
    EVENT = "EVENT"
    IMPORT = "IMPORT"


class QueryType(StrEnum):
    """Deterministic QueryRouter categories."""

    EXACT_IDENTIFIER = "EXACT_IDENTIFIER"
    CONVERSATION_HISTORY = "CONVERSATION_HISTORY"
    USER_MEMORY = "USER_MEMORY"
    DECISION = "DECISION"
    DOCUMENT_LOCAL = "DOCUMENT_LOCAL"
    DOCUMENT_MULTI_HOP = "DOCUMENT_MULTI_HOP"
    ENTITY_RELATION = "ENTITY_RELATION"
    TEMPORAL = "TEMPORAL"
    GLOBAL_SUMMARY = "GLOBAL_SUMMARY"
    GENERAL_SEMANTIC = "GENERAL_SEMANTIC"


class DedupDecision(StrEnum):
    CREATE = "CREATE"
    REINFORCE = "REINFORCE"
    MERGE = "MERGE"
    UPDATE = "UPDATE"
    SUPERSEDE = "SUPERSEDE"
    CONTRADICT = "CONTRADICT"
    IGNORE = "IGNORE"


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    CANCELLED = "CANCELLED"


class ArchiveStatus(StrEnum):
    STAGED = "STAGED"  # durable in PostgreSQL, not yet in blob store
    ARCHIVING = "ARCHIVING"
    ARCHIVED = "ARCHIVED"  # blob written, checksum + generation verified, manifest persisted
    PURGED = "PURGED"  # large staged payload removed from hot DB after grace period


class EvidenceStatus(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    INSUFFICIENT = "INSUFFICIENT"


class ContextGraphEdge(StrEnum):
    """Deterministic structural links of the Document Context Graph."""

    PARENT = "PARENT"
    CHILD = "CHILD"
    PREVIOUS = "PREVIOUS"
    NEXT = "NEXT"
    ON_PAGE = "ON_PAGE"
    IN_TABLE = "IN_TABLE"
    FOOTNOTE = "FOOTNOTE"
    CROSS_REFERENCE = "CROSS_REFERENCE"
    MENTIONS = "MENTIONS"
    DEFINED_BY = "DEFINED_BY"
    DEFINES = "DEFINES"


class ErrorCode(StrEnum):
    """Public API error categories (see ``api/errors.py`` for the envelope)."""

    VALIDATION = "VALIDATION"
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    SCOPE_DENIED = "SCOPE_DENIED"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    RATE_LIMIT = "RATE_LIMIT"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    RETRYABLE_PROCESSING = "RETRYABLE_PROCESSING"
    CORRUPT_SOURCE = "CORRUPT_SOURCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INTERNAL = "INTERNAL"
