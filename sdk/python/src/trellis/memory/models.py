"""SDK-facing models. Mirrors the public API contract; no internal types leak here.

The closed vocabularies below are the API's enums spelled as Literals, so a wrong value is
a type error in the caller's editor and a 422 from the service, never a silent no-op. They
are kept in step with the service by a test in the service repository.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentKeyStatus(BaseModel):
    """Metadata only; neither a virtual key nor encrypted material is returned."""

    registered: bool
    revoked: bool
    revision: int
    updated_at: datetime | None = None


class ModelPolicy(BaseModel):
    """The tenant's model policy: the uses it allows, whether reads are assisted, the model a
    use calls. ``stored=False``: none is set, so the default applies (every use, reads
    assisted, the service's model per use)."""

    model_config = ConfigDict(extra="allow")

    stored: bool
    uses: list[str]
    read_assist: bool
    models: dict[str, str] = Field(default_factory=dict)
    revision: int
    updated_at: datetime | None = None


class ModelUsageDay(BaseModel):
    model_config = ConfigDict(extra="allow")

    day: date
    use: LLMUse
    tokens: int
    calls: int


class ModelUsage(BaseModel):
    """The tenant's model tokens and calls per day and use."""

    model_config = ConfigDict(extra="allow")

    since: date
    until: date
    days: list[ModelUsageDay] = Field(default_factory=list)


# --------------------------------------------------------------------------- vocabularies

MemoryType = Literal[
    "WORKING",
    "CONVERSATION",
    "EPISODIC",
    "SEMANTIC",
    "PROCEDURAL",
    "USER",
    "PREFERENCE",
    "AGENT",
    "SHARED",
    "TASK",
    "WORK",
    "TOOL",
    "SKILL",
    "DECISION",
    "FAILURE",
    "OUTCOME",
    "ARTIFACT",
    "KNOWLEDGE_RAG",
    "SUMMARY",
    "DERIVED",
    "POLICY",
    "OBSERVATION",
    "BELIEF",
    "ENTITY_SUMMARY",
    "CUSTOM",
]
Lifetime = Literal["EPHEMERAL", "SHORT_TERM", "LONG_TERM", "ARCHIVAL"]
Visibility = Literal[
    "PRIVATE",
    "USER",
    "AGENT_GROUP",
    "RUN",
    "THREAD",
    "WORKSPACE",
    "TENANT",
]
ScopeLevel = Literal["AGENT", "AGENT_GROUP", "THREAD", "USER", "WORKSPACE", "TENANT"]
TemporalStatus = Literal[
    "CURRENT", "SUPERSEDED", "EXPIRED", "CONTRADICTED", "RETRACTED", "ARCHIVED"
]
EvidenceStatus = Literal["COMPLETE", "INCOMPLETE", "INSUFFICIENT"]
#: EVENT: something that happened, told to the service to learn from (always INTERNAL)
MessageRole = Literal["USER", "ASSISTANT", "SYSTEM", "TOOL", "AGENT", "EVENT"]
MessageKind = Literal["VISIBLE", "INTERNAL"]
JobStatus = Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "RETRYING", "CANCELLED"]
DocumentStatus = Literal["STAGED", "READY", "FAILED"]
ArchiveStatus = Literal["STAGED", "ARCHIVING", "ARCHIVED", "PURGED"]
#: What ``search`` reads: memories, document passages, document summaries, thread messages.
SearchKind = Literal["memory", "chunk", "summary", "episode", "message"]
ClaimVerdictValue = Literal["supported", "unsupported", "contradicted", "borderline"]
GroundingMethod = Literal["citation", "nli", "judge"]
ToolStatus = Literal["ok", "error", "timeout", "rejected", "cancelled"]
SideEffects = Literal["read", "write", "irreversible"]
#: What a piece of evidence points at (the service's ``EvidenceSource``).
EvidenceSource = Literal[
    "message",
    "file",
    "document_chunk",
    "agent_result",
    "tool_result",
    "import",
    "observation",
    "statement",
    "memory",
    "graph_fact",
    "summary",
    "episode",
    "feedback",
]
#: Where a catalog entry comes from: the contracts' tool sources and ``manual`` (written
#: through the API).
ToolSource = Literal["manual", "local", "mcp", "memory", "openapi", "a2a"]
#: What a model call was for (the service's ``LLMUse``).
LLMUse = Literal[
    "contextual_extraction",
    "relation_extraction",
    "entity_resolution",
    "conflict_adjudication",
    "summaries",
    "reflection",
    "memory_connections",
    "query_expansion",
    "chunk_context",
    "memory_restatement",
    "grounding_judge",
    "procedure_abstraction",
]
#: What ``keys.whoami`` calls a credential: an issued key's role, or the mode of one that is
#: not an issued key.
KeySelfRole = Literal["platform", "admin", "service", "trusted_dev", "jwt"]


#: the service's id grammar (domain/ids.py ID_PATTERN)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:\-]{0,199}")


class Scope(BaseModel):
    """Identity and lineage for one request. Built by ``MemoryClient.bind``.

    ``tenant_id`` may be omitted when the client's API key names its tenant (``api_key``
    mode); given, it must agree with the key. ``trace_id`` travels as ``traceparent`` and is
    continued by the service when it is a W3C trace id (32 hex); any other value is sent as
    the correlation id instead, which the service echoes but does not trace. An explicit
    ``correlation_id`` wins. Every id here follows the service's grammar (a letter or digit,
    then letters, digits and ``._:-``, at most 200 characters), checked when the scope is
    built rather than refused by the service or rejected by the HTTP client as a header.
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: str | None = None
    workspace_id: str | None = None
    user_id: str | None = None
    thread_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    work_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    agent_group_id: str | None = None
    agent_run_id: str | None = None
    parent_agent_run_id: str | None = None
    trace_id: str | None = None
    correlation_id: str | None = None
    custom_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "tenant_id",
        "workspace_id",
        "user_id",
        "thread_id",
        "session_id",
        "turn_id",
        "work_id",
        "task_id",
        "agent_id",
        "agent_group_id",
        "agent_run_id",
        "parent_agent_run_id",
        "trace_id",
        "correlation_id",
    )
    @classmethod
    def _ids_follow_the_service_grammar(cls, value: str | None) -> str | None:
        if value is not None and not _ID.fullmatch(value):
            raise ValueError(
                "not an id: a letter or digit, then letters, digits and ._:-, at most 200 "
                "characters"
            )
        return value


class MessageAck(BaseModel):
    """Durable acknowledgement: the message and its processing jobs are committed."""

    model_config = ConfigDict(frozen=True)

    message_id: str
    thread_id: str
    session_id: str
    turn_id: str
    sequence: int
    job_ids: list[str] = Field(default_factory=list)
    deduplicated: bool = False


class RememberAck(BaseModel):
    """``remember``: the memory stored now (or the one that already held the same content)."""

    model_config = ConfigDict(frozen=True)

    memory_id: str
    deduplicated: bool = False
    job_ids: list[str] = Field(default_factory=list)


class SupersedeAck(BaseModel):
    """``update``: the new, current version and the one it replaced."""

    model_config = ConfigDict(frozen=True)

    memory_id: str
    supersedes: str


class FileHandle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    document_id: str
    filename: str
    checksum: str
    size_bytes: int
    job_ids: list[str] = Field(default_factory=list)
    deduplicated: bool = False

    @property
    def job_id(self) -> str | None:
        return self.job_ids[0] if self.job_ids else None


class JobHandle(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    status: JobStatus
    attempts: int = 0
    last_error: str | None = None


class EvidenceRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    source_type: EvidenceSource
    source_id: str
    message_id: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    observed_at: datetime | None = None


class MemoryResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    memory_id: str
    content: str
    memory_type: MemoryType
    lifetime: Lifetime
    visibility: Visibility
    scope_level: ScopeLevel | None = None
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    temporal_status: TemporalStatus = "CURRENT"
    supersedes: str | None = None
    superseded_by: str | None = None
    importance: float | None = None
    category: str | None = None
    score: float | None = None
    confidence: float | None = None
    owner_principal: str | None = None
    reinforcement_count: int = 1
    contributors: list[str] = Field(default_factory=list)
    #: part of reinforcement_count that is the agent restating its own output (never raises
    #: confidence) — without it, a high count and no contributors cannot be interpreted
    echoes: int = 0
    contradicts: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class SearchItem(BaseModel):
    """One ``search`` result: enough to use it and cite it."""

    model_config = ConfigDict(frozen=True, extra="allow")

    id: str
    kind: SearchKind
    text: str
    observed_on: str | None = None
    document_id: str | None = None
    page: int | None = None
    #: the conversation an ``episode`` item is
    thread_id: str | None = None
    #: true when a later memory replaced this one (only ``as_of``/``known_at`` return those)
    superseded: bool | None = None
    #: ranking detail, only with ``debug=True``
    debug: dict[str, Any] | None = None


class ContextMemory(BaseModel):
    """A memory of a full context (``context(..., format="full")``)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    id: str
    text: str
    #: 0..1, how close it is to the question; comparable across the bundle
    relevance: float
    observed_at: str | None = None
    #: who or what it is about
    subject: str | None = None
    #: relative dates in the text, resolved: [{"text": "last week", "date": "a..b"}]
    dates: list[dict[str, str]] = Field(default_factory=list)
    #: the messages or documents it came from
    sources: list[str] = Field(default_factory=list)


class ContextPassage(BaseModel):
    """A document passage of a full context."""

    model_config = ConfigDict(frozen=True, extra="allow")

    id: str
    text: str
    relevance: float
    #: None: a document passage (a chunk); relation or memory: a required companion
    kind: Literal["relation", "memory"] | None = None
    document_id: str | None = None
    page: int | None = None
    section: str | None = None


class ContextFact(BaseModel):
    """A knowledge-graph fact of a full context."""

    model_config = ConfigDict(frozen=True, extra="allow")

    id: str
    subject: str
    predicate: str
    object: str
    relevance: float
    observed_at: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    document_id: str | None = None


class ContextSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    id: str
    text: str
    relevance: float


class WindowMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    id: str
    role: MessageRole
    text: str


class Conversation(BaseModel):
    """The thread's recent messages a full context carries."""

    model_config = ConfigDict(frozen=True, extra="allow")

    thread_id: str | None = None
    messages: list[WindowMessage] = Field(default_factory=list)


class PinnedBlock(BaseModel):
    """A pinned profile block as a full context carries it."""

    model_config = ConfigDict(frozen=True, extra="allow")

    block: str
    text: str


class ClaimVerdict(BaseModel):
    """One claim of an answer and what the grounding cascade decided about it."""

    model_config = ConfigDict(frozen=True, extra="allow")

    claim: str
    verdict: ClaimVerdictValue
    support: float = 0.0
    contradiction: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)
    contradicted_by: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    method: GroundingMethod = "nli"
    notes: list[str] = Field(default_factory=list)


class GroundingReport(BaseModel):
    """Per-claim verdicts and the per-claim hallucination rate of a verified answer."""

    model_config = ConfigDict(frozen=True, extra="allow")

    claims: list[ClaimVerdict] = Field(default_factory=list)
    supported: int = 0
    unsupported: int = 0
    contradicted: int = 0
    borderline: int = 0
    per_claim_hallucination_rate: float = 0.0
    nli_provider: str = ""
    representative: bool = False
    judge_consulted: int = 0
    llm_tokens: int = 0
    evidence_count: int = 0
    unused_count: int = 0
    notes: list[str] = Field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return self.per_claim_hallucination_rate == 0.0

    @property
    def score(self) -> float:
        """The share of claims grounded (1 - the per-claim hallucination rate)."""
        return 1.0 - self.per_claim_hallucination_rate


class VerifyReport(GroundingReport):
    """``verify``: the grounding report, and the RUN feedback it was recorded as."""

    feedback_id: str | None = None


class ToolChoiceBrief(BaseModel):
    """A tool that fits the task, as the prompt form carries it."""

    model_config = ConfigDict(frozen=True, extra="allow")

    name: str
    #: 0..1, how well it fits the task
    confidence: float


class PromptContext(BaseModel):
    """What a prompt needs (``context()``): the rendered context, citing by handle ([m1],
    [d2]...), the ``bundle_id`` that ``verify`` and the handles refer to, and its size."""

    model_config = ConfigDict(frozen=True, extra="allow")

    bundle_id: str
    rendered: str
    token_estimate: int
    #: INSUFFICIENT: the memory holds nothing for this question - say you do not know
    evidence_status: EvidenceStatus = "COMPLETE"
    #: the tools that fit the task, best first (only when ``tools`` were given)
    tools: list[ToolChoiceBrief] | None = None
    #: the build's diagnostics (only with ``debug=True``)
    diagnostics: dict[str, Any] | None = None

    @property
    def tool_names(self) -> list[str]:
        """The tools that fit, best first; empty when ``tools`` were not given."""
        return [t.name for t in self.tools or ()]


class ContextBundle(BaseModel):
    """The context as structured data (``context(..., format="full")``), for a caller that
    builds its own prompt: the same content the prompt form renders, without the rendering.
    A section with nothing in it is empty."""

    model_config = ConfigDict(frozen=True, extra="allow")

    bundle_id: str
    evidence_status: EvidenceStatus
    token_estimate: int
    #: required companion evidence that is not there
    missing_evidence: list[str] = Field(default_factory=list)
    conversation: Conversation | None = None
    thread_summary: str | None = None
    profile: list[PinnedBlock] = Field(default_factory=list)
    procedures: list[ProcedureView] = Field(default_factory=list)
    #: the tools that fit, best first (only when ``tools`` were given)
    tools: list[ToolChoice] = Field(default_factory=list)
    memories: list[ContextMemory] = Field(default_factory=list)
    knowledge: list[ContextPassage] = Field(default_factory=list)
    graph_facts: list[ContextFact] = Field(default_factory=list)
    summaries: list[ContextSummary] = Field(default_factory=list)
    diagnostics: dict[str, Any] | None = None

    @property
    def insufficient(self) -> bool:
        return self.evidence_status == "INSUFFICIENT"


class ThreadInfo(BaseModel):
    """A thread, with its durable summary once it has one."""

    model_config = ConfigDict(frozen=True, extra="allow")

    thread_id: str
    tenant_id: str
    title: str | None = None
    created_at: datetime | None = None
    custom_metadata: dict[str, Any] = Field(default_factory=dict)
    summary: ThreadSummary | None = None


class DocumentInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    document_id: str
    title: str
    filename: str
    media_type: str
    size_bytes: int
    checksum: str
    status: DocumentStatus
    archive_status: ArchiveStatus
    current_version_id: str | None = None
    thread_id: str | None = None


class Message(BaseModel):
    """One message to append (``history.add``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: MessageRole
    content: str
    kind: MessageKind = "VISIBLE"
    occurred_at: datetime | None = None
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    custom_metadata: dict[str, Any] = Field(default_factory=dict)
    source_system: str | None = None
    source_message_id: str | None = None
    parent_message_id: str | None = None


class MessageInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    message_id: str
    role: MessageRole
    kind: MessageKind
    sequence: int
    content: str
    occurred_at: datetime | None = None


class GraphEntity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    entity_id: str
    name: str
    canonical_name: str
    entity_type: str = "THING"
    mention_count: int = 1
    aliases: list[str] = Field(default_factory=list)
    summary: str = ""


GraphLayer = Literal["entity", "temporal", "causal", "structural", "procedural"]
RelationStatus = Literal["CURRENT", "SUPERSEDED", "RETRACTED", "INVALIDATED"]


class GraphFact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    relation_id: str
    subject: str
    predicate: str
    object: str
    fact_text: str = ""
    status: RelationStatus = "CURRENT"
    layer: GraphLayer = "entity"
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    observed_at: datetime | None = None
    confidence: float = 0.5
    memory_id: str | None = None
    document_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class GraphNeighborhood(BaseModel):
    """The graph around an entity (``entity(..., depth=n)``)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    entities: list[GraphEntity] = Field(default_factory=list)
    facts: list[GraphFact] = Field(default_factory=list)
    visited: int = 0


class EntityValue(BaseModel):
    """The newest current value of one predicate the entity is the subject of."""

    model_config = ConfigDict(frozen=True, extra="allow")

    predicate: str
    value: str
    relation_id: str
    valid_from: datetime | None = None
    observed_at: datetime


class EntityProfile(BaseModel):
    """``GET /v1/graph/entities/{id}``: current value per predicate, relations, history, and
    with a depth the graph around it."""

    model_config = ConfigDict(frozen=True, extra="allow")

    entity: GraphEntity
    current: list[EntityValue] = Field(default_factory=list)
    relations: list[GraphFact] = Field(default_factory=list)
    history: list[GraphFact] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    neighborhood: GraphNeighborhood | None = None


# --------------------------------------------------------------------------- tools


class ToolResult(BaseModel):
    """``record_tool``: the stored invocation (``recorded=False`` when it was a retry)."""

    model_config = ConfigDict(extra="allow")

    invocation_id: str
    step: int = 0
    args_hash: str = ""
    recorded: bool = True


class ToolStats(BaseModel):
    """What the service has seen a tool do (calls, success, latency, approvals)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    calls: int = 0
    successes: int = 0
    failures: int = 0
    success_rate: float | None = None
    avg_latency_ms: float | None = None
    approvals: int = 0
    rejections: int = 0
    edits: int = 0
    last_used_at: datetime | None = None


class CatalogTool(BaseModel):
    """One catalog entry: what a tool is and does (``GET /v1/tools``)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    tool_id: str
    name: str
    version: int = 1
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    argument_entity_types: dict[str, str] = Field(default_factory=dict)
    side_effects: SideEffects | None = None
    source: ToolSource = "manual"
    server: str | None = None
    examples: list[dict[str, Any]] = Field(default_factory=list)
    #: the MCP annotations (readOnlyHint, destructiveHint, idempotentHint, openWorldHint)
    annotations: dict[str, bool] = Field(default_factory=dict)
    #: ask a person before a call when this is true (``trellis.memory.approval``)
    approve_when: str | None = None
    #: read (run), write (run and notify) or irreversible (ask)
    risk: SideEffects = "write"
    schema_hash: str = ""
    workspace_id: str | None = None
    stats: ToolStats = Field(default_factory=ToolStats)


class ApprovalSuggestion(BaseModel):
    """A rule the approvals given so far support; the service applies it only when it is
    accepted (``tools.accept_suggestion``)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    id: str
    accepted: bool = False
    tool: str
    arg_shape: str
    suggestion: Literal["auto_approve", "always_ask"]
    approvals: int
    rejections: int
    edits: int
    support: int
    approve_rate: float
    agent_id: str | None = None


class MissingArgument(BaseModel):
    """A required argument nothing could fill: ask the user ``question``."""

    model_config = ConfigDict(frozen=True, extra="allow")

    arg: str
    question: str
    #: the entity type the argument names (e.g. ORG), when the catalog says
    entity_type: str | None = None


class ToolChoice(BaseModel):
    """A tool that fits the task: how well, how it has done, and its arguments."""

    model_config = ConfigDict(frozen=True, extra="allow")

    name: str
    #: 0..1, how well it fits the task
    confidence: float
    #: share of its recorded calls that succeeded; None: never called
    success_rate: float | None = None
    #: the learned plan's next step
    next: bool = False
    #: argument values already found, typed as the tool's schema says
    args: dict[str, Any] = Field(default_factory=dict)
    #: required arguments nothing found
    missing: list[MissingArgument] = Field(default_factory=list)


class ToolHints(BaseModel):
    """Which tools fit a task, best first, and the learned plan (``tool_hints``)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    tools: list[ToolChoice] = Field(default_factory=list)
    plan: ProcedureView | None = None

    @property
    def next(self) -> ToolChoice | None:
        """The plan's next step, else the best choice."""
        return next((t for t in self.tools if t.next), self.tools[0] if self.tools else None)


class AgentTool(BaseModel):
    """One of the memory tools an agent may call (``call_agent_tool``)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    name: str
    description: str
    input_schema: dict[str, Any]


# --------------------------------------------------------------------------- profile, summary


class ProfileBlock(BaseModel):
    """A pinned block of text: ``user``, ``agent``, ``workspace`` or ``<level>.<name>``."""

    model_config = ConfigDict(frozen=True, extra="allow")

    block: str
    text: str
    version: int
    updated_at: datetime | None = None
    #: the standing question the service keeps this block answering
    source_query: str | None = None


class ThreadSummary(BaseModel):
    """The durable summary of a thread, up to ``covers_to_sequence``."""

    model_config = ConfigDict(frozen=True, extra="allow")

    text: str
    covers_to_sequence: int
    version: int
    thread_id: str | None = None
    model: str | None = None
    created_at: datetime | None = None


class ProcedureView(BaseModel):
    """A procedure learned for the task: its tool sequence and how often it worked."""

    model_config = ConfigDict(frozen=True, extra="allow")

    id: str
    title: str | None = None
    #: tool names, in order
    steps: list[str] = Field(default_factory=list)
    success_rate: float = 0.0
    #: the successful runs it was learned from
    runs: int = 0


# --- platform administration -----------------------------------------------------

KeyRole = Literal["admin", "service"]
MemberRole = Literal["admin", "member", "viewer"]
TenantStatus = Literal["active", "suspended"]


class TenantInfo(BaseModel):
    tenant_id: str
    name: str
    status: TenantStatus
    retention_days: int | None = None
    rate_limit_per_minute: int | None = None
    created_at: datetime
    updated_at: datetime


class ApiKeyInfo(BaseModel):
    key_id: str
    tenant_id: str
    role: KeyRole
    name: str
    workspace_id: str | None = None
    created_by: str
    created_at: datetime
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    last_used_at: datetime | None = None
    may_act_as: list[str] = Field(default_factory=lambda: ["*"])


class KeyInfo(BaseModel):
    """Who a key is (``keys.whoami``)."""

    key_id: str
    tenant_id: str | None = None
    principal: str
    role: KeySelfRole
    may_act_as: list[str] = Field(default_factory=list)


class IssuedKey(ApiKeyInfo):
    """The record plus the secret, which the service shows exactly once.

    ``token`` is None when the response was an idempotent replay (the same
    ``Idempotency-Key`` sent twice): the key exists, but the secret is not shown again.
    """

    token: str | None = None


class CreatedTenant(BaseModel):
    tenant: TenantInfo
    admin_key: IssuedKey


class WorkspaceInfo(BaseModel):
    workspace_id: str
    tenant_id: str
    name: str
    created_at: datetime


class WorkspaceMemberInfo(BaseModel):
    workspace_id: str
    principal: str
    role: MemberRole
    added_by: str
    added_at: datetime


class ReadAuditRecord(BaseModel):
    credential: str
    principal: str
    kind: Literal["recall", "context"]
    query_hash: str
    scope_fingerprint: str
    record_ids: list[str]
    at: datetime


# --------------------------------------------------------------------------- pagination


class Page[T](BaseModel):
    """One page of a list route: the items and the cursor of the next page (None on the last).
    Pass ``next_cursor`` back as ``cursor`` to continue; every paged route accepts it."""

    model_config = ConfigDict(frozen=True)

    items: list[T]
    next_cursor: str | None = None

    @property
    def has_more(self) -> bool:
        return self.next_cursor is not None


# --------------------------------------------------------------------------- feedback

FeedbackTargetKind = Literal["run", "memory", "tool_call", "procedure"]
FeedbackVerdict = Literal["confirm", "reject", "correct", "approve", "edit"]
#: who judged: a person (human, or interrupt: an answer to an interrupt), the grounding judge,
#: or the run's own final status (system); a run's outcome follows the highest-ranked
FeedbackSource = Literal["human", "interrupt", "judge", "system"]
ProjectionAction = Literal[
    "none",
    "memory_reinforced",
    "memory_retracted",
    "memory_superseded",
    "run_labelled",
    "tool_call_counted",
    "procedure_rejected",
]


class FeedbackProjection(BaseModel):
    """What the service did with a record once its projector ran."""

    model_config = ConfigDict(frozen=True, extra="allow")

    action: ProjectionAction
    memory_id: str | None = None
    memory_ids: list[str] = Field(default_factory=list)
    run_id: str | None = None
    superseded_by: str | None = None
    reason: str | None = None
    projected_at: datetime


#: where a vote that waits for a tenant admin stands
ReviewState = Literal["pending", "approved", "dismissed"]


class FeedbackReview(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    state: ReviewState
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    note: str | None = None


class Feedback(BaseModel):
    """A stored judgement (the ``trellis.contracts.Feedback`` record plus its projection)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    feedback_id: str
    tenant_id: str
    workspace_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    agent_run_id: str | None = None
    trace_id: str | None = None
    target_kind: FeedbackTargetKind
    target_id: str
    verdict: FeedbackVerdict
    correction: Any = None
    score: float | None = None
    comment: str | None = None
    reviewer: str | None = None
    source: FeedbackSource = "human"
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    projection: FeedbackProjection | None = None
    #: null: applied as it arrived; else where its review stands
    review: FeedbackReview | None = None
    #: in the review queue only: how the author's verdicts fared (pending/approved/dismissed)
    author_record: dict[str, int] | None = None
