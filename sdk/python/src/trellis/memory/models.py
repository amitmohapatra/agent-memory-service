"""SDK-facing models. Mirrors the public API contract; no internal types leak here.

The closed vocabularies below are the API's enums spelled as Literals, so a wrong value is
a type error in the caller's editor and a 422 from the service, never a silent no-op. They
are kept in step with the service by a test in the service repository.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentKeyStatus(BaseModel):
    """Metadata only; neither a virtual key nor encrypted material is returned."""

    registered: bool
    revoked: bool
    revision: int
    updated_at: datetime | None = None


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
Representation = Literal[
    "RAW_FILE",
    "DOCUMENT",
    "DOCUMENT_VERSION",
    "SECTION",
    "SUBSECTION",
    "PARAGRAPH",
    "TABLE",
    "CODE_BLOCK",
    "CHUNK",
    "SUMMARY",
    "ENTITY",
    "RELATION",
    "EMBEDDING",
    "MESSAGE",
    "MEMORY",
]
QueryType = Literal[
    "EXACT_IDENTIFIER",
    "CONVERSATION_HISTORY",
    "USER_MEMORY",
    "DECISION",
    "DOCUMENT_LOCAL",
    "DOCUMENT_MULTI_HOP",
    "ENTITY_RELATION",
    "TEMPORAL",
    "GLOBAL_SUMMARY",
    "GENERAL_SEMANTIC",
]
EvidenceStatus = Literal["COMPLETE", "INCOMPLETE", "INSUFFICIENT"]
MessageRole = Literal["USER", "ASSISTANT", "SYSTEM", "TOOL", "AGENT"]
MessageKind = Literal["VISIBLE", "INTERNAL"]
ObservationKind = Literal[
    "MESSAGE", "FILE", "AGENT_RESULT", "TOOL_RESULT", "DECISION", "FEEDBACK", "EVENT", "IMPORT"
]
JobStatus = Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "RETRYING", "CANCELLED"]
DocumentStatus = Literal["STAGED", "READY", "FAILED"]
ArchiveStatus = Literal["STAGED", "ARCHIVING", "ARCHIVED", "PURGED"]
#: What ``recall`` searches: document passages, canonical memories, rolled-up summaries.
RecallKind = Literal["chunk", "memory", "summary"]
#: What ``verify`` accepts as an item's kind: a bundle item's representation or the record
#: kind of an unused item.
EvidenceKind = Literal[
    "CHUNK",
    "TABLE",
    "PARAGRAPH",
    "SECTION",
    "SUBSECTION",
    "CODE_BLOCK",
    "chunk",
    "SUMMARY",
    "summary",
    "ENTITY",
    "RELATION",
    "MEMORY",
    "memory",
    "fact",
]
ClaimVerdictValue = Literal["supported", "unsupported", "contradicted", "borderline"]
GroundingMethod = Literal["citation", "nli", "judge"]
ToolSource = Literal["bifrost-mcp", "langgraph", "adk", "crewai", "mcp", "manual"]
ToolStatus = Literal["ok", "error", "timeout", "rejected"]
SideEffects = Literal["none", "read", "write", "external", "unknown"]
CacheScope = Literal["run", "thread", "user", "tenant"]


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


class ObservationAck(BaseModel):
    model_config = ConfigDict(frozen=True)

    observation_id: str
    job_ids: list[str] = Field(default_factory=list)
    deduplicated: bool = False


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

    source_type: str
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


class ContextItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    item_id: str
    representation: Representation
    text: str
    score: float = 0.0
    citation: str
    document_id: str | None = None
    page: int | None = None
    section_path: str | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @property
    def predicate(self) -> str | None:
        return self.attributes.get("predicate")

    def verification_item(self) -> dict[str, Any]:
        """Preserve provenance when forwarding retrieved text as grounding evidence."""
        return {
            "item_id": self.item_id,
            "text": self.text,
            "kind": self.representation,
            "citation": self.citation,
            "attributes": dict(self.attributes),
        }


class ConversationWindow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    thread_id: str | None = None
    message_ids: list[str] = Field(default_factory=list)
    rendered: str = ""
    summary: str | None = None


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


class UnusedEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    item_id: str
    kind: EvidenceKind = "chunk"
    text: str


class EvidenceReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    status: EvidenceStatus
    required_groups: list[str] = Field(default_factory=list)
    satisfied_groups: list[str] = Field(default_factory=list)
    missing_groups: list[str] = Field(default_factory=list)
    escalations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    unused: list[UnusedEvidence] = Field(default_factory=list)
    grounding: GroundingReport | None = None
    llm_tokens: int = 0


class ContextBundle(BaseModel):
    """Bounded, ranked context for the current turn. ``rendered`` is ready to prompt with."""

    model_config = ConfigDict(frozen=True, extra="allow")

    query: str
    query_type: QueryType
    bundle_id: str = ""
    conversation: ConversationWindow
    memories: list[ContextItem] = Field(default_factory=list)
    knowledge: list[ContextItem] = Field(default_factory=list)
    graph_facts: list[ContextItem] = Field(default_factory=list)
    summaries: list[ContextItem] = Field(default_factory=list)
    evidence: EvidenceReport
    token_budget: int
    token_estimate: int
    rendered: str = ""
    cache_hit: bool = False

    @property
    def insufficient(self) -> bool:
        return self.evidence.status == "INSUFFICIENT"

    @property
    def grounding(self) -> GroundingReport | None:
        return self.evidence.grounding

    def evidence_items(self) -> list[dict[str, Any]]:
        """The packed evidence as ``/v1/verify`` items, in citation order (``[1]`` is the
        first memory, then facts, summaries, knowledge)."""
        return [
            i.verification_item()
            for group in (self.memories, self.graph_facts, self.summaries, self.knowledge)
            for i in group
        ]


class ThreadInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    thread_id: str
    tenant_id: str
    title: str | None = None
    created_at: datetime | None = None


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


class GraphFact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    relation_id: str
    subject: str
    predicate: str
    object: str
    fact_text: str = ""
    status: TemporalStatus = "CURRENT"
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    observed_at: datetime | None = None
    confidence: float = 0.5
    memory_id: str | None = None
    document_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class GraphAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    matched: list[GraphEntity] = Field(default_factory=list)
    entities: list[GraphEntity] = Field(default_factory=list)
    facts: list[GraphFact] = Field(default_factory=list)
    visited: int = 0


# --------------------------------------------------------------------------- tool memory


class ToolPolicyModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    deterministic: bool = False
    side_effects: SideEffects = "unknown"
    cacheable: bool = False
    cache_ttl_seconds: int = 300
    cache_scope: CacheScope = "run"
    cost_hint: float | None = None
    redact: list[str] = Field(default_factory=list)


class Tool(BaseModel):
    model_config = ConfigDict(extra="allow")

    tool_id: str
    name: str
    version: int = 1
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    source: ToolSource = "manual"
    policy: ToolPolicyModel = Field(default_factory=ToolPolicyModel)
    stats: dict[str, Any] | None = None


class ToolCall(BaseModel):
    """One call an agent is about to make, or has just made."""

    model_config = ConfigDict(extra="allow")

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    task: str = ""
    step: int | None = None


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    invocation_id: str | None = None
    step: int = 0
    args_hash: str = ""
    recorded: bool = True
    cached: bool = False
    age_seconds: float | None = None
    output: Any = None
    output_summary: str | None = None
    output_fields: dict[str, Any] = Field(default_factory=dict)


class ToolSuggestion(BaseModel):
    model_config = ConfigDict(extra="allow")

    tool: str
    confidence: float = 0.0
    argument_template: dict[str, Any] = Field(default_factory=dict)
    supporting_procedures: list[str] = Field(default_factory=list)
    supporting_invocations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence_status: str = "NONE"

    def render(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.argument_template.items())
        line = f"{self.tool}({args})  [confidence {self.confidence:.2f}]"
        if self.warnings:
            line += "\n  warning: " + "; ".join(self.warnings)
        return line


class NextSteps(BaseModel):
    model_config = ConfigDict(extra="allow")

    suggestions: list[ToolSuggestion] = Field(default_factory=list)
    stop: bool = False
    matched_procedure: str | None = None
    matched_prefix_length: int = 0

    def render(self) -> str:
        if self.stop:
            return "nothing further to call"
        return "\n".join(s.render() for s in self.suggestions) or "no suggestion"


class ToolPlan(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_pattern: str = ""
    steps: list[dict[str, Any]] = Field(default_factory=list)
    valid: bool = False
    reason: str | None = None
    problems: list[str] = Field(default_factory=list)
    support: int = 0
    success_rate: float = 0.0
    script: str | None = None
    rendered: str | None = None
    run_ids: list[str] = Field(default_factory=list)
    invocation_ids: list[str] = Field(default_factory=list)

    def render(self) -> str:
        if self.rendered:
            return self.rendered
        if not self.valid:
            return f"no validated plan ({self.reason or 'unknown'})"
        return "\n".join(f"{i + 1}. {s.get('tool')}" for i, s in enumerate(self.steps))

    def render_script(self) -> str:
        """Starlark form for Bifrost code mode."""
        return self.script or ""


class BriefSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["mental_model", "knowledge_page"] = "mental_model"
    title: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=4000)
    use_llm: bool = False
    refresh_seconds: int = Field(default=3600, ge=60, le=86400)


class BriefOutput(BaseModel):
    text: str
    sources: list[ContextItem] = Field(default_factory=list)
    generated: bool = False
    generation_profile: str | None = None
    revision_fingerprint: str
    valid_until: datetime
    built_at: datetime


class BriefInfo(BaseModel):
    brief_id: str
    spec: BriefSpec


class Brief(BriefInfo):
    status: Literal["pending", "ready", "stale"]
    output: BriefOutput | None = None


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
    role: str
    name: str
    workspace_id: str | None = None
    created_by: str
    created_at: datetime
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    last_used_at: datetime | None = None


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


class GroupInfo(BaseModel):
    group_id: str
    tenant_id: str
    name: str
    created_at: datetime


class GroupMemberInfo(BaseModel):
    group_id: str
    user_id: str
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

FeedbackTargetKind = Literal["run", "answer", "memory", "tool_call", "brief", "procedure"]
FeedbackVerdict = Literal["confirm", "reject", "correct", "approve", "edit"]
FeedbackSource = Literal["human", "judge", "interrupt"]
ProjectionAction = Literal["none", "memory_reinforced", "memory_retracted", "memory_superseded"]


class FeedbackProjection(BaseModel):
    """What the service did with a record once its projector ran."""

    model_config = ConfigDict(frozen=True, extra="allow")

    action: ProjectionAction
    memory_id: str | None = None
    superseded_by: str | None = None
    reason: str | None = None
    projected_at: datetime


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


# --------------------------------------------------------------------------- webhooks

WebhookEvent = Literal[
    "memory.created",
    "memory.superseded",
    "memory.retracted",
    "feedback.received",
    "feedback.projected",
    "webhook.test",
]
DeliveryStatus = Literal["PENDING", "DELIVERED", "FAILED", "DEAD"]


class WebhookInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    subscription_id: str
    tenant_id: str
    workspace_id: str | None = None
    url: str
    events: list[WebhookEvent]
    description: str | None = None
    enabled: bool
    failures: int = 0
    created_by: str
    created_at: datetime
    updated_at: datetime


class WebhookCreated(WebhookInfo):
    """The subscription plus its HMAC secret, shown once (None on an idempotent replay)."""

    secret: str | None = None


class DeliveryInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    delivery_id: str
    subscription_id: str
    event_id: str
    event_type: WebhookEvent
    status: DeliveryStatus
    attempts: int
    status_code: int | None = None
    last_error: str | None = None
    created_at: datetime
    delivered_at: datetime | None = None


class WebhookEventPayload(BaseModel):
    """What a receiver gets: the event, as the service serialises it."""

    model_config = ConfigDict(frozen=True, extra="allow")

    event_id: str
    type: WebhookEvent
    tenant_id: str
    workspace_id: str | None = None
    occurred_at: datetime
    data: dict[str, Any] = Field(default_factory=dict)
