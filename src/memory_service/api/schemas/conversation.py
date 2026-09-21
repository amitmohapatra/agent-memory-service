"""Public API models for threads and messages (typed, with examples)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from memory_service.api.deps import ScopeBody
from memory_service.domain.enums import (
    ArchiveStatus,
    Lifetime,
    MemoryType,
    MessageKind,
    MessageRole,
    Visibility,
)

_SCOPE_EXAMPLE: dict[str, Any] = {
    "thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
    "session_id": "ses_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
    "turn_id": "trn_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
}


class ProcessingHintsIn(BaseModel):
    """Expert-only overrides; omit for the default 'the service decides' behaviour.

    Every field here is a *hint*: the service classifies observations on its own, and the
    right call for almost every caller is to send none of them. They exist for the cases
    where the caller genuinely knows something the extractor cannot infer — an import whose
    provenance is already known, a note that must not outlive the run.

    Setting one wrongly is worse than leaving it unset: a hint overrides the classifier, so a
    ``memory_type`` that does not match the content makes the memory unreachable by the
    queries that should find it.
    """

    model_config = ConfigDict(extra="forbid")

    lifetime: Lifetime | None = Field(
        default=None,
        examples=["LONG_TERM"],
        description=(
            "How long this should survive. EPHEMERAL: within the turn only. SHORT_TERM: the "
            "current thread/session. LONG_TERM: durable, the default for facts about a user "
            "or the world. ARCHIVAL: cold storage, retained for audit rather than retrieval. "
            "Omit to let the extractor choose from the content."
        ),
    )
    memory_type: MemoryType | None = Field(
        default=None,
        examples=["PREFERENCE"],
        description=(
            "What kind of thing this is. The ones a caller normally means: SEMANTIC (a fact "
            "about the world), PREFERENCE (how the user likes things), EPISODIC (something "
            "that happened), DECISION (a choice and its reason), PROCEDURAL (how to do "
            "something). ENTITY_SUMMARY, TOOL, OBSERVATION, AGENT, BELIEF, TASK and USER are "
            "written by the pipeline itself; setting them by hand mislabels the record. The "
            "remaining values (WORKING, CONVERSATION, SHARED, WORK, SKILL, DECISION, FAILURE, "
            "OUTCOME, ARTIFACT, KNOWLEDGE_RAG, SUMMARY, DERIVED, POLICY, CUSTOM) are accepted "
            "but never produced by extraction — they exist for callers importing records whose "
            "type is already known. Omit unless that is what you are doing."
        ),
    )
    custom_type: str | None = Field(
        default=None,
        examples=["release_note"],
        description=(
            "Required when memory_type=CUSTOM, and meaningless otherwise: the caller's own "
            "label for a record this service's taxonomy has no name for. Without it a CUSTOM "
            "memory is rejected, so omitting it turns the hint into a failed write rather "
            "than a stored memory."
        ),
    )
    visibility: Visibility | None = Field(
        default=None,
        examples=["USER"],
        description=(
            "Who may retrieve it, narrowest first: PRIVATE, RUN, THREAD, WORK, AGENT_GROUP, "
            "GROUP, USER, WORKSPACE, TENANT, GLOBAL. Each level is a superset of the ones "
            "before it. Omit to inherit the scope the observation was submitted in — which is "
            "the safe answer; widening by hand is how one tenant's data reaches another."
        ),
    )
    importance: float | None = Field(
        default=None,
        ge=0,
        le=1,
        examples=[0.8],
        description=(
            "0-1 prior on how much this matters, influencing admission and ranking. Omit "
            "unless you have a real signal; a blanket high value just flattens ranking."
        ),
    )
    skip_extraction: bool = Field(
        default=False,
        description="Store the observation verbatim without deriving memories from it.",
    )
    skip_embedding: bool = Field(
        default=False, description="Do not index for semantic search (keyword/graph only)."
    )
    skip_graph: bool = Field(
        default=False, description="Do not extract entities or relations into the graph."
    )
    skip_summary: bool = Field(
        default=False, description="Do not roll this into thread or entity summaries."
    )


class AttachmentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(..., examples=["report.pdf"])
    media_type: str = Field(..., examples=["application/pdf"])
    size_bytes: int = Field(..., ge=0, examples=[204800])
    checksum: str = Field(
        ...,
        description="SHA-256 hex of the raw bytes",
        examples=["9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"],
    )
    document_id: str | None = Field(
        default=None, description="Set when the file was already ingested via /v1/files"
    )


class CreateThreadRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"scope": {"thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH"}, "title": "Q3 planning"}
            ]
        },
    )

    scope: ScopeBody = Field(
        default_factory=ScopeBody, examples=[{"thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH"}]
    )
    thread_id: str | None = Field(
        default=None,
        description="Client-generated id; generated when absent",
        examples=["thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH"],
    )
    title: str | None = Field(default=None, examples=["Q3 planning"])
    custom_metadata: dict[str, Any] = Field(default_factory=dict, examples=[{"channel": "web"}])


class ThreadResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                    "tenant_id": "acme",
                    "workspace_id": "ws-finance",
                    "owner_user_id": "u-123",
                    "title": "Q3 planning",
                    "revision": 3,
                    "created_at": "2026-09-14T10:00:00Z",
                    "updated_at": "2026-09-14T10:05:00Z",
                    "custom_metadata": {},
                }
            ]
        }
    )

    thread_id: str
    tenant_id: str
    workspace_id: str | None = None
    owner_user_id: str | None = None
    title: str | None = None
    revision: int
    created_at: datetime
    updated_at: datetime
    custom_metadata: dict[str, Any] = Field(default_factory=dict)


class CreateMessageRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "scope": _SCOPE_EXAMPLE,
                    "role": "USER",
                    "kind": "VISIBLE",
                    "content": "Why did EBITDA increase despite lower revenue?",
                }
            ]
        },
    )

    scope: ScopeBody = Field(
        ...,
        description="Lineage: thread/session/turn (+ agent fields for internal messages)",
        examples=[_SCOPE_EXAMPLE],
    )
    role: MessageRole = Field(..., examples=["USER"])
    kind: MessageKind = Field(
        default=MessageKind.VISIBLE,
        description=(
            "VISIBLE messages form the chat history; INTERNAL messages record agent/tool execution."
        ),
        examples=["VISIBLE"],
    )
    content: str = Field(
        ...,
        min_length=0,
        max_length=2_000_000,
        examples=["Why did EBITDA increase despite lower revenue?"],
    )
    attachments: list[AttachmentIn] = Field(default_factory=list)
    hints: ProcessingHintsIn = Field(default_factory=ProcessingHintsIn)
    custom_metadata: dict[str, Any] = Field(default_factory=dict, examples=[{"ui_locale": "en-GB"}])
    occurred_at: datetime | None = Field(default=None, description="Original timestamp for imports")
    source_system: str | None = Field(default=None, examples=["slack"])
    source_message_id: str | None = Field(default=None, examples=["1726300000.000100"])
    parent_message_id: str | None = None


class MessageAckResponse(BaseModel):
    """Durable acknowledgement. Returned only after the message and its jobs are committed."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "message_id": "msg_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                    "thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                    "session_id": "ses_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                    "turn_id": "trn_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                    "sequence": 7,
                    "job_ids": ["obx_1042", "obx_1043"],
                    "deduplicated": False,
                    "observation_id": "obs_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                }
            ]
        }
    )

    message_id: str
    thread_id: str
    session_id: str
    turn_id: str
    sequence: int
    job_ids: list[str] = Field(
        default_factory=list, description="Job references; poll GET /v1/jobs/{job_id}"
    )
    deduplicated: bool = False
    observation_id: str | None = None


class MessageResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "message_id": "msg_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                    "thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                    "session_id": "ses_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                    "turn_id": "trn_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                    "role": "USER",
                    "kind": "VISIBLE",
                    "sequence": 7,
                    "content": "Why did EBITDA increase?",
                    "author_principal": "user:u-123",
                    "occurred_at": "2026-09-14T10:00:00Z",
                    "archive_status": "STAGED",
                    "attachments": [],
                    "custom_metadata": {},
                }
            ]
        }
    )

    message_id: str
    thread_id: str
    session_id: str
    turn_id: str
    role: MessageRole
    kind: MessageKind
    sequence: int
    content: str
    author_principal: str
    agent_run_id: str | None = None
    occurred_at: datetime
    archive_status: ArchiveStatus
    attachments: list[AttachmentIn] = Field(default_factory=list)
    custom_metadata: dict[str, Any] = Field(default_factory=dict)


class MessageListResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
                    "messages": [],
                    "next_before_sequence": None,
                }
            ]
        }
    )

    thread_id: str
    messages: list[MessageResponse]
    next_before_sequence: int | None = Field(
        default=None, description="Pass as ?before_sequence= to page backwards"
    )


class JobResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "job_id": "obx_1042",
                    "task_name": "memory.process_observation",
                    "queue": "chat-fast",
                    "status": "SUCCEEDED",
                    "attempts": 1,
                    "last_error": None,
                }
            ]
        }
    )

    job_id: str
    task_name: str
    queue: str
    status: str
    attempts: int = 0
    last_error: str | None = None
