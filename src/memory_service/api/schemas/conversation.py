"""Public API models for threads and messages (typed, with examples)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_service.api.deps import ScopeBody
from memory_service.api.validation import CustomMetadata
from memory_service.domain.enums import ArchiveStatus, JobStatus, MessageKind, MessageRole
from memory_service.domain.instants import UTC_RULE, UtcDateTime

_SCOPE_EXAMPLE: dict[str, Any] = {
    "thread_id": "thr_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
    "session_id": "ses_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
    "turn_id": "trn_01J8ZK7Q9V3W2X1Y0ZABCDEFGH",
}


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
        default=None, description="Set when the file was already ingested via POST /v1/documents"
    )


class PatchThreadRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"title": "Q3 planning", "custom_metadata": {}}]},
    )

    scope: ScopeBody = Field(default_factory=ScopeBody)
    title: str | None = Field(default=None, max_length=500, examples=["Q3 planning"])
    custom_metadata: CustomMetadata | None = Field(
        default=None, description="replaces the thread's metadata when given"
    )


class ThreadSummaryBody(BaseModel):
    """The thread's durable summary: every message up to ``covers_to_sequence``."""

    text: str
    covers_to_sequence: int
    version: int
    model: str = Field(description="the model that wrote it, or extractive without one")
    created_at: datetime


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
    summary: ThreadSummaryBody | None = Field(
        default=None, description="the durable summary, once the thread has one"
    )


class MessageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: MessageRole = Field(
        ...,
        description=(
            "Who produced the message: USER, ASSISTANT or SYSTEM for the visible chat; TOOL "
            "for a tool's output and AGENT for an internal agent step (kind=INTERNAL); EVENT "
            "for something that happened, which the service learns from (always INTERNAL)."
        ),
        examples=["USER"],
    )
    kind: MessageKind = Field(
        default=MessageKind.VISIBLE,
        description="VISIBLE messages form the chat history; INTERNAL messages record "
        "agent/tool execution and events.",
        examples=["VISIBLE"],
    )
    content: str = Field(
        ...,
        min_length=0,
        max_length=2_000_000,
        examples=["Why did EBITDA increase despite lower revenue?"],
    )
    attachments: list[AttachmentIn] = Field(default_factory=list)
    custom_metadata: CustomMetadata = Field(default_factory=dict, examples=[{"ui_locale": "en-GB"}])
    occurred_at: UtcDateTime | None = Field(
        default=None,
        description="When the message was originally sent, for imports of an older "
        f"conversation; omitted: now. {UTC_RULE}",
    )
    source_system: str | None = Field(default=None, max_length=100, examples=["slack"])
    source_message_id: str | None = Field(
        default=None, max_length=400, examples=["1726300000.000100"]
    )
    parent_message_id: str | None = None

    @model_validator(mode="after")
    def _an_event_says_something(self) -> MessageIn:
        """An EVENT is only what it says happened: an empty one has nothing to learn from
        (``/v1/observations``, which EVENT replaced, refused it too)."""
        if self.role is MessageRole.EVENT and not self.content.strip():
            raise ValueError("an EVENT message needs content")
        return self


#: Messages one request appends.
MESSAGES_MAX = 100


class CreateMessagesRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "scope": _SCOPE_EXAMPLE,
                    "messages": [
                        {"role": "USER", "content": "Why did EBITDA increase?"},
                        {"role": "ASSISTANT", "content": "Lower costs [d1]."},
                    ],
                }
            ]
        },
    )

    scope: ScopeBody = Field(
        ...,
        description="Lineage: the thread (defaults to the agent run's id when a run is in the "
        "scope), and optionally the session and turn (+ agent fields for internal messages). "
        "Without a session the messages join the thread's own session; without a turn a USER "
        "message opens the thread's next turn and any other joins its latest. The "
        "acknowledgements return the ids used.",
        examples=[_SCOPE_EXAMPLE],
    )
    messages: list[MessageIn] = Field(..., min_length=1, max_length=MESSAGES_MAX)


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
    role: MessageRole = Field(
        ...,
        description="Who produced it: USER, ASSISTANT, SYSTEM, TOOL, AGENT or EVENT.",
    )
    kind: MessageKind = Field(
        ...,
        description="VISIBLE messages form the chat history; INTERNAL ones are agent/tool steps.",
    )
    sequence: int
    content: str
    author_principal: str
    agent_run_id: str | None = None
    occurred_at: datetime
    archive_status: ArchiveStatus = Field(
        ...,
        description=(
            "Where the content lives: STAGED (PostgreSQL only), ARCHIVING, ARCHIVED (blob "
            "written and verified) or PURGED (large payload removed from the hot store)."
        ),
    )
    attachments: list[AttachmentIn] = Field(default_factory=list)
    custom_metadata: dict[str, Any] = Field(default_factory=dict)


class MessagesAckResponse(BaseModel):
    """One acknowledgement per message, in request order."""

    messages: list[MessageAckResponse]


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
    next_cursor: str | None = Field(
        default=None, description="pass as `cursor` for the next page; null on the last"
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
    status: JobStatus = Field(
        ...,
        description="PENDING (queued, not picked up), RUNNING, SUCCEEDED, FAILED (attempts "
        "exhausted; see last_error), RETRYING (failed, will run again) or CANCELLED.",
    )
    attempts: int = 0
    last_error: str | None = None
