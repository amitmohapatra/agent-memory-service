"""Conversation data model: threads, sessions, turns, messages and agent runs.

ID semantics
------------
new chat                 -> new thread, new session, new turn
next question, same chat -> same thread, same session, new turn
reopen chat later        -> same thread, new session, new turn
internal agent run       -> inherits tenant/workspace/user/thread/session/turn/trace,
                            adds agent/agent_run/task/parent_run

Visible UI history contains only USER/ASSISTANT messages of kind VISIBLE. Internal
execution (planner, agent results, tool results) is recorded as INTERNAL messages
and agent runs and never appears as visible chat automatically.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.enums import ArchiveStatus, MessageKind, MessageRole
from memory_service.domain.ids import new_id


class Thread(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(default_factory=lambda: new_id("thread"))
    tenant_id: str
    workspace_id: str | None = None
    owner_user_id: str | None = None
    title: str | None = None
    source_system: str | None = Field(default=None, description="set for imported conversations")
    source_thread_id: str | None = None
    system_metadata: dict[str, Any] = Field(default_factory=dict)
    custom_metadata: dict[str, Any] = Field(default_factory=dict)
    revision: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    archived_at: datetime | None = None
    deleted_at: datetime | None = None


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(default_factory=lambda: new_id("session"))
    thread_id: str
    tenant_id: str
    user_id: str | None = None
    client: str | None = Field(default=None, description="ui | sdk | langgraph | import")
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    custom_metadata: dict[str, Any] = Field(default_factory=dict)


class Turn(BaseModel):
    """One user question and everything that happened while answering it."""

    model_config = ConfigDict(extra="forbid")

    turn_id: str = Field(default_factory=lambda: new_id("turn"))
    session_id: str
    thread_id: str
    tenant_id: str
    sequence: int = Field(..., ge=1, description="1-based position in the thread")
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    custom_metadata: dict[str, Any] = Field(default_factory=dict)


class Attachment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attachment_id: str = Field(default_factory=lambda: new_id("attachment"))
    message_id: str
    document_id: str | None = Field(default=None, description="set once the file is ingested")
    filename: str
    media_type: str
    size_bytes: int = Field(..., ge=0)
    checksum: str = Field(..., description="SHA-256 of the raw bytes")


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(default_factory=lambda: new_id("message"))
    thread_id: str
    session_id: str
    turn_id: str
    tenant_id: str
    role: MessageRole
    kind: MessageKind = MessageKind.VISIBLE
    sequence: int = Field(..., ge=1, description="1-based position within the thread")
    content: str = Field(default="", description="current version text")
    content_hash: str = Field(..., description="SHA-256 of content")
    version: int = 1
    author_principal: str = Field(..., description="user:<id> | agent:<id> | service:<id>")
    agent_run_id: str | None = None
    parent_message_id: str | None = None
    attachments: list[Attachment] = Field(default_factory=list)
    source_system: str | None = None
    source_message_id: str | None = None
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    archive_status: ArchiveStatus = ArchiveStatus.STAGED
    system_metadata: dict[str, Any] = Field(default_factory=dict)
    custom_metadata: dict[str, Any] = Field(default_factory=dict)
    deleted_at: datetime | None = None


class AgentRun(BaseModel):
    """One execution of an agent within a turn. Carries independent lineage."""

    model_config = ConfigDict(extra="forbid")

    agent_run_id: str = Field(default_factory=lambda: new_id("agent_run"))
    tenant_id: str
    thread_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    work_id: str | None = None
    task_id: str | None = None
    agent_id: str
    agent_group_id: str | None = None
    parent_agent_run_id: str | None = None
    status: str = "RUNNING"
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    trace_id: str | None = None
    custom_metadata: dict[str, Any] = Field(default_factory=dict)
