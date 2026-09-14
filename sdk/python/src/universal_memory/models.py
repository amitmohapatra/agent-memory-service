"""SDK-facing models. Mirrors the public API contract; no internal types leak here."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Scope(BaseModel):
    """Identity and lineage for one request. Built by ``MemoryClient.bind``."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    workspace_id: str | None = None
    user_id: str | None = None
    group_ids: list[str] = Field(default_factory=list)
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
    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str
    checksum: str
    size_bytes: int
    job_id: str | None = None
    deduplicated: bool = False


class JobHandle(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    status: str
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
    memory_type: str
    lifetime: str
    visibility: str
    score: float | None = None
    confidence: float | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)


class ContextItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    item_id: str
    representation: str
    text: str
    score: float = 0.0
    citation: str
    document_id: str | None = None
    page: int | None = None
    section_path: str | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)


class ConversationWindow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    thread_id: str | None = None
    message_ids: list[str] = Field(default_factory=list)
    rendered: str = ""
    summary: str | None = None


class EvidenceReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    status: Literal["COMPLETE", "INCOMPLETE", "INSUFFICIENT"]
    missing_groups: list[str] = Field(default_factory=list)
    escalations: list[str] = Field(default_factory=list)


class ContextBundle(BaseModel):
    """Bounded, ranked context for the current turn. ``rendered`` is ready to prompt with."""

    model_config = ConfigDict(frozen=True, extra="allow")

    query: str
    query_type: str
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


class ThreadInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    thread_id: str
    tenant_id: str
    title: str | None = None
    created_at: datetime | None = None


class MessageInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    message_id: str
    role: str
    kind: str
    sequence: int
    content: str
    occurred_at: datetime | None = None
