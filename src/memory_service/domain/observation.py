"""Observations: what applications submit. The Memory Service decides what to do with them."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.enums import (
    Lifetime,
    MemoryType,
    ObservationKind,
    Visibility,
)
from memory_service.domain.ids import new_id


class ProcessingHints(BaseModel):
    """Expert-only overrides. Absent by default: the service decides."""

    model_config = ConfigDict(extra="forbid")

    lifetime: Lifetime | None = None
    memory_type: MemoryType | None = None
    visibility: Visibility | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    skip_extraction: bool = False
    skip_embedding: bool = False
    skip_graph: bool = False
    skip_summary: bool = False


class Observation(BaseModel):
    """'This happened or was learned.'"""

    model_config = ConfigDict(extra="forbid")

    observation_id: str = Field(default_factory=lambda: new_id("observation"))
    tenant_id: str
    kind: ObservationKind
    content: str = Field(
        default="", description="text payload (may be empty for file observations)"
    )
    content_hash: str
    # lineage snapshot (denormalized from the execution context at submit time)
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
    principal_id: str
    trace_id: str | None = None
    # what it refers to
    message_id: str | None = None
    document_id: str | None = None
    tool_run_id: str | None = None
    source_system: str | None = None
    source_id: str | None = None
    hints: ProcessingHints = ProcessingHints()
    custom_metadata: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    processed_at: datetime | None = None
