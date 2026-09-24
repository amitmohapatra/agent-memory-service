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
    #: Required alongside ``memory_type=CUSTOM``, which carries a caller-defined taxonomy.
    #:
    #: Without this field CUSTOM was unreachable: ``CanonicalMemory`` rejects it when
    #: ``custom_type`` is absent, and there was no way to supply one through the import
    #: surface — so every observation hinted CUSTOM produced a failed job and no memory,
    #: silently, while the type sat in the admission table looking supported.
    custom_type: str | None = None
    visibility: Visibility | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    skip_extraction: bool = False
    skip_embedding: bool = False
    skip_graph: bool = False
    skip_summary: bool = False


#: Roles whose text is the system talking rather than the human. A message carrying no role
#: is treated as the human's: the rules that consult this exist to keep an agent's own notes
#: out of the user's memory, and guessing wrong in that direction loses a user fact.
AGENT_ROLES = frozenset({"AGENT", "ASSISTANT", "TOOL", "SYSTEM"})
AGENT_KINDS = frozenset({ObservationKind.AGENT_RESULT, ObservationKind.TOOL_RESULT})


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

    @property
    def agent_authored(self) -> bool:
        """Whether the agent wrote this text, as opposed to merely relaying it.

        Two rules used to ask this question and each answered it differently by reading
        ``agent_id`` - which is true of any request carrying agent lineage, including a
        human's own turn posted by an agent harness. One of them re-typed the human's
        preferences into the agent's private memory; the other refused to keep the turn
        verbatim. The role was recorded here the whole time with nothing reading it.
        """
        if self.kind in AGENT_KINDS:
            return True
        metadata = self.custom_metadata or {}
        if str(metadata.get("kind", "")).upper() == "INTERNAL":
            return True
        return str(metadata.get("role", "")).upper() in AGENT_ROLES
