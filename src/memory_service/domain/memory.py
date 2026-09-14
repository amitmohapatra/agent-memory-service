"""Canonical memory contract and its multi-dimensional classification."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_service.domain.enums import (
    Lifetime,
    MemoryType,
    Representation,
    ScopeLevel,
    TemporalStatus,
    Visibility,
)
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.ids import new_id


class Scope(BaseModel):
    """Where a memory is anchored. Every identifier that is set narrows the scope."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: ScopeLevel
    tenant_id: str
    workspace_id: str | None = None
    user_id: str | None = None
    group_id: str | None = None
    thread_id: str | None = None
    work_id: str | None = None
    agent_id: str | None = None
    agent_group_id: str | None = None

    @model_validator(mode="after")
    def _level_has_anchor(self) -> Scope:
        required = {
            ScopeLevel.AGENT: "agent_id",
            ScopeLevel.AGENT_GROUP: "agent_group_id",
            ScopeLevel.WORK: "work_id",
            ScopeLevel.THREAD: "thread_id",
            ScopeLevel.USER: "user_id",
            ScopeLevel.GROUP: "group_id",
            ScopeLevel.WORKSPACE: "workspace_id",
        }
        field = required.get(self.level)
        if field and getattr(self, field) is None:
            raise ValueError(f"scope level {self.level} requires {field}")
        return self

    def key(self) -> str:
        """Canonical string form used for indexing and cache keys."""
        parts = [f"t={self.tenant_id}", f"l={self.level}"]
        for name in (
            "workspace_id",
            "user_id",
            "group_id",
            "thread_id",
            "work_id",
            "agent_id",
            "agent_group_id",
        ):
            value = getattr(self, name)
            if value:
                parts.append(f"{name[:-3]}={value}")
        return "/".join(parts)


class TemporalState(BaseModel):
    """Bitemporal validity: when the fact was true and when we knew it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: TemporalStatus = TemporalStatus.CURRENT
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    observed_at: datetime
    superseded_by: str | None = Field(default=None, description="memory_id that replaced this one")
    supersedes: str | None = Field(default=None, description="memory_id this one replaced")
    contradicts: list[str] = Field(default_factory=list)

    def is_current_at(self, when: datetime) -> bool:
        if self.status not in (TemporalStatus.CURRENT,):
            return False
        if self.valid_from and when < self.valid_from:
            return False
        return not (self.valid_to and when >= self.valid_to)


class CanonicalMemory(BaseModel):
    """A durable unit of intelligence derived from raw evidence. Never replaces the evidence."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(default_factory=lambda: new_id("memory"))
    tenant_id: str
    scope: Scope
    visibility: Visibility
    owner_principal: str = Field(..., description="user:<id> | agent:<id> | service:<id>")

    lifetime: Lifetime
    memory_type: MemoryType
    custom_type: str | None = Field(
        default=None, description="plugin type name when memory_type=CUSTOM"
    )
    representation: Representation = Representation.MEMORY

    content: str = Field(..., min_length=1, description="Compact canonical text.")
    normalized_hash: str = Field(..., description="Hash of normalized content for dedup.")
    subject: str | None = Field(default=None, description="Entity the memory is about (canonical).")
    predicate: str | None = None
    object: str | None = None

    temporal: TemporalState
    evidence: list[EvidenceRef] = Field(..., min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    reinforcement_count: int = Field(default=1, ge=1)

    system_metadata: dict[str, Any] = Field(default_factory=dict)
    custom_metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    revision: int = 1
    deleted_at: datetime | None = None

    @model_validator(mode="after")
    def _custom_type_consistency(self) -> CanonicalMemory:
        if self.memory_type is MemoryType.CUSTOM and not self.custom_type:
            raise ValueError("custom_type is required when memory_type=CUSTOM")
        if self.scope.tenant_id != self.tenant_id:
            raise ValueError("scope.tenant_id must match tenant_id")
        return self


class MemoryResult(BaseModel):
    """A memory as returned to callers, with retrieval metadata."""

    model_config = ConfigDict(frozen=True)

    memory: CanonicalMemory
    score: float | None = None
    retrievers: list[str] = Field(default_factory=list, description="which strategies hit")
    reason: str | None = None
