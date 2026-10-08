"""Canonical memory contract and its multi-dimensional classification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_service.domain.enums import (
    AdmissionVerdict,
    Lifetime,
    MemoryType,
    Representation,
    ScopeLevel,
    TemporalStatus,
    Visibility,
)
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.ids import new_id
from memory_service.domain.language import detect_language
from memory_service.domain.predicates import predicate_label

UNVERIFIED_MEMORY_CATEGORIES = ("contextual_fact", "assisted", "reflection")


def unverified_representation(metadata: Mapping[str, Any]) -> bool:
    """A model rewrite with source association, but no independently verified entailment."""
    return metadata.get("category") in UNVERIFIED_MEMORY_CATEGORIES or (
        metadata.get("provider") == "llm"
    )


#: Categories kept for good: the turns every other memory is evidence from, and the rules a
#: user said are standing.
LASTING_CATEGORIES = frozenset({"verbatim_turn", "rule"})
#: Kinds whose LONG_TERM memories are kept for good: what the user said about themselves.
LASTING_TYPES = frozenset({MemoryType.USER, MemoryType.PREFERENCE})


def lasting(category: str | None, memory_type: MemoryType, lifetime: Lifetime) -> bool:
    """Whether a memory is kept for good: never lapsing, and left alone by automatic
    forgetting however long it sits unused (``modules.memory.forgetting``). A user's turn
    needs no verbatim copy beside a reading of it in the same words that is lasting
    (``modules.memory.native``); any other reading can lapse, fade or be merged away."""
    return category in LASTING_CATEGORIES or (
        memory_type in LASTING_TYPES and lifetime is Lifetime.LONG_TERM
    )


class Scope(BaseModel):
    """Where a memory is anchored. Every identifier that is set narrows the scope."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: ScopeLevel
    tenant_id: str
    workspace_id: str | None = None
    user_id: str | None = None
    thread_id: str | None = None
    agent_id: str | None = None
    agent_group_id: str | None = None

    @model_validator(mode="after")
    def _level_has_anchor(self) -> Scope:
        required = {
            ScopeLevel.AGENT: "agent_id",
            ScopeLevel.AGENT_GROUP: "agent_group_id",
            ScopeLevel.THREAD: "thread_id",
            ScopeLevel.USER: "user_id",
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
            "thread_id",
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


class AdmissionDecision(BaseModel):
    """Why a candidate was (or was not) admitted as a memory. Stored with the memory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: AdmissionVerdict
    worthiness: float = Field(ge=0.0, le=1.0, description="type prior blended with confidence")
    novelty: float = Field(ge=0.0, le=1.0, description="from the consolidation decision")
    confidence: float = Field(ge=0.0, le=1.0)
    expected_utility: float = Field(ge=0.0, le=1.0, description="lifetime x importance x recency")
    score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    decided_at: datetime


DERIVED_MEMORY_TYPES: frozenset[MemoryType] = frozenset(
    {MemoryType.OBSERVATION, MemoryType.BELIEF, MemoryType.ENTITY_SUMMARY}
)


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
    access_count: int = Field(default=0, ge=0, description="recall hits; drives forgetting")
    last_accessed_at: datetime | None = None

    system_metadata: dict[str, Any] = Field(default_factory=dict)
    custom_metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    revision: int = 1
    deleted_at: datetime | None = None
    #: ISO 639-1 code of ``content`` (``domain.language``); derived when not given
    lang: str = ""

    @model_validator(mode="after")
    def _custom_type_consistency(self) -> CanonicalMemory:
        if self.memory_type is MemoryType.CUSTOM and not self.custom_type:
            raise ValueError("custom_type is required when memory_type=CUSTOM")
        if self.scope.tenant_id != self.tenant_id:
            raise ValueError("scope.tenant_id must match tenant_id")
        if not self.lang:
            self.lang = detect_language(self.content)
        return self


class MemoryResult(BaseModel):
    """A memory as returned to callers, with retrieval metadata."""

    model_config = ConfigDict(frozen=True)

    memory: CanonicalMemory
    score: float | None = None
    retrievers: list[str] = Field(default_factory=list, description="which strategies hit")
    reason: str | None = None


def dated_statement(day: str, text: str) -> str:
    """One source statement kept beside the day it was observed.

    Every relative date inside ``text`` ("last Tuesday") is only interpretable against this
    day, so the two never travel apart.
    """
    return f"[observed {day}] {text}"


def aggregate_statement(subject: str, predicate: str, dated: Sequence[tuple[str, str]]) -> str:
    """The several current values of one multi-valued slot, as one block.

    The shape the renderer groups a bundle into (``domain.context_bundle``): one heading,
    every value dated under it.

    ``dated`` is ``(day, text)`` oldest first; duplicates collapse, order is preserved.
    """
    body = "\n".join(dict.fromkeys(dated_statement(day, text) for day, text in dated))
    return f"{subject} — {predicate_label(predicate)} (source statements):\n{body}"
