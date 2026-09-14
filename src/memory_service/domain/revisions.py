"""Revision-based cache invalidation.

Every mutation increments the revisions it affects. Cache keys embed the relevant
revisions, so stale entries simply stop being addressed; no cache scans, no deletion
storms. The revision counters live in PostgreSQL (canonical) with a Dragonfly read cache.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RevisionKind(StrEnum):
    TENANT = "tenant"
    WORKSPACE = "workspace"
    GROUP = "group"
    USER = "user"
    THREAD = "thread"
    DOCUMENT = "document"
    GRAPH = "graph"
    AGENT = "agent"


class RevisionKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: RevisionKind
    tenant_id: str
    object_id: str = Field(default="", description="empty for tenant-level revision")

    def as_string(self) -> str:
        return f"rev:{self.tenant_id}:{self.kind}:{self.object_id}"


class RevisionSet(BaseModel):
    """Snapshot of the revisions a cached object was built from."""

    model_config = ConfigDict(frozen=True)

    values: dict[str, int] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        from memory_service.domain.ids import stable_key

        return stable_key(*(f"{k}={v}" for k, v in sorted(self.values.items())))
