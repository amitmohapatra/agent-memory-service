"""AuthorizationProvider port (relationship-based access control).

OpenFGA answers "who may access this?". The Memory Service enforces its own boundary
regardless of what upstream authenticated: every retrieval is scope-filtered *before*
any model sees data, and the filter is derived from these decisions.

Object types mirror the OpenFGA model in ``deploy/openfga/model.fga``:
tenant, workspace, group, user, document, thread, memory, agent, work.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.context import MemoryExecutionContext


class RelationTuple(BaseModel):
    """``user`` has ``relation`` on ``object`` (e.g. user:u1 member group:legal)."""

    model_config = ConfigDict(frozen=True)

    user: str = Field(..., description="'user:u1' | 'agent:a1' | 'group:legal#member'")
    relation: str
    object: str = Field(..., description="'document:d1' | 'thread:t1' | 'tenant:acme'")


class AccessCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    user: str
    relation: str
    object: str


class AuthorizedScope(BaseModel):
    """The set of scope anchors the principal may read. Used to build search filters.

    Lists are bounded; when a list would exceed ``max_objects`` the provider sets
    ``truncated=True`` and callers must fall back to per-object checks.
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    principal: str
    workspace_ids: list[str] = Field(default_factory=list)
    group_ids: list[str] = Field(default_factory=list)
    thread_ids: list[str] = Field(default_factory=list)
    work_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    agent_ids: list[str] = Field(default_factory=list)
    agent_group_ids: list[str] = Field(default_factory=list)
    user_id: str | None = None
    truncated: bool = False

    def fingerprint(self) -> str:
        from memory_service.domain.ids import stable_key

        return stable_key(
            self.tenant_id,
            self.principal,
            ",".join(sorted(self.workspace_ids)),
            ",".join(sorted(self.group_ids)),
            ",".join(sorted(self.thread_ids)),
            ",".join(sorted(self.work_ids)),
            ",".join(sorted(self.document_ids)),
            ",".join(sorted(self.agent_ids)),
            ",".join(sorted(self.agent_group_ids)),
            self.user_id or "",
            str(self.truncated),
        )


@runtime_checkable
class AuthorizationProvider(Protocol):
    async def check(self, check: AccessCheck) -> bool: ...

    async def batch_check(self, checks: Sequence[AccessCheck]) -> list[bool]: ...

    async def write(
        self, add: Sequence[RelationTuple], delete: Sequence[RelationTuple] = ()
    ) -> None: ...

    async def list_objects(self, user: str, relation: str, object_type: str) -> list[str]: ...

    async def authorized_scope(self, ctx: MemoryExecutionContext) -> AuthorizedScope:
        """Resolve everything the context's principal may read. Cached by revision."""
        ...

    async def ping(self) -> bool: ...
