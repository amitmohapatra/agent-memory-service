"""Tool memory persistence: the catalog, invocation records, run outcomes, running statistics,
approval patterns and stored procedures.

Every read the request path makes is an indexed lookup bounded by a limit; the scans (mining,
graph edges) belong to the learning job and read the unlearned rows through a partial index.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from memory_service.domain.learning import ApprovalCounts
from memory_service.domain.tools import (
    RunOutcome,
    SkillDecision,
    StoredProcedure,
    ToolDescriptor,
    ToolInvocation,
    ToolStats,
)


@runtime_checkable
class ToolRepository(Protocol):
    # ------------------------------------------------------------------ catalog
    async def upsert(
        self, descriptor: ToolDescriptor, *, fields: frozenset[str] | None = None
    ) -> tuple[ToolDescriptor, bool]:
        """Insert or update by (tenant, workspace, name); an update changes only ``fields``
        (every catalog field when None), and a changed input schema bumps the version.
        Returns the stored entry and whether anything changed."""
        ...

    async def ensure(self, tenant_id: str, name: str) -> ToolDescriptor:
        """The entry a recorded call names, creating a bare tenant-wide one if none exists."""
        ...

    async def by_name(
        self, tenant_id: str, name: str, *, workspace_id: str | None = None
    ) -> ToolDescriptor | None:
        """The workspace's own entry for ``name``, else the tenant-wide one."""
        ...

    async def catalog(
        self,
        tenant_id: str,
        *,
        workspace_id: str | None,
        names: Sequence[str] | None = None,
        limit: int,
        after: str = "",
    ) -> list[ToolDescriptor]:
        """Entries visible in the workspace (its own shadow the tenant's), by name; ``after``
        is the last name of the previous page."""
        ...

    async def catalog_by_ids(
        self, tenant_id: str, tool_ids: Sequence[str]
    ) -> list[ToolDescriptor]: ...

    # ------------------------------------------------------------------ calls
    async def record(self, invocation: ToolInvocation) -> tuple[ToolInvocation, bool]:
        """Insert once; a retry of the same key returns the stored row and False."""
        ...

    async def invocations_for_run(
        self, tenant_id: str, run_id: str, *, scope_keys: Sequence[str] | None = None
    ) -> list[ToolInvocation]: ...

    async def for_pattern(
        self, tenant_id: str, audience: str, pattern: str, *, limit: int
    ) -> list[ToolInvocation]:
        """The newest calls of one task pattern recorded for one audience."""
        ...

    async def unlearned(self, *, tenant_id: str | None, limit: int) -> list[ToolInvocation]:
        """Calls the learning job has not processed yet, oldest first."""
        ...

    async def mark_learned(self, tenant_id: str, invocation_ids: Sequence[str]) -> None: ...

    # ------------------------------------------------------------------ outcomes
    async def set_outcome(self, outcome: RunOutcome) -> None:
        """Label a run; its calls are learned again, since the label changes what they teach."""
        ...

    async def outcome(self, tenant_id: str, run_id: str) -> RunOutcome | None: ...

    async def outcomes(self, tenant_id: str, run_ids: Sequence[str]) -> dict[str, RunOutcome]: ...

    # ------------------------------------------------------------------ statistics
    async def count_call(
        self, tenant_id: str, tool_name: str, *, ok: bool, latency_ms: float | None, at: datetime
    ) -> None: ...

    async def count_verdict(self, tenant_id: str, tool_name: str, verdict: str) -> None:
        """``verdict`` is ``approvals``, ``rejections`` or ``edits``."""
        ...

    async def stats(self, tenant_id: str, names: Sequence[str]) -> dict[str, ToolStats]: ...

    # ------------------------------------------------------------------ approvals
    async def count_approval(
        self, tenant_id: str, agent_id: str, tool_name: str, arg_shape: str, verdict: str
    ) -> None: ...

    async def approval_pattern(
        self, tenant_id: str, agent_id: str, tool_name: str, arg_shape: str
    ) -> ApprovalCounts | None: ...

    async def approval_patterns(
        self,
        tenant_id: str,
        agent_id: str,
        *,
        tool_name: str | None,
        min_support: int,
        limit: int,
        after: tuple[int, str, str] | None = None,
    ) -> list[ApprovalCounts]:
        """Patterns with at least ``min_support`` decisions, most supported first, then by
        tool and shape; ``after`` is the (support, tool, shape) of the previous page's last."""
        ...


@runtime_checkable
class ProcedureRepository(Protocol):
    async def get(self, tenant_id: str, procedure_id: str) -> StoredProcedure | None: ...

    async def by_pattern(
        self, tenant_id: str, scope_key: str, pattern: str
    ) -> StoredProcedure | None: ...

    async def save(self, procedure: StoredProcedure) -> None:
        """Insert or update the row of (tenant, scope, pattern)."""
        ...

    async def visible(
        self, tenant_id: str, scope_keys: Sequence[str], *, limit: int
    ) -> list[StoredProcedure]:
        """Active procedures whose audience the reader holds, most recently learned first."""
        ...

    async def reject(self, tenant_id: str, procedure_id: str) -> bool:
        """A reviewer rejected it: never offered again until its steps change."""
        ...

    async def active(self, tenant_id: str, *, limit: int) -> list[StoredProcedure]:
        """The tenant's active procedures, best supported first: the skill drafts' source."""
        ...

    async def decide(self, tenant_id: str, procedure_id: str, decision: SkillDecision) -> bool:
        """Record the reviewer's decision about its skill draft; False when there is no such
        procedure."""
        ...
