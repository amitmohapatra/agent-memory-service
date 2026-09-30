"""The pinned sections of a pushed context: the profile blocks, the thread's durable summary,
the procedures learned for the task, the items prefetched from what earlier pulls used, and -
when asked - the tool hints.

Each is one indexed read (the procedures a bounded one) and they run concurrently with
retrieval, so they add a round trip, not a stage. The pinned sections are budgeted first: they
may take at most ``PINNED_SHARE`` of the token budget, in priority order.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.context_bundle import (
    ContextItem,
    ProcedureView,
    ProfileBlockView,
    ThreadSummaryView,
)
from memory_service.domain.enums import TemporalStatus
from memory_service.domain.tools import StoredProcedure, ToolHints
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.ingestion.hierarchy import estimate_tokens
from memory_service.modules.retrieval.engine import Candidate, memory_candidate
from memory_service.ports.uow import UnitOfWorkFactory

#: Procedures a context carries.
CONTEXT_PROCEDURES: Final = 3
#: The share of the token budget the pinned sections may take, all together.
PINNED_SHARE: Final = 0.5


class ToolsRequest(BaseModel):
    """``tools`` of a context request: hints for these callable tools (None: any)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    available: list[str] | None = Field(default=None, max_length=500)
    k: int = Field(default=8, ge=1, le=20)

    def fingerprint(self) -> str:
        names = ",".join(sorted(self.available)) if self.available is not None else "*"
        return f"tools:{self.k}:{names}"


@dataclass
class Pinned:
    profile: list[ProfileBlockView] = field(default_factory=list)
    thread_summary: ThreadSummaryView | None = None
    procedures: list[ProcedureView] = field(default_factory=list)
    prefetched: list[Candidate] = field(default_factory=list)
    tools: ToolHints | None = None

    @property
    def covered_to(self) -> int:
        return self.thread_summary.covers_to_sequence if self.thread_summary else 0


def _procedure_view(p: StoredProcedure) -> ProcedureView:
    return ProcedureView(
        id=p.procedure_id,
        title=p.title,
        steps=p.steps,
        success_rate=p.success_rate,
        support=p.support,
    )


def _tokens(pinned: Pinned) -> list[tuple[str, int]]:
    summary = pinned.thread_summary
    tools = pinned.tools
    return [
        ("profile", sum(estimate_tokens(b.text) + 4 for b in pinned.profile)),
        ("thread_summary", estimate_tokens(summary.text) if summary else 0),
        ("procedures", sum(estimate_tokens(p.title) + 8 * len(p.steps) for p in pinned.procedures)),
        ("tools", estimate_tokens(tools.model_dump_json()) // 2 if tools else 0),
    ]


def within_budget(pinned: Pinned, budget: int) -> tuple[Pinned, int]:
    """The pinned sections that fit ``PINNED_SHARE`` of the budget, in priority order, and
    the tokens they take."""
    allowed, used = int(budget * PINNED_SHARE), 0
    empty = {"profile": [], "thread_summary": None, "procedures": [], "tools": None}
    kept: dict[str, Any] = {}
    for name, cost in _tokens(pinned):
        if cost and used + cost <= allowed:
            kept[name] = getattr(pinned, name)
            used += cost
        else:
            kept[name] = empty[name] if cost else getattr(pinned, name)
    return Pinned(**kept, prefetched=pinned.prefetched), used


class ContextSections:
    def __init__(self, uow_factory: UnitOfWorkFactory, services: dict[str, Any]) -> None:
        self.uow_factory = uow_factory
        self.services = services

    async def gather(
        self, ctx: MemoryExecutionContext, query: str, visibility: VisibilitySpecification
    ) -> Pinned:
        """Everything but the tool hints, concurrently."""
        profile, summary, procedures, prefetched = await asyncio.gather(
            self._profile(ctx),
            self._summary(ctx),
            self.services["tool_hints"].procedures(
                ctx, query, list(visibility.keys), k=CONTEXT_PROCEDURES
            ),
            self._prefetched(ctx, query, visibility),
        )
        return Pinned(
            profile=profile,
            thread_summary=summary,
            procedures=[_procedure_view(p) for p in procedures],
            prefetched=prefetched,
        )

    async def tools(
        self,
        ctx: MemoryExecutionContext,
        query: str,
        request: ToolsRequest,
        visibility: VisibilitySpecification,
        *,
        memories: Sequence[ContextItem],
        profile: Sequence[ProfileBlockView],
    ) -> ToolHints:
        return await self.services["tool_hints"].hints(
            ctx,
            query,
            available=request.available,
            k=request.k,
            scope_keys=list(visibility.keys),
            memories=memories,
            profile=profile,
        )

    async def _profile(self, ctx: MemoryExecutionContext) -> list[ProfileBlockView]:
        async with self.uow_factory() as uow:
            blocks = await self.services["profile"].blocks(uow, ctx)
        return [ProfileBlockView(block=b.block, text=b.text, version=b.version) for b in blocks]

    async def _summary(self, ctx: MemoryExecutionContext) -> ThreadSummaryView | None:
        if not ctx.thread_id:
            return None
        async with self.uow_factory() as uow:
            summary = await uow.summaries.latest(ctx.tenant_id, ctx.thread_id)
        if summary is None:
            return None
        return ThreadSummaryView(
            text=summary.text,
            covers_to_sequence=summary.covers_to_sequence,
            version=summary.version,
        )

    async def _prefetched(
        self, ctx: MemoryExecutionContext, query: str, visibility: VisibilitySpecification
    ) -> list[Candidate]:
        """Memories this principal's pulls for requests like this one kept using, still
        current and still visible to it."""
        ids = await self.services["agent_tools"].prefetched(ctx, query)
        if not ids:
            return []
        async with self.uow_factory() as uow:
            memories = await uow.memories.get_many(ctx.tenant_id, ids)
        return [
            memory_candidate(m, retriever="prefetch", score=1.0)
            for m in memories
            if m.temporal.status is TemporalStatus.CURRENT
            and m.deleted_at is None
            and visibility.allows(m.tenant_id, m.system_metadata.get("visibility_keys", []))
        ]
