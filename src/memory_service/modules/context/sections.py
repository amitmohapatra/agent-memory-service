"""The pinned sections of a pushed context: the profile blocks, the thread's durable summary,
the procedures learned for the task and the tool hints (both only for an agent with tools),
and the items prefetched from what earlier pulls used.

Each is one indexed read (the procedures a bounded one) and they run concurrently with
retrieval, so they add a round trip, not a stage. The pinned sections are budgeted first: they
may take at most ``PINNED_SHARE`` of the token budget, in priority order, each costed as the
text it renders to. The thread summary is truncated to what is left rather than dropped.
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
    ProfileBlockView,
    SkillView,
    ThreadSummaryView,
    profile_section,
    skills_section,
    summary_section,
    tools_section,
)
from memory_service.domain.enums import TemporalStatus
from memory_service.domain.tools import StoredProcedure, ToolHints
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.ingestion.hierarchy import estimate_tokens
from memory_service.modules.retrieval.engine import Candidate, QueryVectors, memory_candidate
from memory_service.modules.tools.skills import skill_view
from memory_service.ports.uow import UnitOfWorkFactory

#: Learned skills a context carries.
CONTEXT_PROCEDURES: Final = 3
#: The smallest toolbox the context hints tools for: below it the model already sees every
#: tool, and the hints (read after retrieval) would only add latency. The learned skills are
#: offered to any agent with tools.
TOOL_HINTS_MIN: Final = 5
#: The share of the token budget the pinned sections may take, all together.
PINNED_SHARE: Final = 0.5
#: A summary truncated below this many tokens says too little to keep.
SUMMARY_MIN_TOKENS: Final = 40
#: Characters per token when a summary is cut to fit (estimate_tokens' own ratio).
CHARS_PER_TOKEN: Final = 4


class ToolsRequest(BaseModel):
    """``tools`` of a context request: the agent's callable tools (None: any catalog tool).
    An agent with tools gets the skills it learned for the task; with five or more (or the
    catalog), the tool hints too."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    available: list[str] | None = Field(
        default=None,
        max_length=500,
        description="The tools the agent can call (at most 500); null: any catalog tool; "
        "empty: none.",
    )
    k: int = Field(default=8, ge=1, le=20, description="The most tools to hint (1-20).")
    hints: bool = Field(
        default=True,
        description="false: only the skills the agent learned, no tool hints (which are read "
        "after retrieval): for a caller that offers the model every tool anyway.",
    )

    @property
    def any(self) -> bool:
        """Whether the agent has any tool (an empty list: none)."""
        return self.available is None or bool(self.available)

    @property
    def hinted(self) -> bool:
        """Whether the tools are hinted: asked for, and any catalog tool or a toolbox of
        ``TOOL_HINTS_MIN`` or more."""
        return self.hints and (self.available is None or len(self.available) >= TOOL_HINTS_MIN)

    def fingerprint(self) -> str:
        names = ",".join(sorted(self.available)) if self.available is not None else "*"
        # the default request keeps its key; only an opt-out of hints is a different bundle
        return f"tools:{self.k}:{names}" + ("" if self.hints else ":nohints")


@dataclass
class Pinned:
    profile: list[ProfileBlockView] = field(default_factory=list)
    thread_summary: ThreadSummaryView | None = None
    procedures: list[SkillView] = field(default_factory=list)
    prefetched: list[Candidate] = field(default_factory=list)
    tools: ToolHints | None = None

    @property
    def covered_to(self) -> int:
        return self.thread_summary.covers_to_sequence if self.thread_summary else 0


def _truncated(summary: ThreadSummaryView, tokens: int) -> ThreadSummaryView | None:
    """The summary's newest part whose section fits ``tokens``, or None when too little
    would fit. Cut by characters, then shortened until the estimate agrees (a script the
    estimate counts by bytes costs more than four characters a token)."""
    if tokens < SUMMARY_MIN_TOKENS:
        return None
    chars = tokens * CHARS_PER_TOKEN
    while chars > 0:
        cut = summary.model_copy(update={"text": "..." + summary.text[-chars:].lstrip()})
        if estimate_tokens(summary_section(cut) or "") <= tokens:
            return cut
        chars = chars * 9 // 10
    return None


def within_budget(pinned: Pinned, budget: int) -> tuple[Pinned, int]:
    """The pinned sections that fit ``PINNED_SHARE`` of the budget, in priority order, and
    the tokens they take. A section is costed as the text it renders to; one that does not
    fit is dropped, except the thread summary, which keeps its newest part."""
    allowed, used = int(budget * PINNED_SHARE), 0
    kept = Pinned(prefetched=pinned.prefetched)
    if (text := profile_section(pinned.profile)) and (cost := estimate_tokens(text)) <= allowed:
        kept.profile, used = pinned.profile, used + cost
    if (text := summary_section(pinned.thread_summary)) and pinned.thread_summary is not None:
        cost = estimate_tokens(text)
        if used + cost <= allowed:
            kept.thread_summary, used = pinned.thread_summary, used + cost
        elif (cut := _truncated(pinned.thread_summary, allowed - used)) is not None:
            kept.thread_summary = cut
            used += estimate_tokens(summary_section(cut) or "")
    if (text := skills_section(pinned.procedures)) and used + estimate_tokens(text) <= allowed:
        kept.procedures, used = pinned.procedures, used + estimate_tokens(text)
    if (text := tools_section(pinned.tools)) and used + estimate_tokens(text) <= allowed:
        kept.tools, used = pinned.tools, used + estimate_tokens(text)
    elif pinned.tools is not None:
        # the candidates still narrow the caller's tools; only the rendered section is cut
        kept.tools = pinned.tools.model_copy(update={"next": None, "prefill": {}, "missing": []})
    return kept, used


class ContextSections:
    def __init__(self, uow_factory: UnitOfWorkFactory, services: dict[str, Any]) -> None:
        self.uow_factory = uow_factory
        self.services = services

    async def gather(
        self,
        ctx: MemoryExecutionContext,
        query: str,
        visibility: VisibilitySpecification,
        *,
        procedures: bool,
    ) -> Pinned:
        """Everything but the tool hints, concurrently: the procedures only for an agent with
        tools. The thread summary comes with or without the window: a framework that keeps
        its own history keeps the recent messages, not what was said before them."""
        found_profile, found_summary, found_procedures, prefetched = await asyncio.gather(
            self._profile(ctx),
            self._summary(ctx),
            self.services["tool_hints"].procedures(
                ctx, query, list(visibility.keys), k=CONTEXT_PROCEDURES
            )
            if procedures
            else _empty(),
            self._prefetched(ctx, query, visibility),
        )
        return Pinned(
            profile=found_profile,
            thread_summary=found_summary,
            procedures=[skill_view(p) for p in found_procedures],
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
        vectors: QueryVectors | None,
    ) -> ToolHints:
        return await self.services["tool_hints"].hints(
            ctx,
            query,
            available=request.available,
            k=request.k,
            scope_keys=list(visibility.keys),
            memories=memories,
            profile=profile,
            vectors=vectors,
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


async def _empty() -> list[StoredProcedure]:
    return []
