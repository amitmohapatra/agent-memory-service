"""The pinned sections of a pushed context: the profile blocks, the thread's durable summary,
the procedures learned for the task and the tool hints (both only for an agent with tools).

Each is one indexed read (the procedures a bounded one; the people's pages, from their
beliefs, one more) and they run concurrently with
retrieval, so they add a round trip, not a stage. The pinned sections are budgeted first: they
may take at most ``PINNED_SHARE`` of the token budget, in priority order, each costed as the
text it renders to. The thread summary is truncated to what is left rather than dropped.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.beliefs import Belief
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.context_bundle import (
    PEOPLE_HEADING,
    ContextItem,
    ProfileBlockView,
    SkillView,
    ThreadSummaryView,
    people_section,
    profile_section,
    skills_section,
    summary_section,
    tools_section,
)
from memory_service.domain.tools import StoredProcedure, ToolHints
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.ingestion.hierarchy import estimate_tokens
from memory_service.modules.profile.service import PersonPage
from memory_service.modules.retrieval.engine import QueryVectors
from memory_service.modules.tools.matching import TaskVectors
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
#: How the people's pages (beliefs) are paid for, the one place it is decided (design note
#: §3.7). True: out of the context's token budget, after every other pinned section within
#: ``PINNED_SHARE``, so the memories get what is left. False: on top of it - the pages still
#: fit ``PINNED_SHARE`` of the budget on their own, and the memories keep their whole budget.
#: False, accuracy first: measured on LoCoMo, the memories stop at a count (about 100), not
#: at the token budget, so charging the pages pushed no memory out, and it would once the
#: memories reach the budget.
PEOPLE_CHARGED: Final = False
#: the profile sections ``modules.context.standing`` writes (kept first when the whole
#: profile does not fit)
STANDING_BLOCKS: Final = frozenset({"standing_rules", "preferences", "about_user"})
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
        names = ",".join(sorted(set(self.available))) if self.available is not None else "*"
        # the default request keeps its key; only an opt-out of hints is a different bundle
        return f"tools:{self.k}:{names}" + ("" if self.hints else ":nohints")


@dataclass
class Pinned:
    profile: list[ProfileBlockView] = field(default_factory=list)
    #: a page per person with beliefs, pinned after everything else within the share left
    people: list[PersonPage] = field(default_factory=list)
    #: the pages of ``people`` that fit (``within_budget``)
    pages: list[ProfileBlockView] = field(default_factory=list)
    thread_summary: ThreadSummaryView | None = None
    procedures: list[SkillView] = field(default_factory=list)
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


def _profile_that_fits(
    profile: list[ProfileBlockView], allowed: int
) -> tuple[list[ProfileBlockView], int]:
    """The whole profile when it fits, else the rules and preferences that apply to this
    request (``STANDING_BLOCKS``) when they do, else nothing; and the tokens it takes."""
    for blocks in (profile, [b for b in profile if b.block in STANDING_BLOCKS]):
        if (text := profile_section(blocks)) and (cost := estimate_tokens(text)) <= allowed:
            return blocks, cost
    return [], 0


def within_budget(
    pinned: Pinned, budget: int, relevance: Mapping[str, int] | None = None
) -> tuple[Pinned, int]:
    """The pinned sections that fit ``PINNED_SHARE`` of the budget, in priority order, and
    the tokens they take from it (the people's pages only while ``PEOPLE_CHARGED``). A
    section is costed as the text it renders to; one that does not fit is dropped, except
    the thread summary, which keeps its newest part."""
    allowed, used = int(budget * PINNED_SHARE), 0
    kept = Pinned()
    kept.profile, cost = _profile_that_fits(pinned.profile, allowed)
    used += cost
    if (text := summary_section(pinned.thread_summary)) and pinned.thread_summary is not None:
        cost = estimate_tokens(text)
        if used + cost <= allowed:
            kept.thread_summary, used = pinned.thread_summary, used + cost
        elif (cut := _truncated(pinned.thread_summary, allowed - used)) is not None:
            kept.thread_summary = cut
            used += estimate_tokens(summary_section(cut) or "")
    text = skills_section(pinned.procedures, pinned.tools)
    if text and used + estimate_tokens(text) <= allowed:
        kept.procedures, used = pinned.procedures, used + estimate_tokens(text)
    if (text := tools_section(pinned.tools)) and used + estimate_tokens(text) <= allowed:
        kept.tools, used = pinned.tools, used + estimate_tokens(text)
    elif pinned.tools is not None:
        # the candidates still narrow the caller's tools; only the rendered section is cut
        kept.tools = pinned.tools.model_copy(update={"next": None, "prefill": {}, "missing": []})
    spare = allowed - used if PEOPLE_CHARGED else allowed
    kept.pages = _people_that_fit(pinned.people, spare, relevance or {})
    if PEOPLE_CHARGED:
        used += _people_cost(kept.pages)
    return kept, used


def _people_cost(pages: Sequence[ProfileBlockView]) -> int:
    text = people_section(pages)
    return estimate_tokens(text) if text else 0


def _people_that_fit(
    people: Sequence[PersonPage], spare: int, relevance: Mapping[str, int]
) -> list[ProfileBlockView]:
    """Every person's whole page when the pages fit the ``spare`` tokens of the pinned share
    the rest left; else the lines worth most to this question (``_chosen``), every facet kept
    while any line can go instead. Never a cut at the end and never a character cap: a page
    holds as many lines as the budget does."""
    pages = [v for p in people if (v := p.view()) is not None]
    if _people_cost(pages) <= spare:
        return pages
    share = spare - estimate_tokens(f"## {PEOPLE_HEADING}")
    for _ in range(3):  # per-line estimates against the section's: a token or two off
        pages = [v for p in people if (v := p.view(_chosen(people, share, relevance)))]
        over = _people_cost(pages) - spare
        if over <= 0:
            return pages
        share -= over
    return []


def _worth(b: Belief, relevance: Mapping[str, int]) -> tuple[bool, int, int, datetime, str]:
    """How much a line is worth to this question: a hit of its search first (by rank), then
    the most sources, then the newest."""
    hit = b.belief_id in relevance
    return hit, -relevance.get(b.belief_id, 0), b.proof_count, b.last_seen_at, b.belief_id


def _chosen(people: Sequence[PersonPage], room: int, relevance: Mapping[str, int]) -> set[str]:
    """The current lines to show in ``room`` tokens: earlier values go first, then the line
    worth least (``_worth``) - a facet's last line only once no other line can go."""
    lines = [(p.name, b) for p in people for b in p.current]
    cost = {b.belief_id: estimate_tokens(f"- {b.text}") + 1 for _, b in lines}
    left: dict[tuple[str, str], int] = {}
    for name, b in lines:
        left[(name, b.facet)] = left.get((name, b.facet), 0) + 1
    total = sum(estimate_tokens(f"### {p.name}") + 1 for p in people) + sum(cost.values())
    keep = set(cost)
    order = sorted(lines, key=lambda nb: _worth(nb[1], relevance))
    for protect in (True, False):
        for name, b in order:
            if total <= room:
                return keep
            if b.belief_id in keep and not (protect and left[(name, b.facet)] == 1):
                keep.discard(b.belief_id)
                left[(name, b.facet)] -= 1
                total -= cost[b.belief_id]
    return keep


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
        vectors: TaskVectors = None,
    ) -> Pinned:
        """Everything but the tool hints, concurrently: the procedures only for an agent with
        tools. The thread summary comes with or without the window: a framework that keeps
        its own history keeps the recent messages, not what was said before them.
        ``vectors``: the query's, as the retrieval running beside this encodes them - read
        only for a procedure the query's words miss."""
        found_profile, found_people, found_summary, found_procedures = await asyncio.gather(
            self._profile(ctx),
            self._people(ctx, visibility),
            self._summary(ctx),
            self.services["tool_hints"].procedures(
                ctx, query, list(visibility.keys), k=CONTEXT_PROCEDURES, vectors=vectors
            )
            if procedures
            else _empty(),
        )
        return Pinned(
            profile=found_profile,
            people=found_people,
            thread_summary=found_summary,
            procedures=[skill_view(p) for p in found_procedures],
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

    async def _people(
        self, ctx: MemoryExecutionContext, visibility: VisibilitySpecification
    ) -> list[PersonPage]:
        """The people's pages, where a model can be reached (beliefs are written nowhere
        else): without one the read is today's, round trip for round trip."""
        if not self.services["llm_assist"].available:
            return []
        async with self.uow_factory() as uow:
            return await self.services["profile"].people(uow, ctx, sorted(visibility.keys))

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


async def _empty() -> list[StoredProcedure]:
    return []
