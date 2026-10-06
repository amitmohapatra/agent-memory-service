"""Search: ranked items for a query, in the one shape ``POST /v1/recall`` and the
``memory_search`` agent tool both return.

An item is ``{id, kind, text, observed_on, citation, document_id?, page?}``: what a caller
needs to use a result and cite it, nothing else. Ranking detail (scores, retrievers,
attributes) is attached only when asked for (``debug``).

Kinds: ``memory`` (what was learned or stated), ``chunk`` (document passages), ``summary``
(document summaries), ``episode`` (earlier conversations, one per thread: its summary and a
digest of what followed, across every thread the caller's user owns) and ``message`` (this
messages of this conversation and of the user's earlier ones, this conversation's first;
another user's are never read). A time range keeps only what was
observed within it, and it filters before anything is ranked: the store applies it to
memories, and a message outside it is never scored.
"""

from __future__ import annotations

import asyncio
import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.conversation import Message
from memory_service.domain.enums import MessageKind, QueryType
from memory_service.modules.conversation.service import ConversationService
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.retrieval.engine import (
    Candidate,
    ObservedRange,
    PointInTime,
    RetrievalEngine,
    RetrievalResult,
)
from memory_service.ports.uow import UnitOfWorkFactory

SearchKind = Literal["memory", "chunk", "summary", "episode", "message"]
#: What a search reads when the caller names no kinds. The order is the order the engine
#: interleaves the per-kind rankings in, so document passages lead (as /v1/recall always did).
DEFAULT_KINDS: Final[tuple[SearchKind, ...]] = ("chunk", "memory")
#: The newest messages of the thread one search scores (an indexed, bounded read).
HISTORY_SCAN: Final = 200
#: The newest messages of the user's earlier conversations that share a word with the query
#: one search scores (a time range moves the window: the bound applies within it), read from
#: the user's most recently active threads (every message appended makes its thread active).
OWNED_SCAN: Final = 200
OWNED_THREADS: Final = 100
#: The query's longest words the database filters earlier conversations by.
OWNED_WORDS: Final = 16
_WORD: Final = re.compile(r"\w+")


class SearchItem(BaseModel):
    """One result: enough to use it and cite it."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(
        description="The item's id (mem_..., chk_..., a summary or a message id): what to "
        "cite and to fetch."
    )
    kind: SearchKind = Field(
        description="What the item is: memory (something learned or stated), chunk (a "
        "document passage), summary (a document summary), episode (an earlier conversation "
        "of this user) or message (what was said, here or in an earlier conversation)."
    )
    text: str = Field(description="The item's text (clipped for long passages).")
    observed_on: str | None = Field(
        default=None, description="the day it was observed (YYYY-MM-DD), when known"
    )
    document_id: str | None = Field(
        default=None, description="The document it comes from, for a chunk or a summary."
    )
    page: int | None = Field(
        default=None,
        description="The 1-based page of the document it is on, when the document has pages.",
    )
    thread_id: str | None = Field(
        default=None,
        description="the conversation an episode is, or a message is in (its messages: "
        "/v1/threads)",
    )
    superseded: bool | None = Field(
        default=None,
        description="true when a later memory replaced this one (only a search with as_of "
        "or known_at returns those)",
    )
    debug: dict[str, Any] | None = Field(
        default=None, description="ranking detail; only with debug=true"
    )


@dataclass
class SearchResult:
    items: list[SearchItem]
    query_type: QueryType
    diagnostics: dict[str, Any] = field(default_factory=dict)
    retrieval: RetrievalResult | None = None


def _words(text: str) -> set[str]:
    """Word tokens in any script (``\\w`` is Unicode-aware), casefolded."""
    return {w for w in _WORD.findall(text.casefold()) if len(w) > 1}


def _day(raw: Any) -> str | None:
    return str(raw)[:10] if raw else None


def _clip(text: str, chars: int | None) -> str:
    return text if chars is None or len(text) <= chars else text[:chars] + "…"


def candidate_item(c: Candidate, *, debug: bool, text_chars: int | None = None) -> SearchItem:
    page = c.payload.get("page")
    return SearchItem(
        id=c.record_id,
        kind=cast(SearchKind, c.kind),
        text=_clip(c.text, text_chars),
        observed_on=_day(c.payload.get("observed_at")),
        document_id=c.payload.get("document_id"),
        page=int(page) if isinstance(page, int | float) else None,
        thread_id=c.payload.get("thread_id") if c.kind == "episode" else None,
        superseded=True if c.kind == "memory" and c.payload.get("current") is False else None,
        debug={
            "score": c.score,
            "retrievers": c.retrievers,
            "representation": c.representation.value,
            "section_path": c.payload.get("section_path"),
            "expanded_from": c.expanded_from,
            "expansion_edge": c.expansion_edge,
            "memory_type": c.payload.get("memory_type"),
            "subject": c.payload.get("subject"),
            "predicate": c.payload.get("predicate"),
        }
        if debug
        else None,
    )


def _message_item(m: Message, overlap: int, *, debug: bool, text_chars: int | None) -> SearchItem:
    return SearchItem(
        id=m.message_id,
        kind="message",
        text=_clip(f"{m.role.value}: {m.content}", text_chars),
        observed_on=m.occurred_at.date().isoformat(),
        thread_id=m.thread_id,
        debug={"overlap": overlap, "sequence": m.sequence} if debug else None,
    )


def _within(when: datetime, observed: ObservedRange | None) -> bool:
    if observed is None:
        return True
    start, end = observed
    return (start is None or when >= start) and (end is None or when <= end)


class Searcher:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        engine: RetrievalEngine,
        conversation: ConversationService,
        assist: LLMAssist,
    ) -> None:
        self.uow_factory = uow_factory
        self.engine = engine
        self.conversation = conversation
        self.assist = assist

    async def search(
        self,
        ctx: MemoryExecutionContext,
        query: str,
        *,
        kinds: Sequence[SearchKind] = DEFAULT_KINDS,
        limit: int,
        observed: ObservedRange | None = None,
        at: PointInTime | None = None,
        document_ids: Sequence[str] | None = None,
        debug: bool = False,
        text_chars: int | None = None,
    ) -> SearchResult:
        wanted = set(kinds)
        ranked = [k for k in kinds if k != "message"]
        items: list[SearchItem] = []

        async def ranked_search() -> RetrievalResult | None:
            if not ranked:
                return None
            async with self.assist.reading(ctx):
                return await self.engine.retrieve(
                    ctx,
                    query,
                    limit=limit,
                    kinds=tuple(ranked),
                    document_ids=document_ids,
                    observed=observed,
                    at=at,
                )

        async def history() -> list[SearchItem]:
            if "message" not in wanted:
                return []
            return await self._messages(ctx, query, observed, debug=debug, text_chars=text_chars)

        # independent reads (the index and the messages): neither waits on the other
        retrieved, messages = await asyncio.gather(ranked_search(), history())
        if retrieved is not None:
            items = [
                candidate_item(c, debug=debug, text_chars=text_chars)
                for c in retrieved.candidates
                if c.kind in wanted
            ]
        merged = [
            item
            for pair in itertools.zip_longest(items, messages)
            for item in pair
            if item is not None
        ]
        return SearchResult(
            items=merged[:limit],
            query_type=retrieved.routed.query_type if retrieved else QueryType.CONVERSATION_HISTORY,
            diagnostics=retrieved.diagnostics if retrieved else {},
            retrieval=retrieved,
        )

    async def _messages(
        self,
        ctx: MemoryExecutionContext,
        query: str,
        observed: ObservedRange | None,
        *,
        debug: bool,
        text_chars: int | None,
    ) -> list[SearchItem]:
        """The visible messages sharing words with the query, most shared first: this
        thread's newest, then the newest of the user's earlier conversations (newest first
        among equals); every message when the query shares nothing with any. Without a user,
        this thread's only."""
        wanted = _words(query)
        # independent reads: this thread and the user's earlier ones
        here, owned = await asyncio.gather(self._thread(ctx), self._owned(ctx, wanted, observed))
        earlier = [m for m in owned if m.thread_id != ctx.thread_id]
        scored = [
            (len(wanted & _words(m.content)), m)
            for m in (*here, *earlier)
            if m.kind is MessageKind.VISIBLE and _within(m.occurred_at, observed)
        ]
        hits = [(n, m) for n, m in scored if n] or scored

        def order(nm: tuple[int, Message]) -> tuple[int, bool, float]:
            shared, m = nm
            other = m.thread_id != ctx.thread_id
            # a thread's own order is its sequence; across threads, when it was said
            return (-shared, other, -(m.occurred_at.timestamp() if other else m.sequence))

        hits.sort(key=order)
        return [_message_item(m, n, debug=debug, text_chars=text_chars) for n, m in hits]

    async def _thread(self, ctx: MemoryExecutionContext) -> list[Message]:
        if not ctx.thread_id:
            return []
        async with self.uow_factory() as uow:
            if await uow.threads.get(ctx.tenant_id, ctx.thread_id) is None:
                return []
            return await self.conversation.list_messages(
                uow, ctx, ctx.thread_id, limit=HISTORY_SCAN
            )

    async def _owned(
        self, ctx: MemoryExecutionContext, wanted: set[str], observed: ObservedRange | None
    ) -> list[Message]:
        """The newest messages of the threads the caller's user owns: the user's own
        conversations, which only that user (and the agents acting for them) reads. No user,
        none."""
        if not ctx.user_id:
            return []
        since, until = observed or (None, None)
        async with self.uow_factory() as uow:
            return await uow.messages.owned_recent(
                ctx.tenant_id,
                ctx.user_id,
                limit=OWNED_SCAN,
                threads=OWNED_THREADS,
                words=sorted(wanted, key=lambda w: (-len(w), w))[:OWNED_WORDS],
                since=since,
                until=until,
            )
