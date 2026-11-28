"""The memory tools an agent calls itself (pull mode, ReAct): a fixed set, each a thin call
into the service that owns the capability, in the caller's scope.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from memory_service.config.constants import ContextSettings, RetrievalSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType, Visibility
from memory_service.domain.errors import NotFound, ValidationFailed
from memory_service.domain.instants import UtcDateTime
from memory_service.modules.context.sections import CHARS_PER_TOKEN
from memory_service.modules.context.views import hints_view
from memory_service.modules.retrieval.engine import PointInTime
from memory_service.modules.retrieval.search import (
    DEFAULT_BY_TIME,
    Order,
    SearchKind,
    Speaker,
    default_kinds,
)
from memory_service.observability.logging import get_logger

if TYPE_CHECKING:
    from memory_service.application.services import Services

log = get_logger(__name__)

#: Candidates ``tool_search`` weighs before it names the next tool.
TOOL_SEARCH_K: Final = 8
#: Items a search returns by default, and at most.
#: The most a search names (``k``): as deep as a context ranks memories for a request.
MAX_K: Final[int] = RetrievalSettings.model_fields["memory_recall_k"].default
#: What one search returns at most, in characters of text with the turns around each item:
#: what a context takes at its default budget, so a search without ``k`` sees what a pushed
#: context would, and no more.
RESULT_CHARS: Final[int] = ContextSettings.model_fields["token_budget"].default * CHARS_PER_TOKEN
#: Characters of one returned text.
TEXT_CHARS: Final = 1000
#: The tool the caller answers with its own toolbox (``toolbox`` on the call).
TOOL_SEARCH: Final = "tool_search"

#: The kinds of memory an agent states.
MemoryKind = Literal[
    "SEMANTIC", "PREFERENCE", "EPISODIC", "PROCEDURAL", "TASK", "USER", "TOOL", "OUTCOME"
]
#: Who a stated memory is for.
MemoryScope = Literal["user", "agent", "run", "thread", "group", "workspace"]
_VISIBILITY: Final = {
    "user": Visibility.USER,
    "agent": Visibility.PRIVATE,
    "run": Visibility.RUN,
    "thread": Visibility.THREAD,
    "group": Visibility.AGENT_GROUP,
    "workspace": Visibility.WORKSPACE,
}
_ID: Final = "the memory's id (mem_...) from a search, or its handle in the context ([m3] -> m3)"


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MemorySearchArgs(_Args):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="what to look for, in words: the request's own words first, then other "
        "words for each piece (by meaning; names, codes and numbers match exactly)",
    )
    kinds: list[SearchKind] | None = Field(
        default=None,
        description="memory: what was learned or stated - facts, decisions, rules, preferences. "
        "message: what was said word for word, here and in earlier conversations, your own "
        "answers too ('what did I ask yesterday'). episode: an earlier conversation as a whole "
        "('what did we discuss about X'). fact: relations between people and things the "
        "knowledge graph holds of what the query names; a 'why' question reads the causal "
        "ones first ('what drove the drop'). belief: what memory concluded about a person, "
        "current values only ('where does she work now'). chunk: document passages. summary: "
        "document summaries. Default memory and chunk.",
    )
    time_from: UtcDateTime | None = Field(
        default=None, description="only what was said or seen since this moment (ISO 8601)"
    )
    time_to: UtcDateTime | None = Field(
        default=None, description="only what was said or seen until this moment (ISO 8601)"
    )
    as_of: UtcDateTime | None = Field(
        default=None,
        description="what was true at this moment, values replaced since included ('what was "
        "her address in March')",
    )
    known_at: UtcDateTime | None = Field(
        default=None,
        description="only what had been learned by this moment ('what did we know before "
        "the meeting')",
    )
    order: Order = Field(
        default="relevance",
        description="relevance: by meaning. newest or oldest: the messages by when they were "
        "said, whatever the query - the last or the first thing said",
    )
    said_by: Speaker | None = Field(
        default=None,
        description="with newest or oldest: user for only what the person said ('what did I "
        "ask last'), assistant for only what you answered ('what did you tell me last')",
    )
    here: bool = Field(
        default=False,
        description="with newest or oldest: only this conversation (default: every "
        "conversation of the user, this one included)",
    )
    k: int | None = Field(
        default=None,
        ge=1,
        le=MAX_K,
        description="how many items: leave it out for everything memory ranks for the request, "
        "as deep as a context would carry it (best first); set it only to narrow (at most "
        f"{MAX_K}; with order newest or oldest, {DEFAULT_BY_TIME} unless set)",
    )


class MemoryRememberArgs(_Args):
    content: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="the note, one statement, kept word for word: say who and what in full "
        "('Option for SKU-2210: move 22 units from Hartford to Boston'), not 'that one'",
    )
    kind: MemoryKind = Field(
        default="SEMANTIC",
        description="what it is: SEMANTIC a fact; PREFERENCE how the person wants things done; "
        "EPISODIC something that happened, with when; PROCEDURAL steps that worked; TASK "
        "something to do or decided to do; USER who the person is; TOOL what a tool needs or "
        "returned; OUTCOME how a task ended",
    )
    scope: MemoryScope = Field(
        default="user",
        description="who may read it later: user (this person, with any agent); agent (only "
        "this agent, for this person); run (this run only); thread (this conversation only); "
        "group (the agents of this agent group); workspace (everyone in the person's team)",
    )
    valid_to: UtcDateTime | None = Field(
        default=None,
        description="when it stops holding, for something said for a period ('for this week "
        "only': the end of that week, ISO 8601); after it the note is no longer found",
    )


class MemoryUpdateArgs(_Args):
    id: str = Field(..., min_length=1, max_length=200, description=_ID)
    content: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="the statement that replaces it, whole (not only what changed)",
    )


class MemoryForgetArgs(_Args):
    id: str = Field(..., min_length=1, max_length=200, description=_ID)


class ProfileEditArgs(_Args):
    block: str = Field(
        ...,
        max_length=60,
        description="which pinned block: user (who the person is), agent (this agent's own "
        "notes), workspace (the team's), optionally .<name> for a named block of it",
    )
    old: str = Field(
        default="",
        max_length=4000,
        description="the exact text to replace, copied from the block; empty only for a "
        "block that holds nothing yet (for one that holds text the refusal shows it)",
    )
    new: str = Field(
        ...,
        max_length=4000,
        description="the text that takes its place; empty removes the old text",
    )


class ToolSearchArgs(_Args):
    task: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="the task, in words, with the values it names ('reorder SKU-2210 for "
        "Boston from Acme')",
    )


Handler = Callable[[MemoryExecutionContext, Any], Awaitable[Any]]


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    args: type[_Args]

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": lean_schema(self.args.model_json_schema()),
        }


def lean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """The arguments' schema as a model reads it, every call: no ``title`` repeating a name,
    and an optional argument as its one type - it is optional by not being ``required``, not
    by an ``anyOf`` with null and a ``default: null`` beside it."""
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
        options = [o for o in prop.get("anyOf", ()) if o != {"type": "null"}]
        if "anyOf" in prop and len(options) == 1:
            del prop["anyOf"]
            prop.update(options[0])
        if "default" in prop and prop["default"] is None:
            del prop["default"]
    return schema


TOOLS: Final = (
    AgentTool(
        "memory_search",
        "Search this person's long-term memory, their earlier conversations and the documents "
        "they may read. Use it when the answer depends on something said, decided or uploaded "
        "before that your context does not show; not for what can change now (a stock "
        "figure, a price, a status, availability): ask your own tools for that. Returns items "
        "(id, kind, text, observed_on; superseded when a later statement replaced it): cite "
        "an item by its id. Pick kinds by what you need (see kinds): what was learned, what "
        "was said word for word, an earlier conversation, a relation, what memory concluded "
        "about a person, a document. For the last or first thing said use order newest or "
        "oldest; for what was true at a past moment, as_of; for a period, time_from and "
        "time_to. A request about several things is answered from pieces said at different "
        "times: search for each piece, and again in other words, until nothing new comes "
        "back.",
        MemorySearchArgs,
    ),
    AgentTool(
        "memory_remember",
        "Note something for later, verbatim: a decision you made, an outcome, a value you "
        "worked out, or what the person asked you to note. What the person says in the chat "
        "is kept and read into memory on its own: do not restate it here, it would be stored "
        "twice.",
        MemoryRememberArgs,
    ),
    AgentTool(
        "memory_update",
        "Replace a note of yours with a corrected one, when what it says is no longer right "
        "(a decision changed, a value was wrong). The old one stays in its history (found "
        "with memory_search as_of a moment before). Only a memory you wrote: what the person "
        "said is theirs - when they correct themselves, their new words are kept on their own.",
        MemoryUpdateArgs,
    ),
    AgentTool(
        "memory_forget",
        "Forget a note of yours that should not be used again (wrong, or asked to be dropped): "
        "it is no longer found or attached. Only a memory you wrote: what the person said is "
        "theirs to forget, through the application, not through you.",
        MemoryForgetArgs,
    ),
    AgentTool(
        "profile_edit",
        "Edit a pinned profile block - text that goes with every request (the person's name, "
        "role, team, language; this agent's standing notes) - when the person tells you to "
        "change it or it is plainly out of date. Replace the exact old text with the new; a "
        "block that holds text is never overwritten blind (the refusal shows its text).",
        ProfileEditArgs,
    ),
    AgentTool(
        TOOL_SEARCH,
        "Which of your own tools fit a task, best first: each with a 0-1 confidence, the "
        "argument values already known (from memory and the conversation), the ones still "
        "missing with the question to ask, and the next step of a plan learned from earlier "
        "runs of the same task. Use it when you are unsure which tool to call or what an "
        "argument should be; it calls nothing itself.",
        ToolSearchArgs,
    ),
)
BY_NAME: Final = {tool.name: tool for tool in TOOLS}


class AgentTools:
    def __init__(self, uow_factory: Any, services: Services) -> None:
        self.uow_factory = uow_factory
        self.services = services

    @staticmethod
    @functools.cache
    def specs() -> tuple[dict[str, Any], ...]:
        """Every tool's name, description and input schema. The set is fixed for the life
        of the process, so the schemas are built once; callers read them, never change them."""
        return tuple(tool.spec() for tool in TOOLS)

    async def call(
        self,
        ctx: MemoryExecutionContext,
        name: str,
        raw: dict[str, Any],
        *,
        toolbox: Sequence[str] | None = None,
    ) -> Any:
        """Run one tool. ``toolbox`` is the caller's own tools, which ``tool_search`` chooses
        among (every catalog tool without it)."""
        tool = BY_NAME.get(name)
        if tool is None:
            raise NotFound(f"no agent tool {name!r}")
        try:
            args = tool.args.model_validate(raw)
        except PydanticValidationError as exc:
            raise ValidationFailed(
                f"invalid arguments for {name}",
                details={
                    "errors": [{"loc": ["args", *e["loc"]], "msg": e["msg"]} for e in exc.errors()]
                },
            ) from exc
        if name == TOOL_SEARCH:
            return await self._tool_search(ctx, ToolSearchArgs.model_validate(args), toolbox)
        handler: Handler = getattr(self, f"_{name}")
        return await handler(ctx, args)

    # ------------------------------------------------------------------ memory
    async def _memory_search(self, ctx: MemoryExecutionContext, args: MemorySearchArgs) -> Any:
        observed = (args.time_from, args.time_to) if args.time_from or args.time_to else None
        found = await self.services.search.search(
            ctx,
            args.query,
            kinds=args.kinds or default_kinds(args.query),
            limit=args.k,
            observed=observed,
            at=PointInTime(as_of=args.as_of, known_at=args.known_at),
            text_chars=TEXT_CHARS,
            order=args.order,
            said_by=args.said_by,
            here=args.here,
        )
        return within_chars(
            [item.model_dump(mode="json", exclude_none=True) for item in found.items], RESULT_CHARS
        )

    async def _memory_remember(self, ctx: MemoryExecutionContext, args: MemoryRememberArgs) -> Any:
        async with self.uow_factory() as uow:
            ack = await self.services.memory.remember(
                uow,
                ctx,
                content=args.content,
                memory_type=MemoryType(args.kind),
                lifetime=Lifetime.LONG_TERM,
                visibility=_VISIBILITY[args.scope],
                valid_to=args.valid_to,
            )
            await uow.commit()
        return {"id": ack.memory_id, "deduplicated": ack.deduplicated}

    async def _memory_update(self, ctx: MemoryExecutionContext, args: MemoryUpdateArgs) -> Any:
        memory_id = await self.services.bundle_records.resolve(ctx, args.id)
        async with self.uow_factory() as uow:
            new = await self.services.memory.supersede(
                uow, ctx, memory_id, content=args.content, reason="updated by the agent"
            )
            await uow.commit()
        return {"id": new.memory_id, "supersedes": memory_id}

    async def _memory_forget(self, ctx: MemoryExecutionContext, args: MemoryForgetArgs) -> Any:
        memory_id = await self.services.bundle_records.resolve(ctx, args.id)
        async with self.uow_factory() as uow:
            forgotten = await self.services.memory.forget(uow, ctx, memory_id)
            await uow.commit()
        return {"id": memory_id, "forgotten": forgotten is not None}

    async def _profile_edit(self, ctx: MemoryExecutionContext, args: ProfileEditArgs) -> Any:
        async with self.uow_factory() as uow:
            if not args.old:
                # a whole block is replaced only by one who has read it: a context may not
                # carry the block's every line (only who the person is; rules where they apply)
                held = next(
                    (
                        b
                        for b in await self.services.profile.blocks(uow, ctx)
                        if b.block == args.block
                    ),
                    None,
                )
                if held is not None and held.text.strip():
                    raise ValidationFailed(
                        f"{args.block} holds text: pass the exact text to replace as old "
                        "(all of it to replace the block)",
                        details={"block": args.block, "text": held.text},
                    )
            block = await self.services.profile.edit(uow, ctx, args.block, args.old, args.new)
            await uow.commit()
        return {"block": block.block, "text": block.text, "version": block.version}

    # ------------------------------------------------------------------ tools
    async def _tool_search(
        self, ctx: MemoryExecutionContext, args: ToolSearchArgs, toolbox: Sequence[str] | None
    ) -> Any:
        visibility = await self.services.authz.visibility(ctx)
        async with self.uow_factory() as uow:
            profile = await self.services.profile.blocks(uow, ctx)
        hints = await self.services.tool_hints.hints(
            ctx,
            args.task,
            available=toolbox,
            k=TOOL_SEARCH_K,
            scope_keys=list(visibility.keys),
            profile=profile,
            visibility=visibility,
        )
        return hints_view(hints)


def within_chars(items: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """``items`` best first, as many as their text and the turns around them fit ``budget``
    characters (always the first)."""
    out: list[dict[str, Any]] = []
    used = 0
    for item in items:
        size = sum(len(str(item.get(f) or "")) for f in ("text", "before", "after"))
        if out and used + size > budget:
            break
        out.append(item)
        used += size
    return out
