"""The memory tools an agent calls itself (pull mode, ReAct): a fixed set, each a thin call
into the service that owns the capability, in the caller's scope.

Every call is logged as a pull (the request's pattern, the tool, its arguments, the ids it
returned); a returned id the run later cites or acts on is marked used. The prefetch job
learns from those which items to pre-include in the pushed context for similar requests.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType, Visibility
from memory_service.domain.errors import NotFound, ValidationFailed
from memory_service.domain.pulls import (
    PREFETCH_BATCH,
    PREFETCH_MAX,
    PREFETCH_MIN_PULLS,
    PREFETCH_MIN_RATE,
    PULL_RESULTS_MAX,
    PULL_SETTLE,
    AgentPull,
)
from memory_service.modules.tools.patterns import task_pattern
from memory_service.observability.logging import get_logger

log = get_logger(__name__)

#: Items a search tool returns by default, and at most.
DEFAULT_K: Final = 8
MAX_K: Final = 20
#: Messages of the thread one history search looks through, newest first.
HISTORY_SCAN: Final = 200
#: Characters of one returned text.
TEXT_CHARS: Final = 1000
_WORD: Final = re.compile(r"\w+")

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


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MemorySearchArgs(_Args):
    query: str = Field(..., min_length=1, max_length=2000, description="what to look for")
    kinds: list[Literal["memory", "chunk", "summary"]] | None = Field(
        default=None,
        description="memory (what was learned or stated), chunk (document passages), summary "
        "(document summaries); omitted: memories and document passages",
    )
    time_from: datetime | None = Field(default=None, description="only what was observed since")
    time_to: datetime | None = Field(default=None, description="only what was observed until")
    k: int = Field(default=DEFAULT_K, ge=1, le=MAX_K, description="how many results")


class MemoryRememberArgs(_Args):
    content: str = Field(..., min_length=1, max_length=4000, description="the statement, verbatim")
    kind: MemoryKind = Field(
        default="SEMANTIC",
        description="SEMANTIC (a fact), PREFERENCE, EPISODIC (something that happened), "
        "PROCEDURAL (how to do something), TASK, USER (about the user), TOOL, OUTCOME",
    )
    scope: MemoryScope = Field(
        default="user",
        description="who may read it: user (the user and their agents), agent (this agent, for "
        "this user), run (this run and the run that started it), thread, group (the agent "
        "group), workspace",
    )


class MemoryUpdateArgs(_Args):
    id: str = Field(..., min_length=1, max_length=200, description="the memory to change")
    content: str | None = Field(
        default=None, max_length=4000, description="the new statement, replacing the old one"
    )
    invalidate: bool = Field(
        default=False, description="true: the memory is no longer true and has no replacement"
    )
    reason: str = Field(..., min_length=1, max_length=500, description="why it changed")


class MemoryForgetArgs(_Args):
    id: str = Field(..., min_length=1, max_length=200, description="the memory to forget")
    reason: str = Field(..., min_length=1, max_length=500, description="why")


class HistorySearchArgs(_Args):
    query: str | None = Field(
        default=None, max_length=2000, description="words the message contains; omitted: any"
    )
    time_from: datetime | None = None
    time_to: datetime | None = None
    k: int = Field(default=DEFAULT_K, ge=1, le=MAX_K)


class ProfileEditArgs(_Args):
    block: str = Field(
        ...,
        max_length=60,
        description="user, agent or workspace, optionally .<name> (user.preferences)",
    )
    old: str = Field(
        ..., max_length=4000, description="the exact text to replace; empty replaces the block"
    )
    new: str = Field(..., max_length=4000, description="the text to put in its place")


class TaskArgs(_Args):
    task: str = Field(..., min_length=1, max_length=2000, description="the task, in words")
    k: int = Field(default=3, ge=1, le=MAX_K)


class RecordOutcomeArgs(_Args):
    success: bool = Field(..., description="whether this run achieved its task")
    note: str | None = Field(default=None, max_length=1000)


Handler = Callable[[MemoryExecutionContext, Any], Awaitable[Any]]


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    args: type[_Args]

    def spec(self) -> dict[str, Any]:
        schema = self.args.model_json_schema()
        schema.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": schema}


TOOLS: Final = (
    AgentTool(
        "memory_search",
        "Search what is remembered for this user, agent and thread, and the documents "
        "they may read. Returns ranked items with ids, kinds, text and when they were observed.",
        MemorySearchArgs,
    ),
    AgentTool(
        "memory_remember",
        "Remember a statement verbatim, now. Use for facts, preferences and decisions worth "
        "keeping beyond this conversation.",
        MemoryRememberArgs,
    ),
    AgentTool(
        "memory_update",
        "Replace a memory with a new statement, or mark it no longer true. The old version "
        "stays in the history.",
        MemoryUpdateArgs,
    ),
    AgentTool("memory_forget", "Forget a memory: it is no longer retrieved.", MemoryForgetArgs),
    AgentTool(
        "history_search",
        "Search the messages of this conversation, newest first, by words and time.",
        HistorySearchArgs,
    ),
    AgentTool(
        "profile_edit",
        "Edit a pinned profile block in place: replace the exact old text with the new text "
        "(empty old replaces the whole block). Pinned blocks are part of every context.",
        ProfileEditArgs,
    ),
    AgentTool(
        "procedures_search",
        "Find procedures learned from earlier successful runs of similar tasks: the steps, "
        "where their arguments come from, and how often they worked.",
        TaskArgs,
    ),
    AgentTool(
        "tool_search",
        "Find which tools fit a task, the learned plan, the next step and argument values "
        "found in memory.",
        TaskArgs,
    ),
    AgentTool(
        "record_outcome",
        "Record whether this run achieved its task: successful runs teach procedures.",
        RecordOutcomeArgs,
    ),
)
BY_NAME: Final = {tool.name: tool for tool in TOOLS}


def _text(value: str) -> str:
    return value if len(value) <= TEXT_CHARS else value[:TEXT_CHARS] + "…"


def _within(observed: str | None, time_from: datetime | None, time_to: datetime | None) -> bool:
    if time_from is None and time_to is None:
        return True
    if not observed:
        return False
    when = datetime.fromisoformat(observed)
    return (time_from is None or when >= time_from) and (time_to is None or when <= time_to)


def _words(text: str) -> set[str]:
    """Word tokens in any script (``\\w`` is Unicode-aware), casefolded."""
    return {w for w in _WORD.findall(text.casefold()) if len(w) > 1}


def result_ids(result: Any) -> list[str]:
    """The ids a result names: items of a list, or the one a write returned."""
    items = result if isinstance(result, list) else [result]
    return [str(i["id"]) for i in items if isinstance(i, dict) and i.get("id")][:PULL_RESULTS_MAX]


class AgentTools:
    def __init__(self, uow_factory: Any, services: dict[str, Any]) -> None:
        self.uow_factory = uow_factory
        self.services = services

    @staticmethod
    def specs() -> list[dict[str, Any]]:
        return [tool.spec() for tool in TOOLS]

    async def call(self, ctx: MemoryExecutionContext, name: str, raw: dict[str, Any]) -> Any:
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
        handler: Handler = getattr(self, f"_{name}")
        result = await handler(ctx, args)
        await self._pull(ctx, name, args, result)
        return result

    async def _pull(self, ctx: MemoryExecutionContext, name: str, args: _Args, result: Any) -> None:
        asked = getattr(args, "query", None) or getattr(args, "task", None) or ""
        pull = AgentPull(
            tenant_id=ctx.tenant_id,
            scope_key=ctx.principal_id,
            run_id=ctx.agent_run_id,
            pattern=task_pattern(asked),
            tool=name,
            args=args.model_dump(mode="json", exclude_none=True),
            result_ids=result_ids(result) if name.endswith("_search") else [],
        )
        async with self.uow_factory() as uow:
            await uow.pulls.add(pull)
            await uow.commit()

    async def used(self, ctx: MemoryExecutionContext, ids: Sequence[str]) -> None:
        """The run used these ids (cited them in a verified answer, or acted on them)."""
        if ctx.agent_run_id and ids:
            async with self.uow_factory() as uow:
                await uow.pulls.mark_used(ctx.tenant_id, ctx.agent_run_id, ids)
                await uow.commit()

    # ------------------------------------------------------------------ prefetch learning
    async def learn_prefetch(self, *, now: datetime | None = None) -> int:
        """The periodic prefetch job: fold settled pulls into the per-pattern counts."""
        before = (now or datetime.now(UTC)) - PULL_SETTLE
        async with self.uow_factory() as uow:
            pulls = await uow.pulls.settled(before=before, limit=PREFETCH_BATCH)
            if pulls:
                await uow.pulls.fold(pulls)
                await uow.commit()
        return len(pulls)

    async def prefetched(self, ctx: MemoryExecutionContext, query: str) -> list[str]:
        """The items this principal's pulls for requests of this pattern used most: one
        indexed read of the precomputed counts."""
        pattern = task_pattern(query)
        if not pattern:
            return []
        async with self.uow_factory() as uow:
            return await uow.pulls.prefetch(
                ctx.tenant_id,
                ctx.principal_id,
                pattern,
                min_pulls=PREFETCH_MIN_PULLS,
                min_rate=PREFETCH_MIN_RATE,
                limit=PREFETCH_MAX,
            )

    # ------------------------------------------------------------------ memory
    async def _memory_search(self, ctx: MemoryExecutionContext, args: MemorySearchArgs) -> Any:
        kinds = tuple(args.kinds or ("memory", "chunk"))
        # summaries are indexed beside the chunks and come back with them
        searched = tuple(dict.fromkeys("chunk" if k == "summary" else k for k in kinds))
        async with self.services["llm_assist"].reading(ctx, use_llm=None):
            found = await self.services["retrieval"].retrieve(
                ctx, args.query, limit=MAX_K * 2, kinds=searched
            )
        items = [
            {
                "id": c.record_id,
                "kind": c.kind,
                "text": _text(c.text),
                "observed_at": c.payload.get("observed_at"),
            }
            for c in found.candidates
            if c.kind in kinds
            and _within(c.payload.get("observed_at"), args.time_from, args.time_to)
        ]
        return items[: args.k]

    async def _memory_remember(self, ctx: MemoryExecutionContext, args: MemoryRememberArgs) -> Any:
        async with self.uow_factory() as uow:
            ack = await self.services["memory"].remember(
                uow,
                ctx,
                content=args.content,
                memory_type=MemoryType(args.kind),
                lifetime=Lifetime.LONG_TERM,
                visibility=_VISIBILITY[args.scope],
            )
            await uow.commit()
        return {"id": ack.memory_id, "deduplicated": ack.deduplicated}

    async def _memory_update(self, ctx: MemoryExecutionContext, args: MemoryUpdateArgs) -> Any:
        if (args.content is None) == (not args.invalidate):
            raise ValidationFailed("memory_update needs either content or invalidate=true")
        memory = self.services["memory"]
        async with self.uow_factory() as uow:
            if args.invalidate:
                await memory.retract(uow, ctx, args.id, reason=args.reason)
                result = {"id": args.id, "invalidated": True}
            else:
                new = await memory.supersede(
                    uow, ctx, args.id, content=str(args.content), reason=args.reason
                )
                result = {"id": new.memory_id, "supersedes": args.id}
            await uow.commit()
        await self.used(ctx, [args.id])
        return result

    async def _memory_forget(self, ctx: MemoryExecutionContext, args: MemoryForgetArgs) -> Any:
        async with self.uow_factory() as uow:
            forgotten = await self.services["memory"].forget(uow, ctx, args.id)
            await uow.commit()
        await self.used(ctx, [args.id])
        log.info("agent_tools.forget", memory_id=args.id, reason=args.reason, **ctx.log_fields())
        return {"id": args.id, "forgotten": forgotten is not None}

    # ------------------------------------------------------------------ conversation
    async def _history_search(self, ctx: MemoryExecutionContext, args: HistorySearchArgs) -> Any:
        if not ctx.thread_id:
            raise ValidationFailed("history_search needs a thread in the scope")
        async with self.uow_factory() as uow:
            messages = await self.services["conversation"].list_messages(
                uow, ctx, ctx.thread_id, limit=HISTORY_SCAN
            )
        wanted = _words(args.query or "")
        hits = [
            {
                "id": m.message_id,
                "role": m.role.value,
                "text": _text(m.content),
                "sequence": m.sequence,
                "observed_at": m.occurred_at.isoformat(),
            }
            for m in reversed(messages)
            if (not wanted or wanted & _words(m.content))
            and _within(m.occurred_at.isoformat(), args.time_from, args.time_to)
        ]
        return hits[: args.k]

    async def _profile_edit(self, ctx: MemoryExecutionContext, args: ProfileEditArgs) -> Any:
        profile = self.services["profile"]
        async with self.uow_factory() as uow:
            if args.old:
                block = await profile.edit(uow, ctx, args.block, args.old, args.new)
            else:
                block = await profile.put(uow, ctx, args.block, args.new)
            await uow.commit()
        return {"block": block.block, "text": block.text, "version": block.version}

    # ------------------------------------------------------------------ tools
    async def _scope_keys(self, ctx: MemoryExecutionContext) -> list[str]:
        return list((await self.services["authz"].visibility(ctx)).keys)

    async def _procedures_search(self, ctx: MemoryExecutionContext, args: TaskArgs) -> Any:
        found = await self.services["tool_hints"].procedures(
            ctx, args.task, await self._scope_keys(ctx), k=args.k
        )
        return [
            {
                "id": p.procedure_id,
                "title": p.title,
                "strategy": p.strategy,
                "steps": p.tools,
                "success_rate": p.success_rate,
                "support": p.support,
            }
            for p in found
        ]

    async def _tool_search(self, ctx: MemoryExecutionContext, args: TaskArgs) -> Any:
        hints = await self.services["tool_hints"].hints(
            ctx, args.task, available=None, k=args.k, scope_keys=await self._scope_keys(ctx)
        )
        return hints.model_dump(mode="json")

    async def _record_outcome(self, ctx: MemoryExecutionContext, args: RecordOutcomeArgs) -> Any:
        if not ctx.agent_run_id:
            raise ValidationFailed("record_outcome needs an agent run in the scope")
        async with self.uow_factory() as uow:
            outcome = await self.services["tool_memory"].set_outcome(
                uow, ctx, run_id=ctx.agent_run_id, success=args.success, note=args.note
            )
            await uow.commit()
        return {"run_id": outcome.run_id, "success": outcome.success}
