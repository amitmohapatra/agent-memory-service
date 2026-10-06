"""The memory tools an agent calls itself (pull mode, ReAct): a fixed set, each a thin call
into the service that owns the capability, in the caller's scope.

Every call is logged as a pull (the request's pattern, the tool, its arguments, the ids it
returned); a returned id the run later cites or acts on is marked used. The prefetch job
learns from those which items to pre-include in the pushed context for similar requests.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType, Visibility
from memory_service.domain.errors import NotFound, ValidationFailed
from memory_service.domain.instants import UtcDateTime
from memory_service.domain.pulls import (
    PREFETCH_BATCH,
    PREFETCH_MAX,
    PREFETCH_MIN_PULLS,
    PREFETCH_MIN_RATE,
    PULL_RESULTS_MAX,
    PULL_SETTLE,
    AgentPull,
)
from memory_service.modules.context.views import hints_view
from memory_service.modules.retrieval.engine import PointInTime
from memory_service.modules.retrieval.search import DEFAULT_KINDS, SearchKind
from memory_service.modules.tools.patterns import task_pattern
from memory_service.observability.logging import get_logger

log = get_logger(__name__)

#: Candidates ``tool_search`` weighs before it names the next tool.
TOOL_SEARCH_K: Final = 8
#: Items a search returns by default, and at most.
DEFAULT_K: Final = 8
MAX_K: Final = 20
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
_ID: Final = "a memory id, or its handle in the context ([m3] -> m3)"


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MemorySearchArgs(_Args):
    query: str = Field(..., min_length=1, max_length=2000, description="what to look for")
    kinds: list[SearchKind] | None = Field(
        default=None,
        description="memory, chunk (document passages), summary, episode (earlier "
        "conversations), message (what was said, here and in earlier conversations); default "
        "memory and chunk",
    )
    time_from: UtcDateTime | None = Field(default=None, description="observed since")
    time_to: UtcDateTime | None = Field(default=None, description="observed until")
    as_of: UtcDateTime | None = Field(
        default=None, description="what was true at this moment (includes replaced memories)"
    )
    known_at: UtcDateTime | None = Field(
        default=None, description="what had been learned by this moment"
    )
    k: int = Field(default=DEFAULT_K, ge=1, le=MAX_K)


class MemoryRememberArgs(_Args):
    content: str = Field(..., min_length=1, max_length=4000, description="the statement")
    kind: MemoryKind = Field(default="SEMANTIC")
    scope: MemoryScope = Field(
        default="user",
        description="who may read it: user, agent (this agent for this user), run, thread, "
        "group (the agent group), workspace",
    )


class MemoryUpdateArgs(_Args):
    id: str = Field(..., min_length=1, max_length=200, description=_ID)
    content: str = Field(..., min_length=1, max_length=4000, description="the new statement")


class MemoryForgetArgs(_Args):
    id: str = Field(..., min_length=1, max_length=200, description=_ID)


class ProfileEditArgs(_Args):
    block: str = Field(
        ..., max_length=60, description="user, agent or workspace, optionally .<name>"
    )
    old: str = Field(
        default="", max_length=4000, description="the exact text to replace; empty: all of it"
    )
    new: str = Field(..., max_length=4000)


class ToolSearchArgs(_Args):
    task: str = Field(..., min_length=1, max_length=2000, description="the task, in words")


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
        "Search what is remembered for this user, agent and conversation, and the documents "
        "they may read. Returns items with id, kind, text and observed_on.",
        MemorySearchArgs,
    ),
    AgentTool(
        "memory_remember",
        "Remember a statement verbatim: facts, preferences and decisions worth keeping.",
        MemoryRememberArgs,
    ),
    AgentTool(
        "memory_update",
        "Replace a memory with a new statement; the old one stays in its history.",
        MemoryUpdateArgs,
    ),
    AgentTool("memory_forget", "Forget a memory: it is no longer retrieved.", MemoryForgetArgs),
    AgentTool(
        "profile_edit",
        "Edit a pinned profile block: replace the exact old text with the new text (empty old "
        "replaces the whole block). Profile blocks are part of every context.",
        ProfileEditArgs,
    ),
    AgentTool(
        TOOL_SEARCH,
        "Which of your tools fit a task, best first, each with a 0-1 confidence, the argument "
        "values already known and the ones still missing; the next step of the learned plan.",
        ToolSearchArgs,
    ),
)
BY_NAME: Final = {tool.name: tool for tool in TOOLS}


def result_ids(result: Any) -> list[str]:
    """The ids a result names: items of a list, or the one a write returned."""
    items = result if isinstance(result, list) else [result]
    return [str(i["id"]) for i in items if isinstance(i, dict) and i.get("id")][:PULL_RESULTS_MAX]


class AgentTools:
    def __init__(self, uow_factory: Any, services: dict[str, Any]) -> None:
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
            result = await self._tool_search(ctx, ToolSearchArgs.model_validate(args), toolbox)
        else:
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
        observed = (args.time_from, args.time_to) if args.time_from or args.time_to else None
        found = await self.services["search"].search(
            ctx,
            args.query,
            kinds=args.kinds or DEFAULT_KINDS,
            limit=args.k,
            observed=observed,
            at=PointInTime(as_of=args.as_of, known_at=args.known_at),
            text_chars=TEXT_CHARS,
        )
        return [item.model_dump(mode="json", exclude_none=True) for item in found.items]

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
        memory_id = await self.services["bundle_records"].resolve(ctx, args.id)
        async with self.uow_factory() as uow:
            new = await self.services["memory"].supersede(
                uow, ctx, memory_id, content=args.content, reason="updated by the agent"
            )
            await uow.commit()
        await self.used(ctx, [memory_id])
        return {"id": new.memory_id, "supersedes": memory_id}

    async def _memory_forget(self, ctx: MemoryExecutionContext, args: MemoryForgetArgs) -> Any:
        memory_id = await self.services["bundle_records"].resolve(ctx, args.id)
        async with self.uow_factory() as uow:
            forgotten = await self.services["memory"].forget(uow, ctx, memory_id)
            await uow.commit()
        await self.used(ctx, [memory_id])
        return {"id": memory_id, "forgotten": forgotten is not None}

    async def _profile_edit(self, ctx: MemoryExecutionContext, args: ProfileEditArgs) -> Any:
        async with self.uow_factory() as uow:
            block = await self.services["profile"].edit(uow, ctx, args.block, args.old, args.new)
            await uow.commit()
        return {"block": block.block, "text": block.text, "version": block.version}

    # ------------------------------------------------------------------ tools
    async def _tool_search(
        self, ctx: MemoryExecutionContext, args: ToolSearchArgs, toolbox: Sequence[str] | None
    ) -> Any:
        visibility = await self.services["authz"].visibility(ctx)
        async with self.uow_factory() as uow:
            profile = await self.services["profile"].blocks(uow, ctx)
        hints = await self.services["tool_hints"].hints(
            ctx,
            args.task,
            available=toolbox,
            k=TOOL_SEARCH_K,
            scope_keys=list(visibility.keys),
            profile=profile,
        )
        return hints_view(hints)
