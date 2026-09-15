"""``LangGraphMemory``: the Memory Service as a LangGraph node companion.

    memory = LangGraphMemory(client, tenant_id="acme", user_id="u1")
    graph.add_node("answer", memory.wrap(answer, recall="question", observe="answer"))
    await graph.ainvoke(state, {"configurable": {"thread_id": "chat-42"}})

Per wrapped node, in order:

1. **resolve** the ``MemoryContext`` from the config (thread id, subgraph lineage, explicit
   ``configurable.memory`` overrides — see :mod:`.lineage`); the context is bound for the
   duration of the node so tools can use ``universal_memory.current_context()``;
2. **recall**: fetch a bounded ``ContextBundle`` for a query taken from the state and hand
   it to the node under ``inject`` (default ``"memory"``);
3. run the node (sync or async; ``config``/``runtime``/``store``/``writer`` are injected if
   the node asks for them);
4. **record**: new chat messages in the result (``messages`` convention) go to the thread
   (human -> user turn, ai -> assistant, tool -> internal), and **observe** submits
   observations taken from the result. Every write carries a deterministic idempotency
   key from (thread, checkpoint namespace, node, step, content), so a superstep retried
   from a checkpoint never records twice.

Memory errors on the *write* path always propagate (no silent data loss); on the
*read* path they propagate unless ``strict=False``, in which case the node runs with
``inject=None`` and the error is logged. Graphs must run with ``ainvoke``/``astream``:
the SDK is async.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from universal_memory import MemoryClient, MemoryContext
from universal_memory.models import ContextBundle, ObservationAck
from universal_memory_langgraph.lineage import Lineage, lineage_from_config, scope_fields
from universal_memory_langgraph.messages import MessageView, new_messages, trailing_human

log = logging.getLogger("universal_memory.langgraph")

Query = str | Callable[[Any], str | None]
Observer = str | Callable[[Any, Any], Any]


@dataclass(frozen=True)
class NodeResult:
    """What the wrapper did around one node execution (for tests and telemetry)."""

    node: str
    context: MemoryContext
    bundle: ContextBundle | None
    observations: tuple[ObservationAck, ...]
    recorded_messages: int


class LangGraphMemory:
    def __init__(
        self,
        client: MemoryClient,
        *,
        tenant_id: str,
        user_id: str | None = None,
        workspace_id: str | None = None,
        group_ids: list[str] | None = None,
        agent_group_id: str | None = None,
        thread_prefix: str = "",
        inject: str = "memory",
        token_budget: int | None = None,
        strict: bool = True,
        record_messages: bool = True,
    ) -> None:
        self.client = client
        self.defaults: dict[str, Any] = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "group_ids": list(group_ids or []),
            "agent_group_id": agent_group_id,
        }
        self.thread_prefix = thread_prefix
        self.inject = inject
        self.token_budget = token_budget
        self.strict = strict
        self.record_messages = record_messages
        self.last: NodeResult | None = None

    # -- context resolution -------------------------------------------------
    def lineage(self, config: Mapping[str, Any] | None) -> Lineage:
        return lineage_from_config(config)

    def context(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        agent: str | None = None,
        state: Any = None,
    ) -> MemoryContext:
        """The ``MemoryContext`` for this position in the graph. Inside a node, ``config``
        may be omitted (it is read from LangGraph's context variable); ``state`` lets the
        pending human message name the turn."""
        if config is None:
            config = _running_config()
        pending = trailing_human(state) if state is not None else []
        hint = pending[-1].id if pending and pending[-1].id else None
        fields = scope_fields(
            self.lineage(config),
            defaults=self.defaults,
            thread_prefix=self.thread_prefix,
            agent=agent,
            turn_hint=hint,
        )
        return self.client.bind(**fields)

    # -- node wrapping --------------------------------------------------------
    def wrap(
        self,
        node: Callable[..., Any],
        *,
        name: str | None = None,
        recall: Query | None = None,
        observe: Observer | None = None,
        observe_kind: str | None = None,
        observe_hints: Mapping[str, Any] | None = None,
        agent: str | None = None,
        require_evidence: bool = False,
        token_budget: int | None = None,
        record_messages: bool | None = None,
    ) -> Callable[[Any, Any], Awaitable[Any]]:
        """Return an async node that runs ``node`` with memory around it.

        ``recall``: a state key or ``fn(state) -> query`` (``None``/empty skips recall).
        ``observe``: a result key or ``fn(state, result) -> str | list[str] | dict | None``
        (a dict may carry ``content``, ``kind``, ``hints``, ``metadata``).
        ``agent``: run this node as an agent (RUN-scoped working memory, run id = the
        LangGraph task id, parent = the enclosing subgraph run).
        """
        node_name = name or getattr(node, "__name__", "node")
        accepts = _accepted_kwargs(node)
        record = self.record_messages if record_messages is None else record_messages
        budget = self.token_budget if token_budget is None else token_budget

        async def wrapped(state: Any, config=None) -> Any:
            config = config or _running_config()
            lineage = self.lineage(config)
            ctx = self.context(config, agent=agent, state=state)
            async with ctx:
                recorded = 0
                if record:  # the user's turn arrives as graph input, not as a node result
                    recorded += await self._record_input(ctx, lineage, state)
                bundle = await self._recall(ctx, state, recall, budget, require_evidence)
                inbound = _inject(state, self.inject, bundle) if recall is not None else state
                kwargs = _kwargs_for(accepts, config)
                result = node(inbound, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                if record:
                    recorded += await self._record(ctx, lineage, node_name, state, result)
                acks = await self._observe(
                    ctx, lineage, node_name, state, result, observe, observe_kind, observe_hints
                )
            self.last = NodeResult(node_name, ctx, bundle, tuple(acks), recorded)
            return result

        wrapped.__name__ = node_name
        wrapped.__qualname__ = getattr(node, "__qualname__", node_name)
        wrapped.__doc__ = getattr(node, "__doc__", None)
        return wrapped

    def node(self, **options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator form of :meth:`wrap`."""

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            return self.wrap(fn, **options)

        return deco

    # -- steps ------------------------------------------------------------------
    async def _recall(
        self,
        ctx: MemoryContext,
        state: Any,
        recall: Query | None,
        budget: int | None,
        require_evidence: bool,
    ) -> ContextBundle | None:
        if recall is None:
            return None
        query = _pick(state, recall) if isinstance(recall, str) else recall(state)
        if not query:
            return None
        try:
            return await ctx.context(
                str(query), token_budget=budget, require_evidence=require_evidence
            )
        except Exception:
            if self.strict or require_evidence:
                raise
            log.exception("memory recall failed; continuing without context")
            return None

    async def _record(
        self, ctx: MemoryContext, lineage: Lineage, node: str, state: Any, result: Any
    ) -> int:
        if not ctx.scope.thread_id:
            return 0
        count = 0
        for m in new_messages(state, result):
            key = self._key("msg", lineage, node, m.type, m.id or m.content)
            await self._send_message(ctx, m, key)
            count += 1
        return count

    async def _record_input(self, ctx: MemoryContext, lineage: Lineage, state: Any) -> int:
        """Record the pending human turn (the trailing human messages of the state). The key
        ignores node and step so the same turn seen by several nodes is recorded once."""
        if not ctx.scope.thread_id:
            return 0
        pending = trailing_human(state)
        for m in pending:
            key = self._key("in", Lineage(lineage.thread_id), "", m.id or m.content)
            await self._send_message(ctx, m, key)
        return len(pending)

    async def _send_message(self, ctx: MemoryContext, m: MessageView, key: str) -> None:
        meta: dict[str, Any] = {"langgraph_message_id": m.id} if m.id else {}
        if m.type == "human":
            await ctx.chat.user(m.content, attachments=None, idempotency_key=key, **meta)
        elif m.type == "ai":
            await ctx.chat.assistant(m.content, idempotency_key=key, **meta)
        else:  # tool / system / function -> internal, never in the visible history
            role = "TOOL" if m.type == "tool" else "SYSTEM" if m.type == "system" else "AGENT"
            await ctx.chat.internal(m.content, role=role, idempotency_key=key, **meta)

    async def _observe(
        self,
        ctx: MemoryContext,
        lineage: Lineage,
        node: str,
        state: Any,
        result: Any,
        observe: Observer | None,
        kind: str | None,
        hints: Mapping[str, Any] | None,
    ) -> list[ObservationAck]:
        if observe is None:
            return []
        raw = _pick(result, observe) if isinstance(observe, str) else observe(state, result)
        default_kind = kind or ("AGENT_RESULT" if ctx.scope.agent_id else "EVENT")
        acks: list[ObservationAck] = []
        for item in _as_observations(raw):
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            k = str(item.get("kind") or default_kind)
            key = self._key("obs", lineage, node, k, content)
            acks.append(
                await ctx.observe(
                    content,
                    kind=k,
                    idempotency_key=key,
                    hints={**(hints or {}), **(item.get("hints") or {})},
                    langgraph_node=node,
                    langgraph_path=lineage.path,
                    **(item.get("metadata") or {}),
                )
            )
        return acks

    @staticmethod
    def _key(prefix: str, lineage: Lineage, node: str, *parts: str) -> str:
        """Stable across checkpoint retries: the task id in the namespace is derived from
        (checkpoint, step, node), so the same superstep replays to the same key."""
        h = hashlib.blake2b(digest_size=16)
        for p in (lineage.thread_id or "", lineage.path, node, str(lineage.step or ""), *parts):
            h.update(p.encode())
            h.update(b"\x00")
        return f"lg-{prefix}-{h.hexdigest()}"


# -- helpers --------------------------------------------------------------------


def _running_config() -> Mapping[str, Any]:
    try:
        from langgraph.config import get_config
    except ImportError:  # pragma: no cover - langgraph is optional at import time
        return {}
    try:
        return get_config()
    except RuntimeError:
        return {}


def _accepted_kwargs(node: Callable[..., Any]) -> tuple[str, ...]:
    try:
        params = inspect.signature(node).parameters
    except (TypeError, ValueError):
        return ()
    return tuple(k for k in ("config", "runtime", "store", "writer") if k in params)


def _kwargs_for(accepts: tuple[str, ...], config: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in accepts:
        if k == "config":
            out[k] = config
        elif k == "runtime":
            from langgraph.runtime import get_runtime

            out[k] = get_runtime()
        elif k == "store":
            from langgraph.config import get_store

            out[k] = get_store()
        elif k == "writer":
            from langgraph.config import get_stream_writer

            out[k] = get_stream_writer()
    return out


def _pick(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _inject(state: Any, key: str, bundle: ContextBundle | None) -> Any:
    if isinstance(state, Mapping):
        return {**state, key: bundle}
    try:
        setattr(state, key, bundle)
    except AttributeError:  # frozen models: hand the bundle over via a shallow copy
        copy = getattr(state, "model_copy", None)
        if copy is not None:
            return copy(update={key: bundle})
    return state


def _as_observations(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [{"content": raw}]
    if isinstance(raw, Mapping):
        return [dict(raw)]
    if isinstance(raw, list | tuple):
        out: list[dict[str, Any]] = []
        for item in raw:
            out.extend(_as_observations(item))
        return out
    return [{"content": str(raw)}]
