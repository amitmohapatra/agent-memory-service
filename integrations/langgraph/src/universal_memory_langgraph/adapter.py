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

import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from universal_memory import MemoryClient, MemoryContext
from universal_memory.integrations.core import MemoryHooks, as_observations, idempotency_key
from universal_memory.models import ContextBundle, GroundingReport, ObservationAck
from universal_memory_langgraph.lineage import Lineage, lineage_from_config, scope_fields
from universal_memory_langgraph.messages import new_messages, trailing_human

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
    grounding: GroundingReport | None = None
    max_hallucination_rate: float = 0.0

    @property
    def evidence_complete(self) -> bool | None:
        """Whether the node's answer passed the grounding gate (``None``: not verified)."""
        if self.grounding is None:
            return None
        return self.grounding.per_claim_hallucination_rate <= self.max_hallucination_rate


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
        max_hallucination_rate: float = 0.0,
    ) -> None:
        self.client = client
        self.max_hallucination_rate = max_hallucination_rate
        self.hooks = MemoryHooks(
            client,
            namespace="lg",
            defaults={
                "tenant_id": tenant_id,
                "user_id": user_id,
                "workspace_id": workspace_id,
                "group_ids": list(group_ids or []),
                "agent_group_id": agent_group_id,
            },
            token_budget=token_budget,
            strict=strict,
        )
        self.defaults = self.hooks.defaults
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
        verify_answer: bool = False,
        max_hallucination_rate: float | None = None,
        token_budget: int | None = None,
        record_messages: bool | None = None,
    ) -> Callable[[Any, Any], Awaitable[Any]]:
        """Return an async node that runs ``node`` with memory around it.

        ``recall``: a state key or ``fn(state) -> query`` (``None``/empty skips recall).
        ``observe``: a result key or ``fn(state, result) -> str | list[str] | dict | None``
        (a dict may carry ``content``, ``kind``, ``hints``, ``metadata``).
        ``agent``: run this node as an agent (RUN-scoped working memory, run id = the
        LangGraph task id, parent = the enclosing subgraph run).
        ``verify_answer``: run the grounding cascade on the node's assistant message(s)
        against the recalled bundle before they are recorded; the report lands in
        ``state[inject]["grounding"]`` and an answer whose per-claim hallucination rate
        exceeds ``max_hallucination_rate`` (default 0.0) is recorded with
        ``evidence_complete=False`` instead of as verified.
        """
        node_name = name or getattr(node, "__name__", "node")
        accepts = _accepted_kwargs(node)
        record = self.record_messages if record_messages is None else record_messages
        budget = self.token_budget if token_budget is None else token_budget
        max_rate = (
            self.max_hallucination_rate if max_hallucination_rate is None else max_hallucination_rate
        )

        async def wrapped(state: Any, config=None) -> Any:
            config = config or _running_config()
            lineage = self.lineage(config)
            ctx = self.context(config, agent=agent, state=state)
            async with ctx:
                recorded = 0
                if record:  # the user's turn arrives as graph input, not as a node result
                    recorded += await self._record_input(ctx, lineage, state)
                query = _query(state, recall)
                bundle = await self._recall(ctx, query, budget, require_evidence)
                inbound = _inject(state, self.inject, bundle) if recall is not None else state
                kwargs = _kwargs_for(accepts, config)
                result = node(inbound, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                new = new_messages(state, result)
                grounding = None
                if verify_answer:
                    grounding = await self._verify(ctx, new, bundle, query)
                    if grounding is not None:
                        result = _inject(
                            result, self.inject, {"bundle": bundle, "grounding": grounding}
                        )
                written = await self.hooks.after_run(
                    ctx,
                    key_parts=_key_parts(lineage, node_name),
                    messages=[
                        m.as_message(**_message_meta(m, grounding, max_rate))
                        for m in (new if record else [])
                    ],
                    observations=self._observations(state, result, observe),
                    default_kind=observe_kind,
                    hints=observe_hints,
                    langgraph_node=node_name,
                    langgraph_path=lineage.path,
                )
                recorded += written.recorded_messages
            self.last = NodeResult(
                node_name, ctx, bundle, written.observations, recorded, grounding, max_rate
            )
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
        query: str | None,
        budget: int | None,
        require_evidence: bool,
    ) -> ContextBundle | None:
        if query is None:
            return None
        return await self.hooks.before_run(
            ctx, query, require_evidence=require_evidence, token_budget=budget
        )

    async def _verify(
        self,
        ctx: MemoryContext,
        messages: Sequence[Any],
        bundle: ContextBundle | None,
        query: str | None,
    ) -> GroundingReport | None:
        """The grounding cascade over the node's assistant message(s): against the recalled
        bundle when there is one, else against a fresh retrieval for the recall query (or
        the answer itself). Failures propagate like reads unless ``strict=False``."""
        answer = "\n".join(m.content for m in messages if m.type == "ai").strip()
        if not answer:
            return None
        try:
            if bundle is not None:
                return await ctx.verify(answer, bundle=bundle)
            return await ctx.verify(answer, query=str(query or answer))
        except Exception:
            if self.strict:
                raise
            log.exception("answer verification failed; recording without a grounding report")
            return None

    async def _record_input(self, ctx: MemoryContext, lineage: Lineage, state: Any) -> int:
        """Record the pending human turn (the trailing human messages of the state). The key
        ignores node and step so the same turn seen by several nodes is recorded once."""
        if not ctx.scope.thread_id:
            return 0
        pending = trailing_human(state)
        for m in pending:
            key = self._key("in", Lineage(lineage.thread_id), "", m.id or m.content)
            meta = {"langgraph_message_id": m.id} if m.id else {}
            await self.hooks.record_message(ctx, m.as_message(**meta), key)
        return len(pending)

    @staticmethod
    def _observations(state: Any, result: Any, observe: Observer | None) -> list[Any]:
        if observe is None:
            return []
        raw = _pick(result, observe) if isinstance(observe, str) else observe(state, result)
        return list(as_observations(raw))

    @staticmethod
    def _key(prefix: str, lineage: Lineage, node: str, *parts: str) -> str:
        """Stable across checkpoint retries: the task id in the namespace is derived from
        (checkpoint, step, node), so the same superstep replays to the same key."""
        return idempotency_key("lg", prefix, *_key_parts(lineage, node), *parts)


def _key_parts(lineage: Lineage, node: str) -> tuple[str, str, str, str]:
    return (lineage.thread_id or "", lineage.path, node, str(lineage.step or ""))


def _query(state: Any, recall: Query | None) -> str | None:
    if recall is None:
        return None
    query = _pick(state, recall) if isinstance(recall, str) else recall(state)
    return str(query) if query else None


def _message_meta(
    message: Any, grounding: GroundingReport | None, max_rate: float
) -> dict[str, Any]:
    meta: dict[str, Any] = {"langgraph_message_id": message.id} if message.id else {}
    if grounding is not None and message.type == "ai":
        meta["grounding"] = {
            "evidence_complete": grounding.per_claim_hallucination_rate <= max_rate,
            "hallucination_rate": grounding.per_claim_hallucination_rate,
            "claims": len(grounding.claims),
            "contradicted": grounding.contradicted,
            "unsupported": grounding.unsupported,
            "representative": grounding.representative,
        }
    return meta


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
