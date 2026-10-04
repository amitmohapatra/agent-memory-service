"""The 90% path::

    memory = MemoryClient("http://memory-service:8080", api_key="...")
    ctx = memory.bind(user_id=..., thread_id=...)   # the key names the tenant
    pushed = await ctx.context(question)            # what the prompt gets: pushed.rendered
    ...
    await ctx.history.add([("USER", question), ("ASSISTANT", answer)])
    await ctx.verify(answer, bundle_id=pushed.bundle_id)

The verbs an agent uses every turn live on the context (``context``, ``remember``,
``update``, ``forget``, ``search``, ``history``, ``feedback``, ``record_tool``,
``tool_hints``, ``agent_tools``, ``call_agent_tool``, ``profile``, ``verify``, ``agent``);
everything else is under ``ctx.advanced``.

``MemoryClient`` is static (one per process). ``MemoryContext`` is per request and
immutable; ``contextvars`` propagate it within one async execution for convenience only.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Literal, Self, overload

import httpx

from trellis.memory.admin import AdminAPI, TenantAPI
from trellis.memory.advanced import AdvancedAPI
from trellis.memory.models import (
    AgentTool,
    ContextBundle,
    EvidenceRef,
    Feedback,
    FeedbackSource,
    FeedbackTargetKind,
    FeedbackVerdict,
    Lifetime,
    MemoryType,
    Message,
    MessageAck,
    MessageInfo,
    Page,
    ProfileBlock,
    PromptContext,
    RememberAck,
    Scope,
    SearchItem,
    SearchKind,
    SupersedeAck,
    ThreadInfo,
    ToolHints,
    ToolResult,
    ToolStatus,
    VerifyReport,
    Visibility,
)
from trellis.memory.transport import Transport

_current_context: ContextVar[MemoryContext | None] = ContextVar("trellis.memory_ctx", default=None)

#: How many tool candidates ``tool_hints`` and ``context(tools=...)`` ask for by default.
TOOL_HINTS_K = 8


def current_context() -> MemoryContext | None:
    """The MemoryContext bound in this async execution, if any."""
    return _current_context.get()


class MemoryClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        bearer_token: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._transport = Transport(
            base_url,
            api_key=api_key,
            bearer_token=bearer_token,
            timeout=timeout,
            max_retries=max_retries,
            client=http_client,
        )
        #: Platform administration (the bootstrap key): onboarding tenants.
        self.admin = AdminAPI(self)
        #: Administration of the key's own tenant: keys, workspaces, model keys, the read audit.
        self.tenant = TenantAPI(self)

    def administer(self, tenant_id: str) -> TenantAPI:
        """Tenant administration for a named tenant - the platform key's way in, and the
        development key's. A tenant admin key needs no name: use :attr:`tenant`."""
        return TenantAPI(self, tenant_id=tenant_id)

    def bind(self, **scope: Any) -> MemoryContext:
        """Create a per-request context. Accepts every :class:`Scope` field; ``tenant_id``
        may be omitted when the API key names the tenant."""
        return MemoryContext(self, Scope(**scope))

    async def health(self) -> dict[str, Any]:
        """Readiness (dependencies pinged); ``alive()`` is the cheap liveness probe."""
        return await self._transport.request("GET", "/health/ready")

    async def alive(self) -> dict[str, Any]:
        return await self._transport.request("GET", "/health/live")

    async def version(self) -> dict[str, Any]:
        return await self._transport.request("GET", "/version")

    async def metrics(self) -> str:
        """The Prometheus exposition of the worker that answered, as text."""
        return await self._transport.request_text("GET", "/metrics")

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    @property
    def transport(self) -> Transport:
        return self._transport


class MemoryContext:
    """Per-request handle. Immutable; ``agent`` creates child contexts."""

    def __init__(self, client: MemoryClient, scope: Scope) -> None:
        self._client = client
        self.scope = scope
        #: ``await ctx.history()`` reads the thread; ``.add`` appends to it
        self.history = HistoryAPI(self)
        #: ``await ctx.feedback(...)`` records a judgement; ``.get`` / ``.list_for`` read them
        self.feedback = FeedbackAPI(self)
        #: ``await ctx.profile()`` lists the pinned blocks; ``.edit`` changes one
        self.profile = ProfileAPI(self)
        self.advanced = AdvancedAPI(self)
        self._token: Any = None

    @property
    def client(self) -> MemoryClient:
        return self._client

    # -- context manager: propagate via contextvars ---------------------
    async def __aenter__(self) -> Self:
        self._token = _current_context.set(self)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._token is not None:
            _current_context.reset(self._token)
            self._token = None

    def agent(
        self, agent_id: str, *, agent_run_id: str | None = None, agent_group_id: str | None = None
    ) -> MemoryContext:
        """Context for an agent run acting for this user. Each call is a new run: the agent's
        working notes (``Visibility.RUN``) are readable by this run and by the runs it derives
        with ``.agent(...)`` — hand-off flows down, a child reports up to the run that spawned
        it, and nothing reaches siblings, the user, other agents, or this agent's LATER runs.

        Anything the agent should still know next time is not RUN: use ``visibility="USER"``
        for its user, or ``"PRIVATE"`` for its own durable store (what a scheduled job with no
        user uses). Peers that must collaborate share ``agent_group_id`` and
        ``visibility="AGENT_GROUP"``, which is not bounded by the run tree."""
        return MemoryContext(
            self._client,
            self.scope.model_copy(
                update={
                    "agent_id": agent_id,
                    "agent_run_id": agent_run_id or f"run_{uuid.uuid4().hex}",
                    "agent_group_id": agent_group_id or self.scope.agent_group_id,
                    "parent_agent_run_id": self.scope.agent_run_id,
                }
            ),
        )

    # -- push -----------------------------------------------------------
    @overload
    async def context(
        self,
        query: str,
        *,
        token_budget: int | None = ...,
        tools: Sequence[str] | None = ...,
        window: bool = ...,
        document_ids: Sequence[str] | None = ...,
        format: Literal["prompt"] = ...,
        debug: bool = ...,
    ) -> PromptContext: ...

    @overload
    async def context(
        self,
        query: str,
        *,
        token_budget: int | None = ...,
        tools: Sequence[str] | None = ...,
        window: bool = ...,
        document_ids: Sequence[str] | None = ...,
        format: Literal["full"],
        debug: bool = ...,
    ) -> ContextBundle: ...

    async def context(
        self,
        query: str,
        *,
        token_budget: int | None = None,
        tools: Sequence[str] | None = None,
        window: bool = True,
        document_ids: Sequence[str] | None = None,
        format: Literal["prompt", "full"] = "prompt",
        debug: bool = False,
    ) -> PromptContext | ContextBundle:
        """The context for this turn: the pinned profile, the thread's summary, its recent
        messages (unless ``window=False``: the framework keeps its own history), and the
        memories, documents and facts that answer ``query``, rendered for a prompt within
        ``token_budget``. Items are cited by handle ([m1], [d2]...), which ``update``,
        ``forget`` and ``verify`` accept within the bundle.

        ``tools`` - the agent's own tools - adds the procedures learned for the task and the
        tools that fit, each with its confidence (0..1), the argument values found and the
        required ones missing; the prompt form returns the fitting tools in ``tools``.
        ``format="full"`` returns the same content as structured data, without the rendering."""
        payload: dict[str, Any] = {
            "query": query,
            "scope": self.scope_payload(),
            "window": window,
            "format": format,
            "debug": debug,
        }
        if token_budget is not None:
            payload["token_budget"] = token_budget
        if document_ids is not None:
            payload["document_ids"] = list(document_ids)
        if tools is not None:
            payload["tools"] = {"available": list(tools), "k": TOOL_HINTS_K}
        data = await self._request("POST", "/v1/context", json=payload)
        if format == "full":
            return ContextBundle.model_validate(data)
        return PromptContext.model_validate(data)

    # -- memory ----------------------------------------------------------
    async def remember(
        self,
        content: str,
        *,
        memory_type: MemoryType = "SEMANTIC",
        lifetime: Lifetime = "LONG_TERM",
        visibility: Visibility | None = None,
        subject: str | None = None,
        entities: Sequence[str] = (),
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        idempotency_key: str | None = None,
        **metadata: Any,
    ) -> RememberAck:
        """Store ``content`` verbatim as one memory, now. The same content in the same scope
        returns the memory already stored, with ``deduplicated=True``."""
        payload: dict[str, Any] = {
            "scope": self.scope_payload(),
            "content": content,
            "memory_type": memory_type,
            "lifetime": lifetime,
            "entities": list(entities),
            "custom_metadata": metadata,
        }
        for name, value in (("visibility", visibility), ("subject", subject)):
            if value is not None:
                payload[name] = value
        for name, when in (("valid_from", valid_from), ("valid_to", valid_to)):
            if when is not None:
                payload[name] = when.isoformat()
        key = idempotency_key or _default_key("mem", self.scope, memory_type, content)
        data = await self._request("POST", "/v1/memories", json=payload, idempotency_key=key)
        return RememberAck.model_validate(data)

    async def update(
        self,
        memory: str,
        content: str,
        *,
        reason: str = "",
        bundle_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> SupersedeAck:
        """Replace a memory (its id, or its handle in the context ``bundle_id`` - by default
        this run's latest) with a new version: the new one is current from now, the old one
        is closed (``SUPERSEDED``) and still readable in a temporal view."""
        body: dict[str, Any] = {"scope": self.scope_payload(), "content": content}
        if reason:
            body["reason"] = reason
        if bundle_id:
            body["bundle_id"] = bundle_id
        data = await self._request(
            "POST",
            f"/v1/memories/{memory}/supersede",
            json=body,
            idempotency_key=idempotency_key or _default_key("sup", self.scope, memory, content),
        )
        return SupersedeAck.model_validate(data)

    async def forget(self, memory: str, *, bundle_id: str | None = None) -> None:
        """Forget a memory (its id, or its handle in the context ``bundle_id``; soft delete,
        audited): it stops being retrieved."""
        await self._request(
            "DELETE",
            f"/v1/memories/{memory}",
            params={"bundle_id": bundle_id} if bundle_id else None,
            idempotency_key=f"del-{bundle_id or ''}-{memory}",
        )

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        kinds: Sequence[SearchKind] | None = None,
        time_from: datetime | None = None,
        time_to: datetime | None = None,
        as_of: datetime | None = None,
        known_at: datetime | None = None,
        document_ids: Sequence[str] | None = None,
        debug: bool = False,
    ) -> list[SearchItem]:
        """Ranked items for ``query``: memories and document passages by default; ``kinds``
        also reads document summaries, earlier conversations (``episode``) and this
        thread's messages. ``time_from``/``time_to`` keep what was observed within the range
        (before anything is ranked). ``as_of`` reads memories as they were true then and
        ``known_at`` as they were known then, including ones replaced since."""
        payload: dict[str, Any] = {
            "query": query,
            "scope": self.scope_payload(),
            "limit": limit,
            "debug": debug,
        }
        if kinds is not None:
            payload["kinds"] = list(kinds)
        for name, when in (
            ("time_from", time_from),
            ("time_to", time_to),
            ("as_of", as_of),
            ("known_at", known_at),
        ):
            if when is not None:
                payload[name] = when.isoformat()
        if document_ids is not None:
            payload["document_ids"] = list(document_ids)
        data = await self._request("POST", "/v1/recall", json=payload)
        return [SearchItem.model_validate(i) for i in data.get("items", [])]

    async def verify(
        self, answer: str, *, bundle_id: str, run_id: str | None = None
    ) -> VerifyReport:
        """Verify ``answer`` claim by claim against the context it was given (``bundle_id``);
        its handle citations ([m1]) resolve within it. With a run (``run_id``, or this
        context's agent run) the verdict is recorded as the judge's RUN feedback."""
        body: dict[str, Any] = {
            "scope": self.scope_payload(),
            "bundle_id": bundle_id,
            "answer": answer,
        }
        if run_id is not None:
            body["run_id"] = run_id
        return VerifyReport.model_validate(await self._request("POST", "/v1/verify", json=body))

    # -- tools -----------------------------------------------------------
    async def record_tool(
        self,
        tool: str,
        args: dict[str, Any],
        *,
        output: Any = None,
        output_summary: str | None = None,
        status: ToolStatus = "ok",
        error_class: str | None = None,
        latency_ms: float | None = None,
        cost: float | None = None,
        task: str | None = "",
        step: int | None = None,
        sub_calls: list[dict[str, Any]] | None = None,
        visibility: Visibility = "PRIVATE",
    ) -> ToolResult:
        """Record one tool call this run made (idempotent on run + step + tool + arguments).
        The service never runs a tool; it learns from what was recorded."""
        data = await self._request(
            "POST",
            "/v1/tools/invocations",
            json={
                "scope": self.scope_payload(),
                "tool": tool,
                "args": args,
                "output": output,
                "output_summary": output_summary,
                "status": status,
                "error_class": error_class,
                "latency_ms": latency_ms,
                "cost": cost,
                "task": task or "",
                "step": step,
                "sub_calls": sub_calls or [],
                "visibility": visibility,
            },
        )
        return ToolResult.model_validate(data)

    async def tool_hints(
        self, task: str, *, available: Sequence[str] | None = None, k: int = TOOL_HINTS_K
    ) -> ToolHints:
        """Which tools fit ``task`` (among ``available`` when given), best first: each with
        its confidence (0..1), success rate, whether it is the plan's next step, the argument
        values found and the required ones missing; and the learned plan."""
        data = await self._request(
            "POST",
            "/v1/tools/hints",
            json={
                "scope": self.scope_payload(),
                "task": task,
                "available": list(available) if available is not None else None,
                "k": k,
            },
        )
        return ToolHints.model_validate(data)

    # -- pull (the memory tools an agent calls) ----------------------------
    async def agent_tools(self) -> list[AgentTool]:
        """The memory tools an agent may call, with their JSON input schemas."""
        data = await self._request("GET", "/v1/agent-tools")
        return [AgentTool.model_validate(t) for t in data.get("tools", [])]

    async def call_agent_tool(
        self, name: str, args: Mapping[str, Any], *, toolbox: Sequence[str] | None = None
    ) -> Any:
        """Run one memory tool in this scope and return its ``result``. ``toolbox`` - the
        agent's own tools - is what ``tool_search`` chooses among."""
        body: dict[str, Any] = {"scope": self.scope_payload(), "args": dict(args)}
        if toolbox is not None:
            body["toolbox"] = list(toolbox)
        data = await self._request("POST", f"/v1/agent-tools/{name}", json=body)
        return data.get("result")

    # -- plumbing -------------------------------------------------------
    def scope_payload(self) -> dict[str, Any]:
        """The scope as a request body carries it. The trace id travels as traceparent (or
        as the correlation id), never in the body."""
        return self.scope.model_dump(mode="json", exclude_none=True, exclude={"trace_id"})

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        return await self._client.transport.request(method, path, scope=self.scope, **kwargs)


class FeedbackAPI:
    """Judgements on what the platform did - the learning signal: a verdict on a memory
    reinforces, retracts or corrects it; on a run it decides the run's outcome (a person over
    the judge over the run's own status) and moves the confidence of the memories it cites;
    on a tool call it counts toward approval patterns; on a procedure it can retire it.

    A vote waits for a tenant admin (``review.state == "pending"``) before it changes
    anything; ``pending``, ``approve`` and ``dismiss`` are the review queue."""

    def __init__(self, ctx: MemoryContext) -> None:
        self._ctx = ctx

    async def __call__(
        self,
        target: Any,
        target_id: str | None = None,
        verdict: FeedbackVerdict | None = None,
        *,
        correction: Any = None,
        score: float | None = None,
        comment: str | None = None,
        reviewer: str | None = None,
        source: FeedbackSource = "human",
        evidence_refs: Sequence[EvidenceRef] = (),
        metadata: dict[str, Any] | None = None,
        feedback_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> Feedback:
        """Record one judgement: ``feedback(record)`` sends a ``trellis.contracts.Feedback``
        (any model or mapping of that shape) as it is; ``feedback(kind, target_id, verdict,
        ...)`` builds one, its identity taken from this context. A retry with the same
        ``feedback_id`` returns the stored record."""
        if not isinstance(target, str):
            body = _record(target)
        elif target_id is None or verdict is None:
            raise ValueError("feedback(kind, target_id, verdict) needs all three")
        else:
            scope = self._ctx.scope
            body = {
                "feedback_id": feedback_id,
                "tenant_id": scope.tenant_id,
                "workspace_id": scope.workspace_id,
                "user_id": scope.user_id,
                "agent_id": scope.agent_id,
                "agent_run_id": scope.agent_run_id,
                "target_kind": target,
                "target_id": target_id,
                "verdict": verdict,
                "correction": correction,
                "score": score,
                "comment": comment,
                "reviewer": reviewer,
                "source": source,
                "evidence_refs": [
                    e.model_dump(mode="json", exclude_none=True) for e in evidence_refs
                ],
                "metadata": metadata or {},
            }
        body = {k: v for k, v in body.items() if v is not None and k != "projection"}
        data = await self._ctx._request(
            "POST", "/v1/feedback", json=body, idempotency_key=idempotency_key
        )
        return Feedback.model_validate(data)

    async def get(self, feedback_id: str) -> Feedback:
        return Feedback.model_validate(
            await self._ctx._request("GET", f"/v1/feedback/{feedback_id}")
        )

    async def list_for(
        self,
        target_kind: FeedbackTargetKind | str,
        target_id: str,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[Feedback]:
        """One page of the feedback on a target, newest first; ``page_for`` also returns the
        cursor of the next page."""
        return (await self.page_for(target_kind, target_id, limit=limit, cursor=cursor)).items

    async def page_for(
        self,
        target_kind: FeedbackTargetKind | str,
        target_id: str,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> Page[Feedback]:
        params: dict[str, Any] = {
            "target_kind": str(getattr(target_kind, "value", target_kind)),
            "target_id": target_id,
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor
        data = await self._ctx._request("GET", "/v1/feedback", params=params)
        return Page[Feedback](
            items=[Feedback.model_validate(f) for f in data.get("feedback", [])],
            next_cursor=data.get("next_cursor"),
        )

    async def pending(self, *, limit: int = 100, cursor: str | None = None) -> Page[Feedback]:
        """The review queue (the tenant's administrator key): verdicts that change nothing
        until approved, newest first, each with its author's ``author_record``."""
        params: dict[str, Any] = {"review": "pending", "limit": limit}
        if cursor:
            params["cursor"] = cursor
        data = await self._ctx._request("GET", "/v1/feedback", params=params)
        return Page[Feedback](
            items=[Feedback.model_validate(f) for f in data.get("feedback", [])],
            next_cursor=data.get("next_cursor"),
        )

    async def approve(self, feedback_id: str, *, note: str | None = None) -> Feedback:
        """Apply a pending verdict as if it had just arrived (the tenant's administrator key)."""
        return await self._review(feedback_id, "approve", note)

    async def dismiss(self, feedback_id: str, *, note: str | None = None) -> Feedback:
        """Keep a pending verdict for statistics; it is never applied."""
        return await self._review(feedback_id, "dismiss", note)

    async def _review(self, feedback_id: str, action: str, note: str | None) -> Feedback:
        body = {"note": note} if note is not None else {}
        return Feedback.model_validate(
            await self._ctx._request("POST", f"/v1/feedback/{feedback_id}/{action}", json=body)
        )


def _record(feedback: Any) -> dict[str, Any]:
    if isinstance(feedback, Mapping):
        return dict(feedback)
    dump = getattr(feedback, "model_dump", None)
    if dump is None:
        raise TypeError("feedback() takes a Feedback record (a pydantic model or a mapping)")
    return dict(dump(mode="json"))


#: ``profile.edit(source_query=...)`` not given: the block keeps its standing question.
_KEEP: Any = object()


class ProfileAPI:
    """Pinned profile blocks for this scope: ``user``, ``agent`` and ``workspace`` (and
    ``<level>.<name>`` blocks beside them), always part of the pushed context."""

    def __init__(self, ctx: MemoryContext) -> None:
        self._ctx = ctx

    async def __call__(self) -> list[ProfileBlock]:
        data = await self._ctx._request("GET", "/v1/profile")
        return [ProfileBlock.model_validate(b) for b in data.get("blocks", [])]

    async def edit(
        self,
        block: str,
        new: str | None = None,
        *,
        old: str = "",
        source_query: str | None = _KEEP,
    ) -> ProfileBlock:
        """Replace ``old`` with ``new`` in the block (``ConflictError`` when ``old`` is not in
        it: read the block again and retry), or the whole text when ``old`` is empty.
        ``source_query`` sets a standing question the service answers into the block now and
        every hour (None removes it)."""
        body: dict[str, Any] = {"scope": self._ctx.scope_payload(), "old": old}
        if new is not None:
            body["new"] = new
        if source_query is not _KEEP:
            body["source_query"] = source_query
        data = await self._ctx._request("PATCH", f"/v1/profile/{block}", json=body)
        return ProfileBlock.model_validate(data)


def _message(message: Message | Mapping[str, Any] | tuple[str, str]) -> dict[str, Any]:
    if isinstance(message, tuple):
        role, content = message
        message = Message(role=role, content=content)  # type: ignore[arg-type]
    elif not isinstance(message, Message):
        message = Message.model_validate(dict(message))
    return message.model_dump(mode="json", exclude_none=True)


class HistoryAPI:
    """The thread's transcript: ``await ctx.history()`` reads it, ``add`` appends to it.
    Without a ``thread_id`` in the scope, an agent run's messages go to the thread named by
    its run id."""

    def __init__(self, ctx: MemoryContext) -> None:
        self._ctx = ctx

    def _thread(self) -> str | None:
        return self._ctx.scope.thread_id or self._ctx.scope.agent_run_id

    async def __call__(
        self, *, limit: int = 50, include_internal: bool = False
    ) -> list[MessageInfo]:
        """The thread's latest messages, oldest first (empty without a thread)."""
        thread_id = self._thread()
        if not thread_id:
            return []
        data = await self._ctx._request(
            "GET",
            f"/v1/threads/{thread_id}/messages",
            params={"limit": limit, "include_internal": include_internal},
        )
        return [MessageInfo.model_validate(m) for m in data.get("messages", [])]

    async def add(
        self,
        messages: Sequence[Message | Mapping[str, Any] | tuple[str, str]],
        *,
        idempotency_key: str | None = None,
    ) -> list[MessageAck]:
        """Append messages (``Message``, a mapping of its fields, or ``(role, content)``) in
        one durable write; ``role="EVENT"`` tells the service something that happened. A
        retry with the same ``idempotency_key`` returns the first acknowledgements."""
        body = {"scope": self._ctx.scope_payload(), "messages": [_message(m) for m in messages]}
        scope = self._ctx.scope
        # A turn makes lineage + content the batch's identity. Without one, two identical
        # batches ("ok") are two appends: a fresh key per call, reused across its retries.
        key = idempotency_key or (
            _default_key("msgs", scope, json.dumps(body["messages"], sort_keys=True))
            if scope.turn_id
            else f"msgs-{uuid.uuid4().hex}"
        )
        data = await self._ctx._request("POST", "/v1/messages", json=body, idempotency_key=key)
        return [MessageAck.model_validate(a) for a in data.get("messages", [])]

    async def thread(self) -> ThreadInfo:
        """The thread, with its durable summary once it has one."""
        data = await self._ctx._request("GET", f"/v1/threads/{self._thread()}")
        return ThreadInfo.model_validate(data)

    async def update(
        self, *, title: str | None = None, metadata: dict[str, Any] | None = None
    ) -> ThreadInfo:
        """Set the thread's title and metadata (the thread is created when it does not exist
        yet: messages create threads on demand, this is how one gets a title first)."""
        body: dict[str, Any] = {"scope": self._ctx.scope_payload()}
        if title is not None:
            body["title"] = title
        if metadata is not None:
            body["custom_metadata"] = metadata
        data = await self._ctx._request("PATCH", f"/v1/threads/{self._thread()}", json=body)
        return ThreadInfo.model_validate(data)

    async def message(self, message_id: str) -> MessageInfo:
        data = await self._ctx._request("GET", f"/v1/messages/{message_id}")
        return MessageInfo.model_validate(data)

    async def delete(self) -> None:
        """Soft-delete the thread (owner or tenant admin): messages stop being listed and
        retrieved; archived segments are kept for the retention period."""
        tid = self._thread()
        await self._ctx._request("DELETE", f"/v1/threads/{tid}", idempotency_key=f"delthr-{tid}")


def _default_key(prefix: str, scope: Scope, *parts: str) -> str:
    """Deterministic idempotency key from lineage + content so retries never duplicate."""
    h = hashlib.blake2b(digest_size=16)
    for p in (
        scope.tenant_id or "",
        scope.thread_id or "",
        scope.session_id or "",
        scope.turn_id or "",
        scope.agent_run_id or "",
        *parts,
    ):
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return f"{prefix}-{h.hexdigest()}"
