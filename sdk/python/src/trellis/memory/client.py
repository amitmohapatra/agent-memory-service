"""The 90% path::

    memory = MemoryClient("http://memory-service:8080", api_key="...")
    ctx = memory.bind(tenant_id=..., user_id=..., thread_id=...)  # session/turn optional
    await ctx.chat.user(message)
    bundle = await ctx.context(message)          # push: what the prompt gets
    ...
    await ctx.chat.assistant(answer)

The verbs an agent uses every turn live on the context (``context``, ``remember``,
``update``, ``forget``, ``search``, ``history``, ``observe``, ``feedback``, ``record_tool``,
``outcome``, ``tool_hints``, ``agent_tools``, ``call_agent_tool``, ``profile``, ``summary``,
``verify``); everything else is under ``ctx.advanced``.

``MemoryClient`` is static (one per process). ``MemoryContext`` is per request and
immutable; ``contextvars`` propagate it within one async execution for convenience only.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Self

import httpx

from trellis.memory.admin import AdminAPI, TenantAPI
from trellis.memory.advanced import AdvancedAPI
from trellis.memory.errors import InsufficientEvidence, NotFoundError
from trellis.memory.models import (
    AgentTool,
    ContextBundle,
    ContextItem,
    EvidenceRef,
    Feedback,
    FeedbackSource,
    FeedbackTargetKind,
    FeedbackVerdict,
    GroundingReport,
    Lifetime,
    MemoryType,
    MessageAck,
    MessageInfo,
    MessageKind,
    MessageRole,
    ObservationAck,
    ObservationKind,
    Page,
    ProfileBlock,
    RecallKind,
    RememberAck,
    RunOutcome,
    Scope,
    SupersedeAck,
    ThreadInfo,
    ThreadSummary,
    ToolHints,
    ToolResult,
    ToolStatus,
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
        #: Administration of the key's own tenant: keys, workspaces, groups, the read audit.
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
    """Per-request handle. Immutable; ``derive``/``agent`` create child contexts."""

    def __init__(self, client: MemoryClient, scope: Scope) -> None:
        self._client = client
        self.scope = scope
        self.chat = ChatAPI(self)
        #: ``await ctx.feedback(record)``; ``.get`` / ``.list_for`` / ``.page_for`` read it back
        self.feedback = FeedbackAPI(self)
        #: ``await ctx.profile()`` lists the pinned blocks; ``.set`` / ``.edit`` change one
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

    def derive(self, **changes: Any) -> MemoryContext:
        """Child context (e.g. for an internal agent run): same lineage, new agent fields."""
        return MemoryContext(self._client, self.scope.model_copy(update=changes))

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
        return self.derive(
            agent_id=agent_id,
            agent_run_id=agent_run_id or f"run_{uuid.uuid4().hex}",
            agent_group_id=agent_group_id or self.scope.agent_group_id,
            parent_agent_run_id=self.scope.agent_run_id,
        )

    # -- push -----------------------------------------------------------
    async def context(
        self,
        query: str,
        *,
        token_budget: int | None = None,
        tools: Mapping[str, Any] | None = None,
        since_revision: int | None = None,
        require_evidence: bool = False,
        use_llm: bool | None = None,
        **options: Any,
    ) -> ContextBundle:
        """Bounded, ranked context for this turn: memories, knowledge, the pinned profile,
        the thread summary and the procedures learned for the task, rendered for a prompt.

        ``tools={"available": [names] | None, "k": 8}`` adds tool hints (``bundle.tools``).
        ``since_revision`` (a previous bundle's ``revision``) lists only what changed since
        (``bundle.delta``). With ``require_evidence=True`` an ``INSUFFICIENT`` evidence report
        raises :class:`InsufficientEvidence`. ``use_llm`` omitted follows the model policy."""
        payload: dict[str, Any] = {"query": query, "scope": self.scope_payload(), **options}
        for name, value in (
            ("use_llm", use_llm),
            ("token_budget", token_budget),
            ("since_revision", since_revision),
        ):
            if value is not None:
                payload[name] = value
        if tools is not None:
            payload["tools"] = {"k": TOOL_HINTS_K, **dict(tools)}
        bundle = ContextBundle.model_validate(
            await self._request("POST", "/v1/context", json=payload)
        )
        if require_evidence and bundle.evidence.status == "INSUFFICIENT":
            raise InsufficientEvidence(
                "no sufficient evidence was retrieved for this query",
                code="INSUFFICIENT_EVIDENCE",
                status=200,
                details={"notes": list(bundle.evidence.notes)},
            )
        return bundle

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
        """Store ``content`` verbatim as one memory, now (``observe`` is for evidence the
        service learns from). The same content in the same scope returns the memory already
        stored, with ``deduplicated=True``."""
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
        self, memory_id: str, content: str, *, reason: str, idempotency_key: str | None = None
    ) -> SupersedeAck:
        """Replace a memory with a new version: the new one is current from now, the old one
        is closed (``SUPERSEDED``) and still readable in a temporal view."""
        data = await self._request(
            "POST",
            f"/v1/memories/{memory_id}/supersede",
            json={"scope": self.scope_payload(), "content": content, "reason": reason},
            idempotency_key=idempotency_key or _default_key("sup", self.scope, memory_id, content),
        )
        return SupersedeAck.model_validate(data)

    async def forget(self, memory_id: str) -> None:
        """Forget a memory (soft delete, audited): it stops being retrieved."""
        await self._request(
            "DELETE", f"/v1/memories/{memory_id}", idempotency_key=f"del-{memory_id}"
        )

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        kinds: Sequence[RecallKind] | None = None,
        use_llm: bool | None = None,
        **options: Any,
    ) -> list[ContextItem]:
        """Ranked, scope-filtered evidence (chunks and memories) without bundle assembly.
        ``kinds`` narrows what is searched: chunk (document passages), memory, summary."""
        payload = {"query": query, "scope": self.scope_payload(), "limit": limit, **options}
        if use_llm is not None:
            payload["use_llm"] = use_llm
        if kinds is not None:
            payload["kinds"] = list(kinds)
        data = await self._request("POST", "/v1/recall", json=payload)
        return [ContextItem.model_validate(m) for m in data.get("results", [])]

    async def history(
        self, *, limit: int = 50, include_internal: bool = False
    ) -> list[MessageInfo]:
        """The thread's latest messages, oldest first (empty without a thread)."""
        thread_id = self.scope.thread_id
        if not thread_id:
            return []
        data = await self._request(
            "GET",
            f"/v1/threads/{thread_id}/messages",
            params={"limit": limit, "include_internal": include_internal},
        )
        return [MessageInfo.model_validate(m) for m in data.get("messages", [])]

    async def summary(self) -> ThreadSummary | None:
        """The thread's durable summary, or None when the thread has none yet."""
        thread_id = self.scope.thread_id
        if not thread_id:
            return None
        try:
            data = await self._request("GET", f"/v1/threads/{thread_id}/summary")
        except NotFoundError:
            return None
        return ThreadSummary.model_validate(data)

    async def observe(
        self,
        content: str,
        *,
        kind: ObservationKind = "EVENT",
        idempotency_key: str | None = None,
        hints: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> ObservationAck:
        """Raw evidence the service learns from, asynchronously (``remember`` states a
        memory verbatim)."""
        payload = {
            "kind": kind,
            "content": content,
            "scope": self.scope_payload(),
            "hints": hints or {},
            "custom_metadata": metadata,
        }
        key = idempotency_key or _default_key("obs", self.scope, kind, content)
        data = await self._request("POST", "/v1/observations", json=payload, idempotency_key=key)
        return ObservationAck.model_validate(data)

    async def verify(
        self,
        answer: str,
        *,
        bundle: ContextBundle | None = None,
        query: str | None = None,
        items: Sequence[ContextItem | dict[str, Any]] | None = None,
        unused: Sequence[dict[str, Any]] | None = None,
        document_ids: Sequence[str] | None = None,
        use_llm: bool | None = None,
    ) -> GroundingReport:
        """Verify ``answer`` claim by claim (citation validation, NLI, judge for borderline
        claims, contradiction scan) against a ``bundle`` from :meth:`context`, explicit
        evidence ``items`` or a fresh retrieval for ``query`` under this scope."""
        payload: dict[str, Any] = {"answer": answer, "scope": self.scope_payload()}
        if use_llm is not None:
            payload["use_llm"] = use_llm
        if bundle is not None:
            payload["items"] = bundle.evidence_items()
            payload["unused"] = [u.model_dump(mode="json") for u in bundle.evidence.unused]
        elif items is not None:
            payload["items"] = [_verify_item(i) for i in items]
            if unused:
                payload["unused"] = [dict(u) for u in unused]
        elif query is not None:
            payload["query"] = query
            if document_ids:
                payload["document_ids"] = list(document_ids)
        else:
            raise ValueError("verify() needs a bundle, items or a query")
        data = await self._request("POST", "/v1/verify", json=payload)
        return GroundingReport.model_validate(data)

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

    async def outcome(
        self, *, success: bool, note: str | None = None, run_id: str | None = None
    ) -> RunOutcome:
        """Whether this run (or ``run_id``) achieved its task. Only a successful run
        validates the procedures learned from what it did."""
        run = run_id or self.scope.agent_run_id
        if not run:
            raise ValueError("outcome() needs a run: bind agent_run_id or pass run_id")
        data = await self._request(
            "POST",
            f"/v1/runs/{run}/outcome",
            json={"scope": self.scope_payload(), "success": success, "note": note},
        )
        return RunOutcome.model_validate(data)

    async def tool_hints(
        self, task: str, *, available: Sequence[str] | None = None, k: int = TOOL_HINTS_K
    ) -> ToolHints:
        """Which tools fit ``task`` (among ``available`` when given), the learned plan, the
        next step, argument values found in memory, and what is missing."""
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

    async def call_agent_tool(self, name: str, args: Mapping[str, Any]) -> Any:
        """Run one memory tool in this scope and return its ``result``."""
        data = await self._request(
            "POST",
            f"/v1/agent-tools/{name}",
            json={"scope": self.scope_payload(), "args": dict(args)},
        )
        return data.get("result")

    # -- plumbing -------------------------------------------------------
    def scope_payload(self) -> dict[str, Any]:
        """The scope as a request body carries it. The trace id travels as traceparent (or
        as the correlation id), never in the body."""
        return self.scope.model_dump(mode="json", exclude_none=True, exclude={"trace_id"})

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        return await self._client.transport.request(method, path, scope=self.scope, **kwargs)

    async def _request_page(self, path: str, **params: Any) -> tuple[Any, str | None]:
        query = {k: v for k, v in params.items() if v is not None}
        return await self._client.transport.request_page(path, scope=self.scope, params=query)


class FeedbackAPI:
    """Judgements on what the platform did: stored apart from memory, learned from off the
    request path (a verdict on a memory reinforces, retracts or corrects it; on an answer it
    adjusts the cited memories; on a tool call it counts toward approval patterns)."""

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


def _record(feedback: Any) -> dict[str, Any]:
    if isinstance(feedback, Mapping):
        return dict(feedback)
    dump = getattr(feedback, "model_dump", None)
    if dump is None:
        raise TypeError("feedback() takes a Feedback record (a pydantic model or a mapping)")
    return dict(dump(mode="json"))


class ProfileAPI:
    """Pinned profile blocks for this scope: ``user``, ``agent`` and ``workspace`` (and
    ``<level>.<name>`` blocks beside them), always part of the pushed context."""

    def __init__(self, ctx: MemoryContext) -> None:
        self._ctx = ctx

    async def __call__(self) -> list[ProfileBlock]:
        data = await self._ctx._request("GET", "/v1/profile")
        return [ProfileBlock.model_validate(b) for b in data.get("blocks", [])]

    async def set(self, block: str, text: str) -> ProfileBlock:
        """Replace the block's text."""
        data = await self._ctx._request(
            "PUT",
            f"/v1/profile/{block}",
            json={"scope": self._ctx.scope_payload(), "text": text},
        )
        return ProfileBlock.model_validate(data)

    async def edit(self, block: str, old: str, new: str) -> ProfileBlock:
        """Replace ``old`` with ``new`` in the block; ``ConflictError`` when ``old`` is not
        in it (read the block again and retry)."""
        data = await self._ctx._request(
            "PATCH",
            f"/v1/profile/{block}",
            json={"scope": self._ctx.scope_payload(), "old": old, "new": new},
        )
        return ProfileBlock.model_validate(data)


class ChatAPI:
    """The thread's transcript: what the user and the assistant said."""

    def __init__(self, ctx: MemoryContext) -> None:
        self._ctx = ctx

    async def user(
        self,
        content: str,
        *,
        attachments: list[Any] | None = None,
        idempotency_key: str | None = None,
        **metadata: Any,
    ) -> MessageAck:
        ack = await self._message("USER", content, idempotency_key=idempotency_key, **metadata)
        for attachment in attachments or ():
            await self._ctx.advanced.documents.add(attachment, message_id=ack.message_id)
        return ack

    async def assistant(
        self, content: str, *, idempotency_key: str | None = None, **metadata: Any
    ) -> MessageAck:
        return await self._message(
            "ASSISTANT", content, idempotency_key=idempotency_key, **metadata
        )

    async def internal(
        self,
        content: str,
        *,
        role: MessageRole = "AGENT",
        idempotency_key: str | None = None,
        **metadata: Any,
    ) -> MessageAck:
        return await self._message(
            role, content, kind="INTERNAL", idempotency_key=idempotency_key, **metadata
        )

    async def thread(self) -> ThreadInfo:
        data = await self._ctx._request("GET", f"/v1/threads/{self._ctx.scope.thread_id}")
        return ThreadInfo.model_validate(data)

    async def create(self, *, title: str | None = None, **metadata: Any) -> ThreadInfo:
        """Create the context's thread explicitly (idempotent: an existing thread is
        returned). Messages create threads on demand, so this is for titles/metadata."""
        payload: dict[str, Any] = {"scope": self._ctx.scope_payload(), "custom_metadata": metadata}
        if self._ctx.scope.thread_id:
            payload["thread_id"] = self._ctx.scope.thread_id
        if title is not None:
            payload["title"] = title
        data = await self._ctx._request("POST", "/v1/threads", json=payload)
        return ThreadInfo.model_validate(data)

    async def message(self, message_id: str) -> MessageInfo:
        data = await self._ctx._request("GET", f"/v1/messages/{message_id}")
        return MessageInfo.model_validate(data)

    async def delete_thread(self, thread_id: str | None = None) -> None:
        """Soft-delete a thread (owner or tenant admin): messages stop being listed and
        retrieved; archived segments are kept for the retention period."""
        tid = thread_id or self._ctx.scope.thread_id
        await self._ctx._request("DELETE", f"/v1/threads/{tid}", idempotency_key=f"delthr-{tid}")

    async def _message(
        self,
        role: MessageRole,
        content: str,
        *,
        kind: MessageKind = "VISIBLE",
        idempotency_key: str | None = None,
        **metadata: Any,
    ) -> MessageAck:
        payload = {
            "role": role,
            "kind": kind,
            "content": content,
            "scope": self._ctx.scope_payload(),
            "custom_metadata": metadata,
        }
        scope = self._ctx.scope
        # A turn makes lineage + content a message's identity. Without one, two identical
        # messages in a thread ("ok") are two messages: a fresh key per call, which the
        # transport reuses across its own retries of that call.
        key = idempotency_key or (
            _default_key("msg", scope, role, kind, content)
            if scope.turn_id
            else f"msg-{uuid.uuid4().hex}"
        )
        data = await self._ctx._request("POST", "/v1/messages", json=payload, idempotency_key=key)
        return MessageAck.model_validate(data)


def _verify_item(item: ContextItem | dict[str, Any]) -> dict[str, Any]:
    if isinstance(item, ContextItem):
        return item.verification_item()
    return {
        k: v
        for k, v in dict(item).items()
        if k in ("item_id", "text", "kind", "citation", "attributes")
    }


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
