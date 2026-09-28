"""The 90% path::

    memory = MemoryClient("http://memory-service:8080", api_key="...")
    ctx = memory.bind(tenant_id=..., user_id=..., thread_id=..., session_id=..., turn_id=...)
    await ctx.chat.user(message, attachments=files)
    bundle = await ctx.context(message)
    ...
    await ctx.chat.assistant(answer)

``MemoryClient`` is static (one per process). ``MemoryContext`` is per request and
immutable; ``contextvars`` propagate it within one async execution for convenience only.
"""

from __future__ import annotations

import hashlib
import time
import uuid
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Self

import httpx

from trellis.memory.errors import InsufficientEvidence
from trellis.memory.models import (
    AgentKeyStatus,
    ApiKeyInfo,
    Brief,
    BriefInfo,
    BriefSpec,
    ContextBundle,
    ContextItem,
    CreatedTenant,
    DeliveryInfo,
    DocumentInfo,
    EvidenceRef,
    Feedback,
    FeedbackSource,
    FeedbackTargetKind,
    FeedbackVerdict,
    FileHandle,
    GraphAnswer,
    GroundingReport,
    GroupInfo,
    GroupMemberInfo,
    IssuedKey,
    JobHandle,
    KeyRole,
    Lifetime,
    MemberRole,
    MemoryResult,
    MemoryType,
    MessageAck,
    MessageInfo,
    MessageKind,
    MessageRole,
    ObservationAck,
    ObservationKind,
    Page,
    ReadAuditRecord,
    RecallKind,
    Scope,
    TenantInfo,
    ThreadInfo,
    ToolCall,
    ToolPlan,
    ToolResult,
    ToolStatus,
    Visibility,
    WebhookCreated,
    WebhookEvent,
    WebhookInfo,
    WorkspaceInfo,
    WorkspaceMemberInfo,
)
from trellis.memory.transport import HEADER_TENANT, Transport

#: the release that removes the aliases this SDK keeps for one release (ADR 0022)
ALIASES_REMOVED_IN = "0.3.0"

_current_context: ContextVar[MemoryContext | None] = ContextVar("trellis.memory_ctx", default=None)


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
    """Per-request handle. Immutable; ``derive`` creates child contexts for agent runs."""

    def __init__(self, client: MemoryClient, scope: Scope) -> None:
        self._client = client
        self.scope = scope
        self.chat = ChatAPI(self)
        self.documents = DocumentsAPI(self)
        self.graph = GraphAPI(self)
        self.briefs = BriefsAPI(self)
        self.tools = ToolsAPI(self)
        self.runs = RunsAPI(self)
        self.feedback = FeedbackAPI(self)
        self._token: Any = None

    @property
    def files(self) -> DocumentsAPI:
        """Deprecated spelling of :attr:`documents` (ADR 0022); removed in
        ``ALIASES_REMOVED_IN``."""
        warnings.warn(
            "MemoryContext.files is deprecated; use MemoryContext.documents "
            f"(removed in {ALIASES_REMOVED_IN})",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.documents

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

    # -- 90% path -------------------------------------------------------
    async def context(
        self,
        query: str,
        *,
        token_budget: int | None = None,
        require_evidence: bool = False,
        use_llm: bool = False,
        **options: Any,
    ) -> ContextBundle:
        """Bounded, ranked context for this turn. With ``require_evidence=True`` an
        ``INSUFFICIENT`` evidence report raises :class:`InsufficientEvidence` instead of
        returning a bundle the caller might answer from anyway."""
        payload: dict[str, Any] = {"query": query, "scope": self._scope_payload(), **options}
        payload["use_llm"] = use_llm
        if token_budget is not None:
            payload["token_budget"] = token_budget
        data = await self._request("POST", "/v1/context", json=payload)
        bundle = ContextBundle.model_validate(data)
        if require_evidence and bundle.evidence.status == "INSUFFICIENT":
            raise InsufficientEvidence(
                "no sufficient evidence was retrieved for this query",
                code="INSUFFICIENT_EVIDENCE",
                status=200,
                details={"notes": list(getattr(bundle.evidence, "notes", []) or [])},
            )
        return bundle

    async def observe(
        self,
        content: str,
        *,
        kind: ObservationKind = "EVENT",
        idempotency_key: str | None = None,
        hints: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> ObservationAck:
        payload = {
            "kind": kind,
            "content": content,
            "scope": self._scope_payload(),
            "hints": hints or {},
            "custom_metadata": metadata,
        }
        key = idempotency_key or _default_key("obs", self.scope, kind, content)
        data = await self._request("POST", "/v1/observations", json=payload, idempotency_key=key)
        return ObservationAck.model_validate(data)

    # -- advanced -------------------------------------------------------
    async def remember(
        self,
        content: str,
        *,
        memory_type: MemoryType = "SEMANTIC",
        lifetime: Lifetime = "LONG_TERM",
        visibility: Visibility | None = None,
        **metadata: Any,
    ) -> ObservationAck:
        hints: dict[str, Any] = {"memory_type": memory_type, "lifetime": lifetime}
        if visibility:
            hints["visibility"] = visibility
        return await self.observe(content, kind="EVENT", hints=hints, **metadata)

    async def recall(
        self,
        query: str,
        *,
        limit: int = 20,
        kinds: Sequence[RecallKind] | None = None,
        use_llm: bool = False,
        **options: Any,
    ) -> list[ContextItem]:
        """Ranked, scope-filtered evidence (chunks and memories) without bundle assembly.
        ``kinds`` narrows what is searched: chunk (document passages), memory, summary."""
        payload = {"query": query, "scope": self._scope_payload(), "limit": limit, **options}
        payload["use_llm"] = use_llm
        if kinds is not None:
            payload["kinds"] = list(kinds)
        data = await self._request("POST", "/v1/recall", json=payload)
        return [ContextItem.model_validate(m) for m in data.get("results", [])]

    async def verify(
        self,
        answer: str,
        *,
        bundle: ContextBundle | None = None,
        query: str | None = None,
        items: Sequence[ContextItem | dict[str, Any]] | None = None,
        unused: Sequence[dict[str, Any]] | None = None,
        document_ids: Sequence[str] | None = None,
        use_llm: bool = False,
    ) -> GroundingReport:
        """Verify ``answer`` claim by claim (citation validation, NLI, judge for borderline
        claims, contradiction scan) against a ``bundle`` from :meth:`context`, explicit
        evidence ``items`` or a fresh retrieval for ``query`` under this scope."""
        payload: dict[str, Any] = {"answer": answer, "scope": self._scope_payload()}
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

    async def get_memory(self, memory_id: str) -> MemoryResult:
        data = await self._request("GET", f"/v1/memories/{memory_id}")
        return MemoryResult.model_validate(data)

    async def memories(
        self,
        *,
        memory_types: Sequence[MemoryType] | None = None,
        include_superseded: bool = False,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[MemoryResult]:
        """Current memories anchored to this context's scopes (user, thread, agent run,
        work, workspace) — the inventory view; ``recall`` is the ranked, query-driven view.
        One page; :meth:`memories_page` also returns the cursor, :meth:`iter_memories`
        walks every page."""
        page = await self.memories_page(
            memory_types=memory_types,
            include_superseded=include_superseded,
            limit=limit,
            cursor=cursor,
        )
        return page.items

    async def memories_page(
        self,
        *,
        memory_types: Sequence[MemoryType] | None = None,
        include_superseded: bool = False,
        limit: int = 100,
        cursor: str | None = None,
    ) -> Page[MemoryResult]:
        params: dict[str, Any] = {"limit": limit, "include_superseded": include_superseded}
        if memory_types:
            params["memory_type"] = list(memory_types)
        if cursor:
            params["cursor"] = cursor
        data = await self._request("GET", "/v1/memories", params=params)
        return Page[MemoryResult](
            items=[MemoryResult.model_validate(m) for m in data.get("memories", [])],
            next_cursor=data.get("next_cursor"),
        )

    async def iter_memories(
        self,
        *,
        memory_types: Sequence[MemoryType] | None = None,
        include_superseded: bool = False,
        page_size: int = 100,
    ) -> AsyncIterator[MemoryResult]:
        """Every memory the inventory view lists, page by page."""
        cursor: str | None = None
        while True:
            page = await self.memories_page(
                memory_types=memory_types,
                include_superseded=include_superseded,
                limit=page_size,
                cursor=cursor,
            )
            for item in page.items:
                yield item
            if page.next_cursor is None:
                return
            cursor = page.next_cursor

    async def forget(self, memory_id: str) -> None:
        await self._request(
            "DELETE", f"/v1/memories/{memory_id}", idempotency_key=f"del-{memory_id}"
        )

    async def job(self, job_id: str) -> JobHandle:
        data = await self._request("GET", f"/v1/jobs/{job_id}")
        return JobHandle.model_validate(data)

    async def set_model_key(
        self, virtual_key: str, *, idempotency_key: str | None = None
    ) -> AgentKeyStatus:
        """Register/rotate this agent's key. The server returns status, never its secret."""
        data = await self._request(
            "PUT",
            "/v1/agents/model-key",
            json={"scope": self._scope_payload(), "virtual_key": virtual_key},
            idempotency_key=idempotency_key,
        )
        return AgentKeyStatus.model_validate(data)

    async def model_key_status(self) -> AgentKeyStatus:
        data = await self._request("GET", "/v1/agents/model-key")
        return AgentKeyStatus.model_validate(data)

    async def revoke_model_key(self, *, idempotency_key: str | None = None) -> AgentKeyStatus:
        data = await self._request(
            "DELETE", "/v1/agents/model-key", idempotency_key=idempotency_key
        )
        return AgentKeyStatus.model_validate(data)

    # -- plumbing -------------------------------------------------------
    def _scope_payload(self) -> dict[str, Any]:
        # trace_id travels as traceparent (or as the correlation id): the body field is
        # deprecated on the service and is not sent
        return self.scope.model_dump(mode="json", exclude_none=True, exclude={"trace_id"})

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        return await self._client.transport.request(method, path, scope=self.scope, **kwargs)

    async def _request_page(self, path: str, **params: Any) -> tuple[Any, str | None]:
        query = {k: v for k, v in params.items() if v is not None}
        return await self._client.transport.request_page(path, scope=self.scope, params=query)


class FeedbackAPI:
    """Judgements on what the platform did: stored apart from memory, learned from off the
    request path (a verdict on a memory reinforces, retracts or corrects it)."""

    def __init__(self, ctx: MemoryContext) -> None:
        self._ctx = ctx

    async def submit(
        self,
        target_kind: FeedbackTargetKind,
        target_id: str,
        verdict: FeedbackVerdict,
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
        """Record one judgement. A retry with the same ``feedback_id`` returns the stored
        record; the identity fields (tenant, workspace, user, agent, run) come from this
        context and nothing else of the scope travels in the body."""
        scope = self._ctx.scope
        payload: dict[str, Any] = {
            "feedback_id": feedback_id,
            "tenant_id": scope.tenant_id,
            "workspace_id": scope.workspace_id,
            "user_id": scope.user_id,
            "agent_id": scope.agent_id,
            "agent_run_id": scope.agent_run_id,
            "target_kind": target_kind,
            "target_id": target_id,
            "verdict": verdict,
            "correction": correction,
            "score": score,
            "comment": comment,
            "reviewer": reviewer,
            "source": source,
            "evidence_refs": [e.model_dump(mode="json", exclude_none=True) for e in evidence_refs],
            "metadata": metadata or {},
        }
        body = {k: v for k, v in payload.items() if v is not None}
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
        target_kind: FeedbackTargetKind,
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
        target_kind: FeedbackTargetKind,
        target_id: str,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> Page[Feedback]:
        params: dict[str, Any] = {
            "target_kind": target_kind,
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


class BriefsAPI:
    """Persistent standing questions and pages; reads never generate text."""

    def __init__(self, ctx: MemoryContext) -> None:
        self.ctx = ctx

    async def create(self, spec: BriefSpec, *, idempotency_key: str | None = None) -> Brief:
        data = await self.ctx._request(
            "POST",
            "/v1/briefs",
            json={"scope": self.ctx._scope_payload(), "spec": spec.model_dump(mode="json")},
            idempotency_key=idempotency_key,
        )
        return Brief.model_validate(data)

    async def update(
        self, brief_id: str, spec: BriefSpec, *, idempotency_key: str | None = None
    ) -> Brief:
        data = await self.ctx._request(
            "PUT",
            f"/v1/briefs/{brief_id}",
            json={"scope": self.ctx._scope_payload(), "spec": spec.model_dump(mode="json")},
            idempotency_key=idempotency_key,
        )
        return Brief.model_validate(data)

    async def get(self, brief_id: str) -> Brief:
        return Brief.model_validate(await self.ctx._request("GET", f"/v1/briefs/{brief_id}"))

    async def list(
        self, *, after: str = "", limit: int = 50, cursor: str | None = None
    ) -> list[BriefInfo]:
        return (await self.page(after=after, limit=limit, cursor=cursor)).items

    async def page(
        self, *, after: str = "", limit: int = 50, cursor: str | None = None
    ) -> Page[BriefInfo]:
        data, next_cursor = await self.ctx._request_page(
            "/v1/briefs", after=after or None, limit=limit, cursor=cursor
        )
        return Page[BriefInfo](
            items=[BriefInfo.model_validate(b) for b in data], next_cursor=next_cursor
        )

    async def delete(self, brief_id: str) -> None:
        await self.ctx._request("DELETE", f"/v1/briefs/{brief_id}")


class ChatAPI:
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
        if attachments:
            for att in attachments:
                await self._ctx.documents.add(att, message_id=ack.message_id)
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

    async def history(
        self, *, limit: int = 50, include_internal: bool = False
    ) -> list[MessageInfo]:
        thread_id = self._ctx.scope.thread_id
        if not thread_id:
            return []
        data = await self._ctx._request(
            "GET",
            f"/v1/threads/{thread_id}/messages",
            params={"limit": limit, "include_internal": include_internal},
        )
        return [MessageInfo.model_validate(m) for m in data.get("messages", [])]

    async def thread(self) -> ThreadInfo:
        data = await self._ctx._request("GET", f"/v1/threads/{self._ctx.scope.thread_id}")
        return ThreadInfo.model_validate(data)

    async def create(self, *, title: str | None = None, **metadata: Any) -> ThreadInfo:
        """Create the context's thread explicitly (idempotent: an existing thread is
        returned). Messages create threads on demand, so this is for titles/metadata."""
        payload: dict[str, Any] = {"scope": self._ctx._scope_payload(), "custom_metadata": metadata}
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
            "scope": self._ctx._scope_payload(),
            "custom_metadata": metadata,
        }
        key = idempotency_key or _default_key("msg", self._ctx.scope, role, kind, content)
        data = await self._ctx._request("POST", "/v1/messages", json=payload, idempotency_key=key)
        return MessageAck.model_validate(data)


class DocumentsAPI:
    """Documents ingested into RAG memory: upload, status, readiness."""

    def __init__(self, ctx: MemoryContext) -> None:
        self._ctx = ctx

    async def add(
        self,
        file: Any,
        *,
        message_id: str | None = None,
        filename: str | None = None,
        media_type: str | None = None,
        title: str | None = None,
        visibility: Visibility | None = None,
        idempotency_key: str | None = None,
        **metadata: Any,
    ) -> FileHandle:
        """``file`` may be bytes, a path, or a (filename, bytes, media_type) tuple.
        ``visibility`` widens who may retrieve the document (default: the thread, else the
        user); ``metadata`` is stored as custom metadata."""
        import json

        name, data, mtype = _coerce_file(file, filename, media_type)
        digest = hashlib.sha256(data).hexdigest()
        # The filename and media type are part of the request the service compares against a
        # replayed key, so they belong in the key: identical bytes uploaded under a different
        # name are a different document, not a replay of the same one.
        fields = (
            f"{name}|{mtype}|{message_id or ''}|{title or ''}|{visibility or ''}"
            f"|{sorted(metadata.items())}"
        )
        form_digest = hashlib.blake2b(fields.encode(), digest_size=6).hexdigest()
        tenant = self._ctx.scope.tenant_id or ""
        key = idempotency_key or f"file-{tenant}-{digest}-{form_digest}"
        form = {"scope": self._ctx.scope.model_dump_json(exclude_none=True, exclude={"trace_id"})}
        if message_id:
            form["message_id"] = message_id
        if title:
            form["title"] = title
        if visibility:
            form["visibility"] = visibility
        if metadata:
            form["custom_metadata"] = json.dumps(metadata)
        result = await self._ctx._request(
            "POST",
            "/v1/documents",
            files={"file": (name, data, mtype)},
            data=form,
            idempotency_key=key,
        )
        return FileHandle.model_validate(result)

    async def document(self, document_id: str) -> DocumentInfo:
        data = await self._ctx._request("GET", f"/v1/documents/{document_id}")
        return DocumentInfo.model_validate(data)

    async def wait_ready(
        self, document_id: str, *, max_wait: float = 60.0, interval: float = 0.5
    ) -> DocumentInfo:
        """Poll until the document is parsed and indexed (READY) or FAILED, or ``max_wait``
        seconds have passed (the last observed status is returned either way)."""
        import asyncio
        import time

        deadline = time.monotonic() + max_wait
        while True:
            doc = await self.document(document_id)
            if doc.status in ("READY", "FAILED") or time.monotonic() >= deadline:
                return doc
            await asyncio.sleep(interval)


def _coerce_file(file: Any, filename: str | None, media_type: str | None) -> tuple[str, bytes, str]:
    if isinstance(file, tuple) and len(file) == 3:
        return file[0], file[1], file[2]
    if isinstance(file, bytes):
        return filename or "upload.bin", file, media_type or "application/octet-stream"
    from pathlib import Path

    path = Path(str(file))
    import mimetypes

    return (
        filename or path.name,
        path.read_bytes(),
        media_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    )


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


class GraphAPI:
    """Knowledge-graph queries: resolve entities in a question (or given names) and traverse
    a bounded, visibility-filtered neighbourhood; ``as_of`` gives the temporal view."""

    def __init__(self, ctx: MemoryContext) -> None:
        self._ctx = ctx

    async def query(
        self,
        query: str | None = None,
        *,
        entities: list[str] | None = None,
        hops: int = 1,
        as_of: datetime | None = None,
        use_llm: bool = False,
    ) -> GraphAnswer:
        payload: dict[str, Any] = {
            "scope": self._ctx._scope_payload(),
            "query": query,
            "entities": entities or [],
            "hops": hops,
            "use_llm": use_llm,
        }
        if as_of is not None:
            payload["as_of"] = as_of.isoformat()
        data = await self._ctx._request("POST", "/v1/graph/query", json=payload)
        return GraphAnswer.model_validate(data)


class ToolsAPI:
    """Tool memory: register what a tool is, record what happened, and ask what to call.

    The service never runs a tool. ``execute`` is the convenience loop an adapter wants —
    look in the cache, run the caller's own executor on a miss, record the outcome — so the
    invocation records, chains and suggestions work the same whether the tool lives in the
    agent framework or behind an MCP gateway.
    """

    def __init__(self, ctx: MemoryContext) -> None:
        self._ctx = ctx

    async def record(
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
        task: str = "",
        step: int | None = None,
        sub_calls: list[dict[str, Any]] | None = None,
        visibility: Visibility = "PRIVATE",
    ) -> ToolResult:
        data = await self._ctx._request(
            "POST",
            "/v1/tools/invocations",
            json={
                "scope": self._ctx._scope_payload(),
                "tool": tool,
                "args": args,
                "output": output,
                "output_summary": output_summary,
                "status": status,
                "error_class": error_class,
                "latency_ms": latency_ms,
                "cost": cost,
                "task": task,
                "step": step,
                "sub_calls": sub_calls or [],
                "visibility": visibility,
            },
        )
        return ToolResult.model_validate(data)

    async def plan(self, task: str, *, available_tools: Sequence[dict[str, Any]]) -> ToolPlan:
        data = await self._ctx._request(
            "POST",
            "/v1/tools/plan",
            json={
                "scope": self._ctx._scope_payload(),
                "task": task,
                "available_tools": list(available_tools),
            },
        )
        return ToolPlan.model_validate(data)

    async def procedures(self, task: str) -> list[dict[str, Any]]:
        """Procedures mined for a task pattern, in the bound scope.

        The agent travels as a query parameter: invocations are recorded against the
        principal ``agent:<id>``, and a GET has no body to carry lineage in.
        """
        params = {"task": task}
        if self._ctx.scope.agent_id:
            params["agent_id"] = self._ctx.scope.agent_id
        if self._ctx.scope.workspace_id:
            params["workspace_id"] = self._ctx.scope.workspace_id
        data = await self._ctx._request("GET", "/v1/tools/procedures", params=params)
        return list(data.get("procedures", []))

    async def execute(
        self,
        call: ToolCall,
        executor: Callable[[str, dict[str, Any]], Awaitable[Any]],
        *,
        visibility: Visibility = "PRIVATE",
    ) -> ToolResult:
        """Run the caller's executor, then record the invocation idempotently.

        ``executor`` is whatever actually runs the tool — a local function, a framework tool
        node, or a POST to Bifrost's ``/v1/mcp/tool/execute``. The service stays out of it.

        There is deliberately no output cache in front of this. Replaying a previous result
        for identical arguments is the staleness bug in another costume: ``stock_level(SKU-1)``
        returning yesterday's 95 units is exactly the failure the rest of this system is built
        to avoid. A tool that is genuinely deterministic should be cached by its own caller,
        which is the only place that knows.
        """
        started = time.perf_counter()
        status: ToolStatus = "ok"
        error_class: str | None = None
        output: Any = None
        try:
            output = await executor(call.tool, call.args)
        except Exception as exc:
            status, error_class = "error", type(exc).__name__
            await self.record(
                call.tool,
                call.args,
                status=status,
                error_class=error_class,
                latency_ms=(time.perf_counter() - started) * 1000,
                task=call.task,
                step=call.step,
                visibility=visibility,
            )
            raise
        result = await self.record(
            call.tool,
            call.args,
            output=output,
            status=status,
            latency_ms=(time.perf_counter() - started) * 1000,
            task=call.task,
            step=call.step,
            visibility=visibility,
        )
        return result.model_copy(update={"output": output})


class RunsAPI:
    """Run outcomes. Only a run labelled successful validates a procedure, so this is how an
    application tells the service that what an agent did actually worked."""

    def __init__(self, ctx: MemoryContext) -> None:
        self._ctx = ctx

    async def outcome(
        self, run_id: str, *, success: bool, note: str | None = None
    ) -> dict[str, Any]:
        return await self._ctx._request(
            "POST",
            f"/v1/runs/{run_id}/outcome",
            json={"scope": self._ctx._scope_payload(), "success": success, "note": note},
        )


# --- platform administration -----------------------------------------------------


class AdminAPI:
    """Onboarding, for the bootstrap key: ``POST /v1/admin/tenants`` and friends."""

    def __init__(self, client: MemoryClient) -> None:
        self._t = client.transport

    async def create_tenant(
        self,
        name: str,
        *,
        tenant_id: str | None = None,
        retention_days: int | None = None,
        rate_limit_per_minute: int | None = None,
        idempotency_key: str | None = None,
    ) -> CreatedTenant:
        """The tenant and its first admin key. The key's token is returned once: keep it.

        Pass ``idempotency_key`` to make a retry safe: the replay carries the same tenant with
        ``admin_key.token`` set to None (the secret is never shown twice).
        """
        payload = {
            "name": name,
            "tenant_id": tenant_id,
            "retention_days": retention_days,
            "rate_limit_per_minute": rate_limit_per_minute,
        }
        return CreatedTenant.model_validate(
            await self._t.request(
                "POST", "/v1/admin/tenants", json=payload, idempotency_key=idempotency_key
            )
        )

    async def tenants(
        self, *, after: str = "", limit: int = 100, cursor: str | None = None
    ) -> list[TenantInfo]:
        return (await self.tenants_page(after=after, limit=limit, cursor=cursor)).items

    async def tenants_page(
        self, *, after: str = "", limit: int = 100, cursor: str | None = None
    ) -> Page[TenantInfo]:
        params = {"after": after or None, "limit": limit, "cursor": cursor}
        data, next_cursor = await self._t.request_page(
            "/v1/admin/tenants", params={k: v for k, v in params.items() if v is not None}
        )
        return Page[TenantInfo](
            items=[TenantInfo.model_validate(t) for t in data], next_cursor=next_cursor
        )

    async def get_tenant(self, tenant_id: str) -> TenantInfo:
        return TenantInfo.model_validate(
            await self._t.request("GET", f"/v1/admin/tenants/{tenant_id}")
        )

    async def update_tenant(self, tenant_id: str, **changes: Any) -> TenantInfo:
        """``name``, ``status``, ``retention_days`` / ``clear_retention``,
        ``rate_limit_per_minute`` / ``clear_rate_limit``."""
        return TenantInfo.model_validate(
            await self._t.request("PATCH", f"/v1/admin/tenants/{tenant_id}", json=changes)
        )


class TenantAPI:
    """Administration of one tenant: its keys, workspaces (teams), groups and read audit."""

    def __init__(self, client: MemoryClient, *, tenant_id: str | None = None) -> None:
        self._t = client.transport
        self._headers = {HEADER_TENANT: tenant_id} if tenant_id else {}
        self.keys = KeysAPI(self)
        self.workspaces = WorkspacesAPI(self)
        self.groups = GroupsAPI(self)
        self.webhooks = WebhooksAPI(self)

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        return await self._t.request(method, path, headers=self._headers, **kwargs)

    async def _page(self, path: str, **params: Any) -> tuple[Any, str | None]:
        query = {k: v for k, v in params.items() if v is not None}
        return await self._t.request_page(path, params=query, headers=self._headers)

    async def model_key_status(self) -> AgentKeyStatus:
        """The tenant's model key: the level every agent and workspace without a key of its
        own resolves to before the operator key."""
        return AgentKeyStatus.model_validate(await self._request("GET", "/v1/model-key"))

    async def set_model_key(
        self, virtual_key: str, *, idempotency_key: str | None = None
    ) -> AgentKeyStatus:
        data = await self._request(
            "PUT",
            "/v1/model-key",
            json={"virtual_key": virtual_key},
            idempotency_key=idempotency_key,
        )
        return AgentKeyStatus.model_validate(data)

    async def revoke_model_key(self, *, idempotency_key: str | None = None) -> AgentKeyStatus:
        data = await self._request("DELETE", "/v1/model-key", idempotency_key=idempotency_key)
        return AgentKeyStatus.model_validate(data)

    async def reads(
        self, *, after: Any = None, before: Any = None, limit: int = 100, cursor: str | None = None
    ) -> list[ReadAuditRecord]:
        """Who read which records, newest first. Page older entries with the cursor (or
        ``before=<the last entry's at>``); ``after`` is a since-filter."""
        return (await self.reads_page(after=after, before=before, limit=limit, cursor=cursor)).items

    async def reads_page(
        self, *, after: Any = None, before: Any = None, limit: int = 100, cursor: str | None = None
    ) -> Page[ReadAuditRecord]:
        params: dict[str, Any] = {"limit": limit, "cursor": cursor}
        for name, value in (("after", after), ("before", before)):
            if value is not None:
                params[name] = value.isoformat() if hasattr(value, "isoformat") else value
        data, next_cursor = await self._page("/v1/reads", **params)
        return Page[ReadAuditRecord](
            items=[ReadAuditRecord.model_validate(r) for r in data], next_cursor=next_cursor
        )


class KeysAPI:
    def __init__(self, tenant: TenantAPI) -> None:
        self._tenant = tenant

    async def issue(
        self,
        role: KeyRole,
        name: str,
        *,
        workspace_id: str | None = None,
        expires_in_days: int | None = None,
        idempotency_key: str | None = None,
    ) -> IssuedKey:
        """A new key; its ``token`` is shown once. With ``idempotency_key`` a retry returns
        the same key and ``token=None``; without it every call issues another key."""
        payload = {
            "role": role,
            "name": name,
            "workspace_id": workspace_id,
            "expires_in_days": expires_in_days,
        }
        return IssuedKey.model_validate(
            await self._tenant._request(
                "POST", "/v1/keys", json=payload, idempotency_key=idempotency_key
            )
        )

    async def list(self, *, limit: int = 100, cursor: str | None = None) -> list[ApiKeyInfo]:
        return (await self.page(limit=limit, cursor=cursor)).items

    async def page(self, *, limit: int = 100, cursor: str | None = None) -> Page[ApiKeyInfo]:
        data, next_cursor = await self._tenant._page("/v1/keys", limit=limit, cursor=cursor)
        return Page[ApiKeyInfo](
            items=[ApiKeyInfo.model_validate(k) for k in data], next_cursor=next_cursor
        )

    async def revoke(self, key_id: str) -> None:
        await self._tenant._request("DELETE", f"/v1/keys/{key_id}")


class WorkspacesAPI:
    def __init__(self, tenant: TenantAPI) -> None:
        self._tenant = tenant

    async def create(
        self, name: str, *, workspace_id: str | None = None, idempotency_key: str | None = None
    ) -> WorkspaceInfo:
        payload = {"name": name, "workspace_id": workspace_id}
        return WorkspaceInfo.model_validate(
            await self._tenant._request(
                "POST", "/v1/workspaces", json=payload, idempotency_key=idempotency_key
            )
        )

    async def list(self, *, limit: int = 100, cursor: str | None = None) -> list[WorkspaceInfo]:
        return (await self.page(limit=limit, cursor=cursor)).items

    async def page(self, *, limit: int = 100, cursor: str | None = None) -> Page[WorkspaceInfo]:
        data, next_cursor = await self._tenant._page("/v1/workspaces", limit=limit, cursor=cursor)
        return Page[WorkspaceInfo](
            items=[WorkspaceInfo.model_validate(w) for w in data], next_cursor=next_cursor
        )

    async def model_key_status(self, workspace_id: str) -> AgentKeyStatus:
        """The team's model key: used by every agent of the workspace without one of its own."""
        data = await self._tenant._request("GET", f"/v1/workspaces/{workspace_id}/model-key")
        return AgentKeyStatus.model_validate(data)

    async def set_model_key(
        self, workspace_id: str, virtual_key: str, *, idempotency_key: str | None = None
    ) -> AgentKeyStatus:
        data = await self._tenant._request(
            "PUT",
            f"/v1/workspaces/{workspace_id}/model-key",
            json={"virtual_key": virtual_key},
            idempotency_key=idempotency_key,
        )
        return AgentKeyStatus.model_validate(data)

    async def revoke_model_key(
        self, workspace_id: str, *, idempotency_key: str | None = None
    ) -> AgentKeyStatus:
        data = await self._tenant._request(
            "DELETE", f"/v1/workspaces/{workspace_id}/model-key", idempotency_key=idempotency_key
        )
        return AgentKeyStatus.model_validate(data)

    async def get(self, workspace_id: str) -> WorkspaceInfo:
        return WorkspaceInfo.model_validate(
            await self._tenant._request("GET", f"/v1/workspaces/{workspace_id}")
        )

    async def delete(self, workspace_id: str) -> None:
        await self._tenant._request("DELETE", f"/v1/workspaces/{workspace_id}")

    async def set_member(
        self, workspace_id: str, principal: str, *, role: MemberRole = "member"
    ) -> WorkspaceMemberInfo:
        """``principal`` is ``user:<id>``, ``agent:<id>`` or ``group:<id>``."""
        return WorkspaceMemberInfo.model_validate(
            await self._tenant._request(
                "PUT", f"/v1/workspaces/{workspace_id}/members/{principal}", json={"role": role}
            )
        )

    async def remove_member(self, workspace_id: str, principal: str) -> None:
        await self._tenant._request("DELETE", f"/v1/workspaces/{workspace_id}/members/{principal}")

    async def members(self, workspace_id: str) -> list[WorkspaceMemberInfo]:
        data = await self._tenant._request("GET", f"/v1/workspaces/{workspace_id}/members")
        return [WorkspaceMemberInfo.model_validate(m) for m in data]


class GroupsAPI:
    def __init__(self, tenant: TenantAPI) -> None:
        self._tenant = tenant

    async def create(
        self, name: str, *, group_id: str | None = None, idempotency_key: str | None = None
    ) -> GroupInfo:
        payload = {"name": name, "group_id": group_id}
        return GroupInfo.model_validate(
            await self._tenant._request(
                "POST", "/v1/groups", json=payload, idempotency_key=idempotency_key
            )
        )

    async def list(self, *, limit: int = 100, cursor: str | None = None) -> list[GroupInfo]:
        return (await self.page(limit=limit, cursor=cursor)).items

    async def page(self, *, limit: int = 100, cursor: str | None = None) -> Page[GroupInfo]:
        data, next_cursor = await self._tenant._page("/v1/groups", limit=limit, cursor=cursor)
        return Page[GroupInfo](
            items=[GroupInfo.model_validate(g) for g in data], next_cursor=next_cursor
        )

    async def delete(self, group_id: str) -> None:
        await self._tenant._request("DELETE", f"/v1/groups/{group_id}")

    async def add_user(self, group_id: str, user_id: str) -> GroupMemberInfo:
        return GroupMemberInfo.model_validate(
            await self._tenant._request("PUT", f"/v1/groups/{group_id}/members/{user_id}")
        )

    async def remove_user(self, group_id: str, user_id: str) -> None:
        await self._tenant._request("DELETE", f"/v1/groups/{group_id}/members/{user_id}")

    async def members(self, group_id: str) -> list[GroupMemberInfo]:
        data = await self._tenant._request("GET", f"/v1/groups/{group_id}/members")
        return [GroupMemberInfo.model_validate(m) for m in data]


#: Deprecated name of :class:`DocumentsAPI` (ADR 0022); removed in ``ALIASES_REMOVED_IN``.
FilesAPI = DocumentsAPI


class WebhooksAPI:
    """Outbound webhooks: the tenant's subscriptions and their deliveries. Verify what you
    receive with :func:`trellis.memory.webhooks.verify_signature`."""

    def __init__(self, tenant: TenantAPI) -> None:
        self._tenant = tenant

    async def create(
        self,
        url: str,
        events: Sequence[WebhookEvent],
        *,
        workspace_id: str | None = None,
        description: str | None = None,
        subscription_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> WebhookCreated:
        """Subscribe ``url`` to ``events``; the returned ``secret`` is shown once."""
        payload = {
            "url": url,
            "events": list(events),
            "workspace_id": workspace_id,
            "description": description,
            "subscription_id": subscription_id,
        }
        body = {k: v for k, v in payload.items() if v is not None}
        data = await self._tenant._request(
            "POST", "/v1/webhooks", json=body, idempotency_key=idempotency_key
        )
        return WebhookCreated.model_validate(data)

    async def list(self, *, limit: int = 100, cursor: str | None = None) -> list[WebhookInfo]:
        return (await self.page(limit=limit, cursor=cursor)).items

    async def page(self, *, limit: int = 100, cursor: str | None = None) -> Page[WebhookInfo]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        data = await self._tenant._request("GET", "/v1/webhooks", params=params)
        return Page[WebhookInfo](
            items=[WebhookInfo.model_validate(w) for w in data.get("webhooks", [])],
            next_cursor=data.get("next_cursor"),
        )

    async def get(self, subscription_id: str) -> WebhookInfo:
        return WebhookInfo.model_validate(
            await self._tenant._request("GET", f"/v1/webhooks/{subscription_id}")
        )

    async def update(
        self,
        subscription_id: str,
        *,
        url: str | None = None,
        events: Sequence[WebhookEvent] | None = None,
        enabled: bool | None = None,
        description: str | None = None,
    ) -> WebhookInfo:
        changes: dict[str, Any] = {}
        if url is not None:
            changes["url"] = url
        if events is not None:
            changes["events"] = list(events)
        if enabled is not None:
            changes["enabled"] = enabled
        if description is not None:
            changes["description"] = description
        data = await self._tenant._request("PATCH", f"/v1/webhooks/{subscription_id}", json=changes)
        return WebhookInfo.model_validate(data)

    async def delete(self, subscription_id: str) -> None:
        await self._tenant._request("DELETE", f"/v1/webhooks/{subscription_id}")

    async def deliveries(
        self, subscription_id: str, *, limit: int = 100, cursor: str | None = None
    ) -> list[DeliveryInfo]:
        return (await self.deliveries_page(subscription_id, limit=limit, cursor=cursor)).items

    async def deliveries_page(
        self, subscription_id: str, *, limit: int = 100, cursor: str | None = None
    ) -> Page[DeliveryInfo]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        data = await self._tenant._request(
            "GET", f"/v1/webhooks/{subscription_id}/deliveries", params=params
        )
        return Page[DeliveryInfo](
            items=[DeliveryInfo.model_validate(d) for d in data.get("deliveries", [])],
            next_cursor=data.get("next_cursor"),
        )

    async def test(self, subscription_id: str) -> DeliveryInfo:
        """Queue a ``webhook.test`` delivery to the subscription's URL."""
        return DeliveryInfo.model_validate(
            await self._tenant._request("POST", f"/v1/webhooks/{subscription_id}/test")
        )
