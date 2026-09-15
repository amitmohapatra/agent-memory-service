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
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Self

import httpx

from universal_memory.errors import InsufficientEvidence
from universal_memory.models import (
    ContextBundle,
    ContextItem,
    DocumentInfo,
    FileHandle,
    GraphAnswer,
    GroundingReport,
    JobHandle,
    MemoryResult,
    MessageAck,
    MessageInfo,
    NextSteps,
    ObservationAck,
    Scope,
    ThreadInfo,
    Tool,
    ToolCall,
    ToolPlan,
    ToolResult,
    ToolSuggestion,
)
from universal_memory.transport import Transport

_current_context: ContextVar[MemoryContext | None] = ContextVar(
    "universal_memory_ctx", default=None
)


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

    def bind(self, **scope: Any) -> MemoryContext:
        """Create a per-request context. Accepts every :class:`Scope` field."""
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
        self.files = FilesAPI(self)
        self.graph = GraphAPI(self)
        self.tools = ToolsAPI(self)
        self.runs = RunsAPI(self)
        self._token: Any = None

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
        working notes (``Visibility.RUN``) are readable by this run, the runs it derives
        with ``.agent(...)`` (hand-off context flows down) and the same agent later — never
        by the user, sibling runs, or other agents. Share explicitly with ``memory_type=
        "SHARED"`` / ``visibility="AGENT_GROUP"``."""
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
        **options: Any,
    ) -> ContextBundle:
        """Bounded, ranked context for this turn. With ``require_evidence=True`` an
        ``INSUFFICIENT`` evidence report raises :class:`InsufficientEvidence` instead of
        returning a bundle the caller might answer from anyway."""
        payload: dict[str, Any] = {"query": query, "scope": self._scope_payload(), **options}
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
        kind: str = "EVENT",
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
        memory_type: str = "SEMANTIC",
        lifetime: str = "LONG_TERM",
        visibility: str | None = None,
        **metadata: Any,
    ) -> ObservationAck:
        hints: dict[str, Any] = {"memory_type": memory_type, "lifetime": lifetime}
        if visibility:
            hints["visibility"] = visibility
        return await self.observe(content, kind="EVENT", hints=hints, **metadata)

    async def recall(self, query: str, *, limit: int = 20, **options: Any) -> list[ContextItem]:
        """Ranked, scope-filtered evidence (chunks and memories) without bundle assembly."""
        payload = {"query": query, "scope": self._scope_payload(), "limit": limit, **options}
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
    ) -> GroundingReport:
        """Verify ``answer`` claim by claim (citation validation, NLI, judge for borderline
        claims, contradiction scan) against a ``bundle`` from :meth:`context`, explicit
        evidence ``items`` or a fresh retrieval for ``query`` under this scope."""
        payload: dict[str, Any] = {"answer": answer, "scope": self._scope_payload()}
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
        memory_types: Sequence[str] | None = None,
        include_superseded: bool = False,
        limit: int = 100,
    ) -> list[MemoryResult]:
        """Current memories anchored to this context's scopes (user, thread, agent run,
        work, workspace) — the inventory view; ``recall`` is the ranked, query-driven view."""
        params: dict[str, Any] = {"limit": limit, "include_superseded": include_superseded}
        if memory_types:
            params["memory_type"] = list(memory_types)
        data = await self._request("GET", "/v1/memories", params=params)
        return [MemoryResult.model_validate(m) for m in data.get("memories", [])]

    async def forget(self, memory_id: str) -> None:
        await self._request(
            "DELETE", f"/v1/memories/{memory_id}", idempotency_key=f"del-{memory_id}"
        )

    async def job(self, job_id: str) -> JobHandle:
        data = await self._request("GET", f"/v1/jobs/{job_id}")
        return JobHandle.model_validate(data)

    # -- plumbing -------------------------------------------------------
    def _scope_payload(self) -> dict[str, Any]:
        return self.scope.model_dump(mode="json", exclude_none=True)

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        return await self._client.transport.request(method, path, scope=self.scope, **kwargs)


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
                await self._ctx.files.add(att, message_id=ack.message_id)
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
        role: str = "AGENT",
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
        role: str,
        content: str,
        *,
        kind: str = "VISIBLE",
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


class FilesAPI:
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
        visibility: str | None = None,
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
        key = idempotency_key or f"file-{self._ctx.scope.tenant_id}-{digest}-{form_digest}"
        form = {"scope": self._ctx.scope.model_dump_json(exclude_none=True)}
        if message_id:
            form["message_id"] = message_id
        if title:
            form["title"] = title
        if visibility:
            form["visibility"] = visibility
        if metadata:
            form["custom_metadata"] = json.dumps(metadata)
        result = await self._ctx._request(
            "POST", "/v1/files", files={"file": (name, data, mtype)}, data=form, idempotency_key=key
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
        return {
            "item_id": item.item_id,
            "text": item.text,
            "kind": item.representation,
            "citation": item.citation,
        }
    return {k: v for k, v in dict(item).items() if k in ("item_id", "text", "kind", "citation")}


def _default_key(prefix: str, scope: Scope, *parts: str) -> str:
    """Deterministic idempotency key from lineage + content so retries never duplicate."""
    h = hashlib.blake2b(digest_size=16)
    for p in (
        scope.tenant_id,
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
    ) -> GraphAnswer:
        payload: dict[str, Any] = {
            "scope": self._ctx._scope_payload(),
            "query": query,
            "entities": entities or [],
            "hops": hops,
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

    async def register(
        self,
        name: str,
        *,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        source: str = "manual",
        server: str | None = None,
        policy: dict[str, Any] | None = None,
    ) -> Tool:
        payload: dict[str, Any] = {
            "scope": self._ctx._scope_payload(),
            "name": name,
            "description": description,
            "input_schema": input_schema,
            "output_schema": output_schema,
            "tags": tags or [],
            "source": source,
            "server": server,
        }
        if policy is not None:
            payload["policy"] = policy
        data = await self._ctx._request("POST", "/v1/tools", json=payload)
        return Tool.model_validate(data)

    async def register_many(self, descriptors: Sequence[dict[str, Any]]) -> list[Tool]:
        return [await self.register(**d) for d in descriptors]

    async def list(self, *, limit: int = 200) -> list[Tool]:
        data = await self._ctx._request("GET", "/v1/tools", params={"limit": limit})
        return [Tool.model_validate(t) for t in data.get("tools", [])]

    async def lookup(self, tool: str, args: dict[str, Any]) -> ToolResult:
        data = await self._ctx._request(
            "POST",
            "/v1/tools/lookup",
            json={"scope": self._ctx._scope_payload(), "tool": tool, "args": args},
        )
        return ToolResult.model_validate(data)

    async def record(
        self,
        tool: str,
        args: dict[str, Any],
        *,
        output: Any = None,
        output_summary: str | None = None,
        status: str = "ok",
        error_class: str | None = None,
        latency_ms: float | None = None,
        cost: float | None = None,
        task: str = "",
        step: int | None = None,
        sub_calls: list[dict[str, Any]] | None = None,
        visibility: str = "RUN",
    ) -> ToolResult:
        data = await self._ctx._request(
            "POST",
            "/v1/tools/record",
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

    async def suggest(
        self,
        task: str,
        *,
        available_tools: Sequence[dict[str, Any]],
        context: str | None = None,
        limit: int = 5,
    ) -> list[ToolSuggestion]:
        data = await self._ctx._request(
            "POST",
            "/v1/tools/suggest",
            json={
                "scope": self._ctx._scope_payload(),
                "task": task,
                "available_tools": list(available_tools),
                "context": context,
                "limit": limit,
            },
        )
        return [ToolSuggestion.model_validate(s) for s in data.get("suggestions", [])]

    async def next(
        self,
        task: str,
        *,
        trajectory_so_far: Sequence[dict[str, Any]],
        available_tools: Sequence[dict[str, Any]],
        limit: int = 3,
    ) -> NextSteps:
        data = await self._ctx._request(
            "POST",
            "/v1/tools/next",
            json={
                "scope": self._ctx._scope_payload(),
                "task": task,
                "trajectory_so_far": list(trajectory_so_far),
                "available_tools": list(available_tools),
                "limit": limit,
            },
        )
        return NextSteps.model_validate(data)

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
        data = await self._ctx._request("GET", "/v1/tools/procedures", params={"task": task})
        return list(data.get("procedures", []))

    async def execute(
        self,
        call: ToolCall,
        executor: Callable[[str, dict[str, Any]], Awaitable[Any]],
        *,
        visibility: str = "RUN",
    ) -> ToolResult:
        """Cache lookup, then the caller's executor, then an idempotent record.

        ``executor`` is whatever actually runs the tool — a local function, a framework tool
        node, or a POST to Bifrost's ``/v1/mcp/tool/execute``. The service stays out of it.
        """
        hit = await self.lookup(call.tool, call.args)
        if hit.cached:
            return hit
        started = time.perf_counter()
        status, error_class, output = "ok", None, None
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
