"""``ctx.advanced``: everything past the verbs an agent calls every turn.

Documents, the knowledge graph, briefs, the tool catalog, this agent's model key, the memory
inventory and job status are bound to the context's scope; ``tenant``, ``admin`` and
``webhooks`` are the client's (unbound) administration objects, reachable from here too.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import time
from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trellis.memory.models import (
    AgentKeyStatus,
    ApprovalSuggestion,
    Brief,
    BriefInfo,
    BriefSpec,
    CatalogTool,
    DocumentInfo,
    EntityProfile,
    FileHandle,
    GraphAnswer,
    GraphEntity,
    GraphLayer,
    JobHandle,
    MemoryResult,
    MemoryType,
    Page,
    Visibility,
)

if TYPE_CHECKING:
    from trellis.memory.admin import AdminAPI, TenantAPI, WebhooksAPI
    from trellis.memory.client import MemoryContext


class AdvancedAPI:
    def __init__(self, ctx: MemoryContext) -> None:
        self._ctx = ctx
        self.documents = DocumentsAPI(ctx)
        self.graph = GraphAPI(ctx)
        self.briefs = BriefsAPI(ctx)
        self.tools = ToolCatalogAPI(ctx)
        self.model_keys = ModelKeysAPI(ctx)
        self.memories = MemoriesAPI(ctx)

    @property
    def tenant(self) -> TenantAPI:
        """Administration of the key's tenant (``MemoryClient.tenant``)."""
        return self._ctx.client.tenant

    @property
    def admin(self) -> AdminAPI:
        """Platform administration (``MemoryClient.admin``)."""
        return self._ctx.client.admin

    @property
    def webhooks(self) -> WebhooksAPI:
        """The tenant's webhook subscriptions (``MemoryClient.tenant.webhooks``)."""
        return self._ctx.client.tenant.webhooks

    async def job(self, job_id: str) -> JobHandle:
        return JobHandle.model_validate(await self._ctx._request("GET", f"/v1/jobs/{job_id}"))


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
        for field, value in (("message_id", message_id), ("title", title)):
            if value:
                form[field] = value
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
    path = Path(str(file))
    return (
        filename or path.name,
        path.read_bytes(),
        media_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    )


class GraphAPI:
    """Knowledge-graph queries: resolve entities in a question (or given names) and traverse
    a bounded, visibility-filtered neighbourhood; ``as_of`` gives the valid-time view and
    ``valid_at`` the knowledge-time view. ``entities``/``entity`` search and profile them."""

    def __init__(self, ctx: MemoryContext) -> None:
        self._ctx = ctx

    async def query(
        self,
        query: str | None = None,
        *,
        entities: list[str] | None = None,
        hops: int = 1,
        as_of: datetime | None = None,
        valid_at: datetime | None = None,
        layers: Sequence[GraphLayer] | None = None,
        use_llm: bool | None = None,
    ) -> GraphAnswer:
        payload: dict[str, Any] = {
            "scope": self._ctx.scope_payload(),
            "query": query,
            "entities": entities or [],
            "hops": hops,
        }
        if use_llm is not None:
            payload["use_llm"] = use_llm
        for name, when in (("as_of", as_of), ("valid_at", valid_at)):
            if when is not None:
                payload[name] = when.isoformat()
        if layers:
            payload["layers"] = list(layers)
        data = await self._ctx._request("POST", "/v1/graph/query", json=payload)
        return GraphAnswer.model_validate(data)

    async def entities(
        self, query: str | None = None, *, entity_type: str | None = None, limit: int = 20
    ) -> list[GraphEntity]:
        """Entities visible in this scope whose name starts with ``query``, most mentioned
        first; ``entity_type`` narrows to one type (ORG, PERSON, ...)."""
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["q"] = query
        if entity_type:
            params["type"] = entity_type
        data = await self._ctx._request("GET", "/v1/graph/entities", params=params)
        return [GraphEntity.model_validate(e) for e in data.get("entities", [])]

    async def entity(self, entity_id: str) -> EntityProfile:
        """One entity: its current value per predicate, relations, history and evidence."""
        data = await self._ctx._request("GET", f"/v1/graph/entities/{entity_id}")
        return EntityProfile.model_validate(data)


class BriefsAPI:
    """Persistent standing questions and pages; reads never generate text."""

    def __init__(self, ctx: MemoryContext) -> None:
        self._ctx = ctx

    async def create(self, spec: BriefSpec, *, idempotency_key: str | None = None) -> Brief:
        data = await self._ctx._request(
            "POST",
            "/v1/briefs",
            json={"scope": self._ctx.scope_payload(), "spec": spec.model_dump(mode="json")},
            idempotency_key=idempotency_key,
        )
        return Brief.model_validate(data)

    async def update(
        self, brief_id: str, spec: BriefSpec, *, idempotency_key: str | None = None
    ) -> Brief:
        data = await self._ctx._request(
            "PUT",
            f"/v1/briefs/{brief_id}",
            json={"scope": self._ctx.scope_payload(), "spec": spec.model_dump(mode="json")},
            idempotency_key=idempotency_key,
        )
        return Brief.model_validate(data)

    async def get(self, brief_id: str) -> Brief:
        return Brief.model_validate(await self._ctx._request("GET", f"/v1/briefs/{brief_id}"))

    async def list(
        self, *, after: str = "", limit: int = 50, cursor: str | None = None
    ) -> list[BriefInfo]:
        return (await self.page(after=after, limit=limit, cursor=cursor)).items

    async def page(
        self, *, after: str = "", limit: int = 50, cursor: str | None = None
    ) -> Page[BriefInfo]:
        data, next_cursor = await self._ctx._request_page(
            "/v1/briefs", after=after or None, limit=limit, cursor=cursor
        )
        return Page[BriefInfo](
            items=[BriefInfo.model_validate(b) for b in data], next_cursor=next_cursor
        )

    async def delete(self, brief_id: str) -> None:
        await self._ctx._request("DELETE", f"/v1/briefs/{brief_id}")


class ToolCatalogAPI:
    """The tool catalog: what each tool is and does (its side effects decide how an agent
    may call it), with the statistics the service keeps; and the approval rules that past
    approvals support."""

    def __init__(self, ctx: MemoryContext) -> None:
        self._ctx = ctx

    async def catalog(self, names: Sequence[str] | None = None) -> list[CatalogTool]:
        """Catalog entries visible in this scope; ``names`` narrows to those tools (a name
        the catalog does not know is absent from the answer)."""
        params: dict[str, Any] = {"names": list(names)} if names else {}
        data = await self._ctx._request("GET", "/v1/tools", params=params)
        return [CatalogTool.model_validate(t) for t in data.get("tools", [])]

    async def put_catalog(
        self, tools: Sequence[dict[str, Any]], *, idempotency_key: str | None = None
    ) -> list[CatalogTool]:
        """Upsert entries by name (``name``, ``description``, ``input_schema``, ``required``,
        ``argument_entity_types``, ``side_effects`` = read | write | irreversible, ``source``,
        ``server``, ``examples``, ``redact``). An unchanged entry is left as it is."""
        data = await self._ctx._request(
            "PUT",
            "/v1/tools/catalog",
            json={"scope": self._ctx.scope_payload(), "tools": [dict(t) for t in tools]},
            idempotency_key=idempotency_key,
        )
        return [CatalogTool.model_validate(t) for t in data.get("tools", [])]

    async def approval_suggestions(self, *, tool: str | None = None) -> list[ApprovalSuggestion]:
        """Rules the approve/reject/edit decisions on this agent's tool calls support. They
        are suggestions: nothing applies them."""
        params = {"tool": tool} if tool else {}
        data = await self._ctx._request("GET", "/v1/tools/approval-suggestions", params=params)
        return [ApprovalSuggestion.model_validate(s) for s in data.get("suggestions", [])]

    async def plan(self, task: str, *, available_tools: Sequence[dict[str, Any]]) -> dict[str, Any]:
        data = await self._ctx._request(
            "POST",
            "/v1/tools/plan",
            json={
                "scope": self._ctx.scope_payload(),
                "task": task,
                "available_tools": list(available_tools),
            },
        )
        return dict(data)

    async def procedures(self, task: str) -> list[dict[str, Any]]:
        params = {"task": task}
        if self._ctx.scope.workspace_id:
            params["workspace_id"] = self._ctx.scope.workspace_id
        data = await self._ctx._request("GET", "/v1/tools/procedures", params=params)
        return list(data.get("procedures", []))


class ModelKeysAPI:
    """This agent's model key: the Bifrost virtual key its memory work is paid with. The
    service returns status, never the secret."""

    def __init__(self, ctx: MemoryContext) -> None:
        self._ctx = ctx

    async def set(self, virtual_key: str, *, idempotency_key: str | None = None) -> AgentKeyStatus:
        """Register or rotate the key (idempotent with the same ``idempotency_key``)."""
        data = await self._ctx._request(
            "PUT",
            "/v1/agents/model-key",
            json={"scope": self._ctx.scope_payload(), "virtual_key": virtual_key},
            idempotency_key=idempotency_key,
        )
        return AgentKeyStatus.model_validate(data)

    async def status(self) -> AgentKeyStatus:
        return AgentKeyStatus.model_validate(
            await self._ctx._request("GET", "/v1/agents/model-key")
        )

    async def revoke(self, *, idempotency_key: str | None = None) -> AgentKeyStatus:
        data = await self._ctx._request(
            "DELETE", "/v1/agents/model-key", idempotency_key=idempotency_key
        )
        return AgentKeyStatus.model_validate(data)


class MemoriesAPI:
    """The memory inventory: current memories anchored to this context's scopes (user,
    thread, agent run, work, workspace). ``ctx.search`` is the ranked, query-driven view."""

    def __init__(self, ctx: MemoryContext) -> None:
        self._ctx = ctx

    async def get(self, memory_id: str) -> MemoryResult:
        return MemoryResult.model_validate(
            await self._ctx._request("GET", f"/v1/memories/{memory_id}")
        )

    async def list(
        self,
        *,
        memory_types: Sequence[MemoryType] | None = None,
        include_superseded: bool = False,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[MemoryResult]:
        """One page; :meth:`page` also returns the cursor, :meth:`iter` walks every page."""
        page = await self.page(
            memory_types=memory_types,
            include_superseded=include_superseded,
            limit=limit,
            cursor=cursor,
        )
        return page.items

    async def page(
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
        data = await self._ctx._request("GET", "/v1/memories", params=params)
        return Page[MemoryResult](
            items=[MemoryResult.model_validate(m) for m in data.get("memories", [])],
            next_cursor=data.get("next_cursor"),
        )

    async def iter(
        self,
        *,
        memory_types: Sequence[MemoryType] | None = None,
        include_superseded: bool = False,
        page_size: int = 100,
    ) -> AsyncIterator[MemoryResult]:
        """Every memory the inventory lists, page by page."""
        cursor: str | None = None
        while True:
            page = await self.page(
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
