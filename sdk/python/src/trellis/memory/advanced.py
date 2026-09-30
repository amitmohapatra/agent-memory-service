"""``ctx.advanced``: everything past the verbs an agent calls every turn.

Documents, the knowledge graph, the tool catalog, this agent's model key, the memory
inventory and job status are bound to the context's scope; ``tenant`` and ``admin`` are the client's
(unbound) administration objects, reachable from here too.
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
    CatalogTool,
    DocumentInfo,
    EntityProfile,
    FileHandle,
    GraphEntity,
    GraphLayer,
    JobHandle,
    MemoryResult,
    MemoryType,
    Page,
    Visibility,
)

if TYPE_CHECKING:
    from trellis.memory.admin import AdminAPI, TenantAPI
    from trellis.memory.client import MemoryContext


class AdvancedAPI:
    def __init__(self, ctx: MemoryContext) -> None:
        self._ctx = ctx
        self.documents = DocumentsAPI(ctx)
        self.graph = GraphAPI(ctx)
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
    """The knowledge graph: the entities a text names (or whose name starts with it), and an
    entity's profile with, at a depth, the bounded, visibility-filtered graph around it;
    ``as_of`` gives the valid-time view and ``valid_at`` the knowledge-time view."""

    def __init__(self, ctx: MemoryContext) -> None:
        self._ctx = ctx

    async def entities(
        self, query: str | None = None, *, entity_type: str | None = None, limit: int = 20
    ) -> list[GraphEntity]:
        """Entities visible in this scope: those ``query`` names first, then those whose name
        starts with it, most mentioned first; ``entity_type`` narrows to one type."""
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["q"] = query
        if entity_type:
            params["type"] = entity_type
        data = await self._ctx._request("GET", "/v1/graph/entities", params=params)
        return [GraphEntity.model_validate(e) for e in data.get("entities", [])]

    async def entity(
        self,
        entity_id: str,
        *,
        depth: int = 0,
        as_of: datetime | None = None,
        valid_at: datetime | None = None,
        layers: Sequence[GraphLayer] | None = None,
    ) -> EntityProfile:
        """One entity: its current value per predicate, relations, history and evidence;
        with ``depth`` (1-3) also the graph around it (``neighborhood``)."""
        params: dict[str, Any] = {"depth": depth}
        for name, when in (("as_of", as_of), ("valid_at", valid_at)):
            if when is not None:
                params[name] = when.isoformat()
        if layers:
            params["layers"] = list(layers)
        data = await self._ctx._request("GET", f"/v1/graph/entities/{entity_id}", params=params)
        return EntityProfile.model_validate(data)


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
        ``argument_entity_types``, ``side_effects`` = read | write | irreversible,
        ``annotations`` (the MCP hints), ``approve_when`` (``trellis.memory.approval``; ""
        removes it), ``source``, ``server``, ``examples``, ``redact``). An existing entry
        changes only in the fields sent: a publisher that sends no ``side_effects`` or
        ``approve_when`` keeps what an administrator set."""
        data = await self._ctx._request(
            "PUT",
            "/v1/tools/catalog",
            json={"scope": self._ctx.scope_payload(), "tools": [dict(t) for t in tools]},
            idempotency_key=idempotency_key,
        )
        return [CatalogTool.model_validate(t) for t in data.get("tools", [])]

    async def approval_suggestions(self, *, tool: str | None = None) -> list[ApprovalSuggestion]:
        """Rules the approve/reject/edit decisions on this agent's tool calls support. They
        are suggestions until accepted."""
        params = {"tool": tool} if tool else {}
        data = await self._ctx._request("GET", "/v1/tools/approval-suggestions", params=params)
        return [ApprovalSuggestion.model_validate(s) for s in data.get("suggestions", [])]

    async def accept_suggestion(self, suggestion_id: str) -> CatalogTool:
        """Write a suggestion's rule into its tool's ``approve_when``."""
        data = await self._ctx._request(
            "POST", f"/v1/tools/approval-suggestions/{suggestion_id}/accept"
        )
        return CatalogTool.model_validate(data)


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
