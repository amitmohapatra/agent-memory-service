"""``universal-memory-mcp``: one MCP server over the SDK.

    MEMORY_URL=http://localhost:8080 MEMORY_API_KEY=... MEMORY_TENANT_ID=acme \\
    MEMORY_USER_ID=u1 uvx universal-memory-mcp

Tools: ``memory.recall``, ``memory.context``, ``memory.observe``, ``memory.remember``,
``memory.forget``, ``memory.graph_query``, ``memory.files.add``, ``memory.threads.create``
/ ``get`` / ``history`` / ``record`` / ``delete``. Every call may carry a ``scope`` that is
validated against the server's authorized scope (tenant fixed; user and workspace must be
the configured ones or allow-listed; agents act for that user with run lineage). Every
result carries ``evidence`` (status + sources) so a client never answers from nothing
without knowing it. ``memory.verify`` is not exposed: the SDK has no ``verify`` yet.
"""

from __future__ import annotations

import base64
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from universal_memory import InsufficientEvidence, MemoryClient, MemoryContext, MemoryError
from universal_memory.integrations.core import (
    Entry,
    MemoryHooks,
    bundle_entries,
    evidence_metadata,
    items_entries,
    render_bundle,
    scope_fields,
)
from universal_memory_mcp.config import ServerConfig

NOT_APPLICABLE = "NOT_APPLICABLE"


class ScopeArgs(BaseModel):
    """Per-call scope. Omitted fields fall back to the server configuration; ``user_id``
    and ``workspace_id`` must be authorized for this server."""

    user_id: str | None = None
    workspace_id: str | None = None
    thread_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    work_id: str | None = None
    agent_id: str | None = Field(default=None, description="Act as this agent (a run for the user)")
    agent_run_id: str | None = Field(default=None, description="Stable run id; new run if omitted")
    parent_agent_run_id: str | None = None
    agent_group_id: str | None = None


def resolve_scope(config: ServerConfig, args: ScopeArgs | None) -> dict[str, Any]:
    """Scope fields for a call, or ``ToolError`` when the call asks for more than the server
    is authorized for. The tenant can never be chosen per call; an agent without an
    explicit ``agent_run_id`` gets a fresh run (pass one to keep RUN-scoped memory)."""
    a = args or ScopeArgs()
    if a.user_id and not config.user_allowed(a.user_id):
        raise ToolError(f"SCOPE_DENIED: user_id {a.user_id!r} is not authorized for this server")
    if a.workspace_id and not config.workspace_allowed(a.workspace_id):
        raise ToolError(
            f"SCOPE_DENIED: workspace_id {a.workspace_id!r} is not authorized for this server"
        )
    overrides = {k: v for k, v in a.model_dump().items() if v is not None}
    agent = overrides.get("agent_id") or config.agent_id
    if agent:
        overrides["agent_id"] = agent
        if not overrides.get("agent_run_id"):
            configured = config.agent_run_id if agent == config.agent_id else None
            overrides["agent_run_id"] = configured or f"mcp-{uuid.uuid4().hex}"
    return scope_fields(
        defaults=config.defaults,
        overrides=overrides,
        thread_id=overrides.get("thread_id") or config.thread_id,
    )


def _entry_payload(e: Entry) -> dict[str, Any]:
    return {"text": e.text, **e.metadata}


def _sources(entries: Sequence[Entry]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for e in entries:
        key = e.citation or e.source_id or e.item_id or e.text[:40]
        seen.setdefault(
            key,
            {
                "citation": e.citation,
                "source_type": e.source_type,
                "source_id": e.source_id,
                "document_id": e.document_id,
                "page": e.page,
                "kind": e.kind,
            },
        )
    return list(seen.values())


def _write_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "evidence": {"status": NOT_APPLICABLE, "sources": []}}


def build_server(config: ServerConfig, client: MemoryClient | None = None) -> MCPServer:
    """The MCP server. ``client`` lets tests drive the real service in-process (ASGI)."""
    memory = client or MemoryClient(
        config.url, api_key=config.api_key, bearer_token=config.bearer_token, timeout=config.timeout
    )
    hooks = MemoryHooks(
        memory, namespace="mcp", defaults=config.defaults, token_budget=config.token_budget
    )
    server = MCPServer(
        config.name,
        instructions=(
            "Durable, scope-aware memory. Call memory.context before answering to get an "
            "evidence-gated bundle; only answer from evidence whose status is COMPLETE or "
            "INCOMPLETE and say what is missing. Record what happened with memory.observe; "
            "store durable facts and preferences with memory.remember."
        ),
    )

    def ctx_for(scope: ScopeArgs | None) -> MemoryContext:
        return memory.bind(**resolve_scope(config, scope))

    async def guard[T](coro: Any) -> T:
        try:
            return await coro
        except InsufficientEvidence as exc:
            raise ToolError(f"INSUFFICIENT_EVIDENCE: {exc.message}") from exc
        except MemoryError as exc:
            raise ToolError(f"{exc.code}: {exc.message}") from exc

    @server.tool(name="memory.recall", description="Ranked, scope-filtered evidence for a query")
    async def recall(
        query: str,
        limit: int = 10,
        kinds: list[str] | None = None,
        scope: ScopeArgs | None = None,
    ) -> dict[str, Any]:
        ctx = ctx_for(scope)
        options: dict[str, Any] = {"kinds": kinds} if kinds else {}
        items = await guard(ctx.recall(query, limit=limit, **options))
        entries = items_entries(items, evidence_status="INCOMPLETE" if items else "INSUFFICIENT")
        return {
            "items": [_entry_payload(e) for e in entries],
            "evidence": {
                "status": "INCOMPLETE" if items else "INSUFFICIENT",
                "sources": _sources(entries),
                "notes": ["recall is not evidence-gated; memory.context verifies a bundle"],
            },
        }

    @server.tool(
        name="memory.context",
        description="Evidence-gated context bundle for a turn (conversation window, "
        "memories, knowledge, graph facts) with a rendered prompt block",
    )
    async def context(
        query: str,
        token_budget: int | None = None,
        require_evidence: bool = False,
        scope: ScopeArgs | None = None,
    ) -> dict[str, Any]:
        ctx = ctx_for(scope)
        bundle = await guard(
            hooks.before_run(
                ctx, query, require_evidence=require_evidence, token_budget=token_budget
            )
        )
        if bundle is None:
            raise ToolError("VALIDATION: query must not be blank")
        entries = bundle_entries(bundle)
        return {
            "rendered": render_bundle(bundle),
            "entries": [_entry_payload(e) for e in entries],
            "conversation": {
                "thread_id": bundle.conversation.thread_id,
                "message_ids": list(bundle.conversation.message_ids),
                "summary": bundle.conversation.summary,
            },
            "evidence": {**evidence_metadata(bundle), "sources": _sources(entries)},
        }

    @server.tool(
        name="memory.observe",
        description="Tell memory something happened (EVENT, DECISION, TOOL_RESULT, "
        "AGENT_RESULT, FEEDBACK...). Idempotent per (scope, kind, content) or key.",
    )
    async def observe(
        content: str,
        kind: str = "EVENT",
        hints: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        metadata: dict[str, Any] | None = None,
        scope: ScopeArgs | None = None,
    ) -> dict[str, Any]:
        ctx = ctx_for(scope)
        ack = await guard(
            ctx.observe(
                content,
                kind=kind,
                idempotency_key=idempotency_key,
                hints=hints,
                **(metadata or {}),
            )
        )
        return _write_result(ack.model_dump())

    @server.tool(
        name="memory.remember",
        description="Store a durable fact or preference (memory_type SEMANTIC, PREFERENCE, "
        "DECISION, SHARED...; visibility USER, RUN, AGENT_GROUP, THREAD...)",
    )
    async def remember(
        content: str,
        memory_type: str = "SEMANTIC",
        lifetime: str = "LONG_TERM",
        visibility: str | None = None,
        metadata: dict[str, Any] | None = None,
        scope: ScopeArgs | None = None,
    ) -> dict[str, Any]:
        ctx = ctx_for(scope)
        ack = await guard(
            ctx.remember(
                content,
                memory_type=memory_type,
                lifetime=lifetime,
                visibility=visibility,
                **(metadata or {}),
            )
        )
        return _write_result(ack.model_dump())

    @server.tool(
        name="memory.forget", description="Forget a memory (soft: it stops being recalled)"
    )
    async def forget(memory_id: str, scope: ScopeArgs | None = None) -> dict[str, Any]:
        ctx = ctx_for(scope)
        await guard(ctx.forget(memory_id))
        return _write_result({"memory_id": memory_id, "forgotten": True})

    @server.tool(
        name="memory.graph_query",
        description="Knowledge-graph facts around entities named in the query (or given), "
        "optionally as of a point in time",
    )
    async def graph_query(
        query: str | None = None,
        entities: list[str] | None = None,
        hops: int = 1,
        as_of: str | None = None,
        scope: ScopeArgs | None = None,
    ) -> dict[str, Any]:
        ctx = ctx_for(scope)
        when = datetime.fromisoformat(as_of) if as_of else None
        answer = await guard(ctx.graph.query(query, entities=entities, hops=hops, as_of=when))
        facts = [f.model_dump(mode="json") for f in answer.facts]
        sources = [
            {
                "citation": f.fact_text or f"{f.subject} {f.predicate} {f.object}",
                "source_type": "RELATION",
                "source_id": f.relation_id,
                "document_id": f.document_id,
                "page": (f.evidence[0].page if f.evidence else None),
                "kind": "graph_fact",
            }
            for f in answer.facts
        ]
        return {
            "matched": [e.model_dump(mode="json") for e in answer.matched],
            "entities": [e.model_dump(mode="json") for e in answer.entities],
            "facts": facts,
            "visited": answer.visited,
            "evidence": {"status": "COMPLETE" if facts else "INSUFFICIENT", "sources": sources},
        }

    @server.tool(
        name="memory.files.add",
        description="Add a document (text, or base64 bytes) to memory; it is parsed, indexed "
        "and its facts extracted. Set wait=true to block until it is READY.",
    )
    async def files_add(
        filename: str,
        content: str | None = None,
        content_base64: str | None = None,
        media_type: str | None = None,
        title: str | None = None,
        visibility: str | None = None,
        wait: bool = False,
        scope: ScopeArgs | None = None,
    ) -> dict[str, Any]:
        if (content is None) == (content_base64 is None):
            raise ToolError("VALIDATION: pass exactly one of content or content_base64")
        data = content.encode() if content is not None else base64.b64decode(content_base64 or "")
        mtype = media_type or ("text/plain" if content is not None else "application/octet-stream")
        ctx = ctx_for(scope)
        handle = await guard(
            ctx.files.add((filename, data, mtype), title=title, visibility=visibility)
        )
        payload: dict[str, Any] = handle.model_dump()
        if wait:
            doc = await guard(ctx.files.wait_ready(handle.document_id))
            payload["status"] = doc.status
        return _write_result(payload)

    @server.tool(name="memory.threads.create", description="Create (or get) a memory thread")
    async def threads_create(
        thread_id: str | None = None, title: str | None = None, scope: ScopeArgs | None = None
    ) -> dict[str, Any]:
        args = (scope or ScopeArgs()).model_copy(update={"thread_id": thread_id})
        info = await guard(ctx_for(args).chat.create(title=title))
        return _write_result(info.model_dump(mode="json"))

    @server.tool(name="memory.threads.get", description="A thread's metadata")
    async def threads_get(thread_id: str, scope: ScopeArgs | None = None) -> dict[str, Any]:
        args = (scope or ScopeArgs()).model_copy(update={"thread_id": thread_id})
        info = await guard(ctx_for(args).chat.thread())
        return _write_result(info.model_dump(mode="json"))

    @server.tool(
        name="memory.threads.history",
        description="Visible messages of a thread (include_internal adds agent/tool notes)",
    )
    async def threads_history(
        thread_id: str,
        limit: int = 50,
        include_internal: bool = False,
        scope: ScopeArgs | None = None,
    ) -> dict[str, Any]:
        args = (scope or ScopeArgs()).model_copy(update={"thread_id": thread_id})
        history = await guard(
            ctx_for(args).chat.history(limit=limit, include_internal=include_internal)
        )
        messages = [m.model_dump(mode="json") for m in history]
        return {
            "thread_id": thread_id,
            "messages": messages,
            "evidence": {
                "status": "COMPLETE" if messages else "INSUFFICIENT",
                "sources": [
                    {
                        "citation": f"message:{m['message_id']}",
                        "source_type": "MESSAGE",
                        "source_id": m["message_id"],
                        "kind": "conversation",
                    }
                    for m in messages
                ],
            },
        }

    @server.tool(
        name="memory.threads.record",
        description="Record a chat message in a thread: role user, assistant (visible) or "
        "agent, tool, system (internal). Idempotent per (scope, role, content) or key.",
    )
    async def threads_record(
        thread_id: str,
        role: str,
        content: str,
        idempotency_key: str | None = None,
        metadata: dict[str, Any] | None = None,
        scope: ScopeArgs | None = None,
    ) -> dict[str, Any]:
        args = (scope or ScopeArgs()).model_copy(update={"thread_id": thread_id})
        ctx = ctx_for(args)
        r = role.lower()
        meta = metadata or {}
        if r == "user":
            ack = await guard(ctx.chat.user(content, idempotency_key=idempotency_key, **meta))
        elif r == "assistant":
            ack = await guard(ctx.chat.assistant(content, idempotency_key=idempotency_key, **meta))
        elif r in ("agent", "tool", "system"):
            ack = await guard(
                ctx.chat.internal(content, role=r.upper(), idempotency_key=idempotency_key, **meta)
            )
        else:
            raise ToolError(f"VALIDATION: unknown role {role!r}")
        return _write_result(ack.model_dump())

    @server.tool(name="memory.threads.delete", description="Soft-delete a thread")
    async def threads_delete(thread_id: str, scope: ScopeArgs | None = None) -> dict[str, Any]:
        args = (scope or ScopeArgs()).model_copy(update={"thread_id": thread_id})
        await guard(ctx_for(args).chat.delete_thread(thread_id))
        return _write_result({"thread_id": thread_id, "deleted": True})

    return server


def main() -> None:
    """Console entry point: stdio transport, configuration from the environment."""
    build_server(ServerConfig.from_env()).run("stdio")


if __name__ == "__main__":
    main()
