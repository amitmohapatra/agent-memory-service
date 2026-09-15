"""Framework-neutral integration core: what every adapter (LangGraph, Google ADK, CrewAI,
MCP, ...) needs to map a framework's session/agent model onto Memory Service scopes and
to read/write memory around a unit of work.

- **scope mapping** — :func:`scope_fields` turns adapter-level identity (tenant, user,
  workspace, agent group), a framework thread/session/turn and an agent lineage (a chain of
  :class:`AgentRun`) into ``Scope`` keyword arguments: app/project/crew -> workspace, user id
  -> user, session/thread -> thread (session and turn derived unless supplied), agent /
  sub-agent -> agent run with parent lineage, team/crew -> agent group;
- **hooks** — :class:`MemoryHooks`: ``before_run`` fetches an evidence-gated
  ``ContextBundle``; ``after_run`` records messages and observations with deterministic
  idempotency keys so a retried step never records twice; ``share`` publishes to the agent
  group explicitly;
- **rendering** — a bundle (or recall results) as flat :class:`Entry` items carrying source,
  page and evidence status, or as prompt-ready text;
- **evidence** — :data:`EvidenceStatus` and :func:`evidence_metadata` so every adapter's
  results carry the same evidence report.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import logging
import re
from collections.abc import Awaitable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from universal_memory.client import MemoryClient, MemoryContext
from universal_memory.models import ContextBundle, ContextItem, MessageAck, ObservationAck

log = logging.getLogger("universal_memory.integrations")

SCOPE_FIELDS = (
    "tenant_id",
    "workspace_id",
    "user_id",
    "group_ids",
    "thread_id",
    "session_id",
    "turn_id",
    "work_id",
    "task_id",
    "agent_id",
    "agent_group_id",
    "agent_run_id",
    "parent_agent_run_id",
    "trace_id",
    "correlation_id",
)

_SAFE = re.compile(r"[^A-Za-z0-9._:\-]")


def safe_id(value: Any, *, max_len: int = 200) -> str:
    """Coerce an arbitrary framework id into the service's id alphabet."""
    cleaned = _SAFE.sub("-", str(value)).strip("-.")[:max_len]
    return cleaned or "x"


# -- scope mapping ------------------------------------------------------------------


@dataclass(frozen=True)
class AgentRun:
    """One agent invocation in a lineage chain: ``agent`` names the agent (its agent id),
    ``run_id`` is the framework's stable id for this invocation (task id, invocation id,
    kickoff id...). Stable ids make retries map to the same run."""

    agent: str
    run_id: str


def scope_fields(
    *,
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
    thread_id: str | None = None,
    thread_prefix: str = "",
    turn_hint: str | None = None,
    step: int | None = None,
    runs: Sequence[AgentRun] = (),
    run_prefix: str = "",
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Build ``Scope`` keyword arguments for one position in a framework's execution.

    ``defaults`` is the adapter-level identity (tenant, user, workspace, agent group...);
    ``overrides`` (explicit per-call scope fields) win over it. ``thread_id`` becomes the
    memory thread; one session per thread and one turn per ``turn_hint`` (else per
    ``step``) are derived unless supplied. ``runs`` is the agent lineage from the
    outermost enclosing agent to the executing one: the last run becomes the scope's
    agent run, the one before it its parent.
    """
    out: dict[str, Any] = {k: v for k, v in defaults.items() if k in SCOPE_FIELDS}
    over = dict(overrides or {})
    out.update({k: v for k, v in over.items() if k in SCOPE_FIELDS})
    if thread_id and not out.get("thread_id"):
        out["thread_id"] = safe_id(f"{thread_prefix}{thread_id}")
    thread = out.get("thread_id")
    if thread and not out.get("session_id"):
        out["session_id"] = safe_id(f"{thread}-session")
    if thread and not out.get("turn_id"):
        out["turn_id"] = safe_id(f"turn-{turn_hint}" if turn_hint else f"{thread}-step{step or 0}")
    if runs and not over.get("agent_run_id"):
        last = runs[-1]
        out["agent_id"] = over.get("agent_id") or safe_id(last.agent)
        out["agent_run_id"] = safe_id(f"{run_prefix}{last.run_id}")
        out["parent_agent_run_id"] = (
            safe_id(f"{run_prefix}{runs[-2].run_id}") if len(runs) > 1 else None
        )
    if correlation_id and not out.get("correlation_id"):
        out["correlation_id"] = safe_id(correlation_id)
    return {k: v for k, v in out.items() if v is not None}


# -- messages and observations ----------------------------------------------------------

Role = Literal["user", "assistant", "tool", "system", "agent"]

_ROLES: dict[str, Role] = {
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "model": "assistant",
    "tool": "tool",
    "function": "tool",
    "system": "system",
    "agent": "agent",
}
_INTERNAL_ROLES = {"tool": "TOOL", "system": "SYSTEM", "agent": "AGENT"}


def normalize_role(value: Any) -> Role | None:
    """Map a framework role/type name (``human``, ``ai``, ``model``...) to a :data:`Role`."""
    return _ROLES.get(str(value or "").lower()) if value is not None else None


def message_text(content: Any) -> str:
    """Text of a message whose content may be a string or a list of parts (text blocks are
    kept, other modalities dropped)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list | tuple):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and block.get("type", "text") == "text":
                parts.append(str(block.get("text") or ""))
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


@dataclass(frozen=True)
class Message:
    """A framework-neutral chat message: ``user`` and ``assistant`` are the visible history,
    ``tool``/``system``/``agent`` are recorded as internal messages."""

    role: Role
    content: str
    id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """Something the application says happened; ``hints`` steer classification
    (``memory_type``, ``visibility``, ``lifetime``...)."""

    content: str
    kind: str | None = None
    hints: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def as_observations(raw: Any) -> list[Observation]:
    """Coerce text, a mapping (``content``, ``kind``, ``hints``, ``metadata``), a list of
    either, or an :class:`Observation` into observations; blanks are dropped."""
    if raw is None:
        return []
    if isinstance(raw, Observation):
        return [raw] if raw.content.strip() else []
    if isinstance(raw, str):
        return [Observation(raw)] if raw.strip() else []
    if isinstance(raw, Mapping):
        content = str(raw.get("content") or "").strip()
        if not content:
            return []
        return [
            Observation(
                content,
                kind=raw.get("kind"),
                hints=dict(raw.get("hints") or {}),
                metadata=dict(raw.get("metadata") or {}),
            )
        ]
    if isinstance(raw, list | tuple):
        out: list[Observation] = []
        for item in raw:
            out.extend(as_observations(item))
        return out
    return as_observations(str(raw))


def idempotency_key(namespace: str, prefix: str, *parts: Any) -> str:
    """Deterministic key from the adapter namespace and the step's identity + content, so a
    retried step (same framework ids, same content) replays the same key."""
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        h.update(("" if p is None else str(p)).encode())
        h.update(b"\x00")
    return f"{namespace}-{prefix}-{h.hexdigest()}"


# -- evidence and rendering -------------------------------------------------------------

EvidenceStatus = Literal["COMPLETE", "INCOMPLETE", "INSUFFICIENT"]

EntryKind = Literal["memory", "knowledge", "graph_fact", "summary", "conversation"]


def evidence_metadata(bundle: ContextBundle) -> dict[str, Any]:
    """The evidence report (and bundle facts) every adapter attaches to its results."""
    return {
        "evidence_status": bundle.evidence.status,
        "required_groups": list(bundle.evidence.required_groups),
        "satisfied_groups": list(bundle.evidence.satisfied_groups),
        "missing_groups": list(bundle.evidence.missing_groups),
        "escalations": list(bundle.evidence.escalations),
        "notes": list(bundle.evidence.notes),
        "query_type": bundle.query_type,
        "token_estimate": bundle.token_estimate,
        "token_budget": bundle.token_budget,
        "cache_hit": bundle.cache_hit,
    }


@dataclass(frozen=True)
class Entry:
    """One retrieved item flattened for a framework's memory record type."""

    text: str
    kind: EntryKind
    citation: str = ""
    score: float = 0.0
    item_id: str | None = None
    source_type: str | None = None
    source_id: str | None = None
    document_id: str | None = None
    message_id: str | None = None
    page: int | None = None
    section_path: str | None = None
    evidence_status: EvidenceStatus | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def metadata(self) -> dict[str, Any]:
        """Flat metadata (``source``, ``page``, ``evidence_status``...) for frameworks that
        carry one metadata dict per record."""
        return {
            "kind": self.kind,
            "citation": self.citation,
            "score": self.score,
            "item_id": self.item_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source": self.citation or self.source_id or self.item_id,
            "document_id": self.document_id,
            "message_id": self.message_id,
            "page": self.page,
            "section_path": self.section_path,
            "evidence_status": self.evidence_status,
            **{k: v for k, v in self.attributes.items() if k not in ("kind", "score")},
        }


def item_entry(
    item: ContextItem, kind: EntryKind, *, evidence_status: EvidenceStatus | None = None
) -> Entry:
    ev = item.evidence[0] if item.evidence else None
    return Entry(
        text=item.text,
        kind=kind,
        citation=item.citation,
        score=item.score,
        item_id=item.item_id,
        source_type=ev.source_type if ev else item.representation,
        source_id=ev.source_id if ev else item.item_id,
        document_id=item.document_id or (ev.document_id if ev else None),
        message_id=ev.message_id if ev else None,
        page=item.page if item.page is not None else (ev.page if ev else None),
        section_path=item.section_path,
        evidence_status=evidence_status,
        attributes=dict(item.attributes),
    )


def items_entries(
    items: Iterable[ContextItem],
    *,
    kind: EntryKind | None = None,
    evidence_status: EvidenceStatus | None = None,
) -> list[Entry]:
    """Entries for ``recall`` results; the kind is taken from the representation unless
    given (memories -> ``memory``, everything else -> ``knowledge``)."""
    out: list[Entry] = []
    for item in items:
        k: EntryKind = kind or (
            "memory" if str(item.representation).upper() == "MEMORY" else "knowledge"
        )
        out.append(item_entry(item, k, evidence_status=evidence_status))
    return out


def bundle_entries(bundle: ContextBundle, *, include_conversation: bool = True) -> list[Entry]:
    """Flatten a bundle: memories, knowledge, graph facts, summaries and (optionally) the
    conversation window, each tagged with the bundle's evidence status."""
    status: EvidenceStatus = bundle.evidence.status
    out: list[Entry] = []
    if include_conversation and bundle.conversation.rendered.strip():
        out.append(
            Entry(
                text=bundle.conversation.rendered,
                kind="conversation",
                citation=f"thread:{bundle.conversation.thread_id or ''}",
                source_type="THREAD",
                source_id=bundle.conversation.thread_id,
                evidence_status=status,
                attributes={"message_ids": list(bundle.conversation.message_ids)},
            )
        )
    for kind, items in (
        ("memory", bundle.memories),
        ("knowledge", bundle.knowledge),
        ("graph_fact", bundle.graph_facts),
        ("summary", bundle.summaries),
    ):
        out.extend(item_entry(i, kind, evidence_status=status) for i in items)  # type: ignore[arg-type]
    return out


def render_entries(entries: Iterable[Entry]) -> str:
    lines = []
    for e in entries:
        where = f" ({e.citation})" if e.citation else ""
        lines.append(f"- [{e.kind}] {e.text}{where}")
    return "\n".join(lines)


def render_bundle(bundle: ContextBundle) -> str:
    """Prompt-ready text: the service's rendering when present, else the flattened entries,
    always followed by the evidence line so a model knows what it may answer from."""
    body = bundle.rendered.strip() or render_entries(bundle_entries(bundle))
    ev = bundle.evidence
    line = f"[evidence: {ev.status.lower()}"
    if ev.missing_groups:
        line += f"; missing: {', '.join(ev.missing_groups)}"
    line += "]"
    return f"{body}\n{line}" if body else line


# -- hooks ------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRecord:
    """What ``after_run`` wrote (for tests and telemetry)."""

    messages: tuple[MessageAck, ...] = ()
    observations: tuple[ObservationAck, ...] = ()

    @property
    def recorded_messages(self) -> int:
        return len(self.messages)


class MemoryHooks:
    """Memory around one unit of work (a node, an agent step, a tool call).

    ``defaults`` is the adapter-level identity; :meth:`context` binds a ``MemoryContext``
    for a framework position via :func:`scope_fields`. Read errors propagate unless
    ``strict=False`` (then ``before_run`` returns ``None`` and logs); write errors always
    propagate (no silent data loss). ``require_evidence`` on a read always propagates.
    """

    def __init__(
        self,
        client: MemoryClient,
        *,
        namespace: str,
        defaults: Mapping[str, Any] | None = None,
        token_budget: int | None = None,
        strict: bool = True,
    ) -> None:
        self.client = client
        self.namespace = namespace
        self.defaults: dict[str, Any] = dict(defaults or {})
        self.token_budget = token_budget
        self.strict = strict

    def scope(self, **position: Any) -> dict[str, Any]:
        return scope_fields(defaults=self.defaults, **position)

    def context(self, **position: Any) -> MemoryContext:
        """A ``MemoryContext`` for a position (``thread_id``, ``runs``, ``overrides``...)."""
        return self.client.bind(**self.scope(**position))

    def key(self, prefix: str, *parts: Any) -> str:
        return idempotency_key(self.namespace, prefix, *parts)

    async def before_run(
        self,
        ctx: MemoryContext,
        query: str | None,
        *,
        require_evidence: bool = False,
        token_budget: int | None = None,
        strict: bool | None = None,
        **options: Any,
    ) -> ContextBundle | None:
        """The evidence-gated context bundle for ``query`` (``None``/blank skips recall)."""
        if not query or not str(query).strip():
            return None
        budget = self.token_budget if token_budget is None else token_budget
        try:
            return await ctx.context(
                str(query), token_budget=budget, require_evidence=require_evidence, **options
            )
        except Exception:
            if (self.strict if strict is None else strict) or require_evidence:
                raise
            log.exception("memory recall failed; continuing without context")
            return None

    async def record_message(
        self, ctx: MemoryContext, message: Message, key: str, **metadata: Any
    ) -> MessageAck:
        meta = {**message.metadata, **metadata}
        if message.role == "user":
            return await ctx.chat.user(message.content, idempotency_key=key, **meta)
        if message.role == "assistant":
            return await ctx.chat.assistant(message.content, idempotency_key=key, **meta)
        return await ctx.chat.internal(
            message.content, role=_INTERNAL_ROLES[message.role], idempotency_key=key, **meta
        )

    async def after_run(
        self,
        ctx: MemoryContext,
        *,
        key_parts: Sequence[Any],
        messages: Iterable[Message] = (),
        observations: Iterable[Observation | Mapping[str, Any] | str] = (),
        default_kind: str | None = None,
        hints: Mapping[str, Any] | None = None,
        **metadata: Any,
    ) -> RunRecord:
        """Record ``messages`` in the thread (visible: user/assistant; internal otherwise)
        and submit ``observations``; every write's idempotency key derives from
        ``key_parts`` (the step's framework identity) and the item's content."""
        acks: list[MessageAck] = []
        if ctx.scope.thread_id:
            for m in messages:
                if not m.content.strip():
                    continue
                key = self.key("msg", *key_parts, m.role, m.id or m.content)
                acks.append(await self.record_message(ctx, m, key))
        kind = default_kind or ("AGENT_RESULT" if ctx.scope.agent_id else "EVENT")
        obs: list[ObservationAck] = []
        for o in as_observations(list(observations)):
            k = o.kind or kind
            obs.append(
                await ctx.observe(
                    o.content,
                    kind=k,
                    idempotency_key=self.key("obs", *key_parts, k, o.content),
                    hints={**(hints or {}), **o.hints},
                    **{**metadata, **o.metadata},
                )
            )
        return RunRecord(tuple(acks), tuple(obs))

    async def share(
        self,
        ctx: MemoryContext,
        content: str,
        *,
        key_parts: Sequence[Any] = (),
        kind: str = "AGENT_RESULT",
        **metadata: Any,
    ) -> ObservationAck:
        """Publish ``content`` to the agent group explicitly (``SHARED`` / ``AGENT_GROUP``):
        the only way an agent's knowledge reaches other agents."""
        return await ctx.observe(
            content,
            kind=kind,
            idempotency_key=self.key("share", *key_parts, content),
            hints={"memory_type": "SHARED", "visibility": "AGENT_GROUP"},
            **metadata,
        )


def run_sync[T](coro: Awaitable[T]) -> T:
    """Run a coroutine from synchronous framework code (CrewAI storages, sync tools). When
    an event loop is already running in this thread, the coroutine runs on a fresh loop in
    a worker thread so the caller's loop is never blocked recursively."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()  # type: ignore[arg-type]
