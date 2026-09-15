"""Observational memory per thread.

``ThreadObserver`` compresses the turns that fell out of a thread's hot window into dated
observation notes (deterministic, extractive: attributes, preferences, decisions, facts,
events, tasks, open questions, each with the turn it came from), stored as one
``MemoryType.OBSERVATION`` memory per batch, thread-scoped, with the source messages as
evidence, and indexed like any other memory. With the ``observation_refinement`` LLM use the
notes are rewritten by the model; the extractive notes stay stored and are the fallback.

``ObservationReflector`` merges observation notes into semantic memory through the
observation pipeline, so consolidation, admission and the landing pass all apply, and marks
each observation reflected exactly once.

Both run from the periodic ``memory.observe`` job and on demand when the pipeline notices a
thread has grown past its hot window.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from memory_service.config.settings import MemoryIntelligenceSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.conversation import Message, Thread
from memory_service.domain.enums import (
    Lifetime,
    MemoryType,
    MessageKind,
    MessageRole,
    ObservationKind,
    ScopeLevel,
    Visibility,
)
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.ids import content_hash
from memory_service.domain.memory import CanonicalMemory, Scope
from memory_service.domain.observation import Observation
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.memory.native import (
    _QUESTION,
    NativeMemoryIntelligence,
    normalized_hash,
    split_clauses,
    split_sentences,
)
from memory_service.modules.memory.pipeline import (
    TASK_MEMORY_INDEX,
    ObservationPipeline,
    build_memory,
    keys_for,
)
from memory_service.observability.logging import get_logger
from memory_service.ports.intelligence import ConsolidationOutcome, MemoryCandidate
from memory_service.ports.tasks import JobSpec, Queue
from memory_service.ports.uow import UnitOfWork, UnitOfWorkFactory

log = get_logger(__name__)

OBSERVATION_CATEGORY = "observation"
NOTE_KINDS = (
    "attribute",
    "preference",
    "instruction",
    "decision",
    "fact",
    "procedure",
    "task",
    "event",
    "question",
    "said",
)
_REFLECTABLE = frozenset(NOTE_KINDS) - {"question", "said"}
_MAX_NOTE_CHARS = 240
_MAX_SOURCE_CHARS = 5000
_REFINE_SYSTEM = (
    "You rewrite the deterministic observation notes of an older part of a conversation into "
    "at most {n} compact, dated notes: facts, decisions, preferences, tasks and open "
    "questions that a future turn may need. Keep every name, number and date, invent "
    "nothing, drop small talk. kind is one of {kinds}."
)
_REFINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["notes"],
    "properties": {
        "notes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["kind", "text"],
                "properties": {
                    "kind": {"type": "string", "enum": list(NOTE_KINDS)},
                    "text": {"type": "string"},
                },
            },
        }
    },
}


def thread_scope(thread: Thread) -> Scope:
    return Scope(
        level=ScopeLevel.THREAD,
        tenant_id=thread.tenant_id,
        workspace_id=thread.workspace_id,
        thread_id=thread.thread_id,
    )


def thread_context(thread: Thread) -> MemoryExecutionContext:
    return MemoryExecutionContext(
        tenant_id=thread.tenant_id,
        workspace_id=thread.workspace_id,
        user_id=thread.owner_user_id,
        thread_id=thread.thread_id,
    )


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s+", " ".join(text.split()), maxsplit=1)[0][:_MAX_NOTE_CHARS]


def extract_notes(
    messages: Sequence[Message],
    native: NativeMemoryIntelligence,
    ctx: MemoryExecutionContext,
    *,
    max_notes: int,
) -> list[dict[str, Any]]:
    """Deterministic extractive notes, oldest first, one dict per note with its turn ref."""
    notes: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(kind: str, text: str, m: Message) -> None:
        key = normalized_hash(text)
        if key in seen or len(notes) >= max_notes:
            return
        seen.add(key)
        notes.append(
            {
                "kind": kind,
                "text": text[:_MAX_NOTE_CHARS],
                "message_id": m.message_id,
                "sequence": m.sequence,
                "role": m.role.value,
                "at": m.occurred_at.isoformat(),
            }
        )

    for m in messages:
        if m.kind is not MessageKind.VISIBLE or not m.content.strip():
            continue
        speaker = ctx if m.role is MessageRole.USER else ctx.model_copy(update={"agent_id": "assistant"})
        evidence = [
            EvidenceRef(
                source_type="message",
                source_id=m.message_id,
                message_id=m.message_id,
                observed_at=m.occurred_at,
            )
        ]
        hit = False
        for sentence in split_sentences(m.content, max_sentences=12):
            for clause in split_clauses(sentence):
                if _QUESTION.search(clause):
                    if m.role is MessageRole.USER:
                        add("question", clause, m)
                        hit = True
                    continue
                cand = native._from_sentence(clause, speaker, evidence)
                if cand is None:
                    continue
                if m.role is not MessageRole.USER and cand.memory_type in (
                    MemoryType.USER,
                    MemoryType.PREFERENCE,
                ):
                    continue
                kind = cand.category or cand.memory_type.value.lower()
                add(kind if kind in NOTE_KINDS else "fact", clause, m)
                hit = True
        if not hit and m.role is MessageRole.USER:
            add("said", _first_sentence(m.content), m)
    return notes


def render_notes(notes: Sequence[dict[str, Any]], *, header: str) -> str:
    lines = [header]
    for n in notes:
        when = str(n.get("at", ""))[:10]
        lines.append(f"- [{when}] {n['kind']}: {n['text']} (turn {n.get('sequence', '?')})")
    return "\n".join(lines)


class ThreadObserver:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        native: NativeMemoryIntelligence,
        *,
        settings: MemoryIntelligenceSettings,
        assist: LLMAssist | None = None,
        active_window: timedelta = timedelta(hours=24),
        scan_limit: int = 500,
    ) -> None:
        self.uow_factory = uow_factory
        self.native = native
        self.cfg = settings
        self.assist = assist or LLMAssist.disabled()
        self.active_window = active_window
        self.scan_limit = scan_limit

    async def observations(self, uow: UnitOfWork, thread: Thread) -> list[CanonicalMemory]:
        rows = await uow.memories.list_scope(
            thread.tenant_id,
            scope_keys=[thread_scope(thread).key()],
            memory_types=[MemoryType.OBSERVATION.value],
            limit=500,
        )
        return sorted(rows, key=lambda m: int(m.system_metadata.get("sequence_to", 0)))

    async def observe_all(self, *, now: datetime | None = None) -> list[str]:
        now = now or datetime.now(UTC)
        async with self.uow_factory() as uow:
            threads = await uow.threads.list_active(
                since=now - self.active_window, limit=self.scan_limit
            )
        created: list[str] = []
        for t in threads:
            created += await self.observe_thread(t.tenant_id, t.thread_id, now=now)
        return created

    async def observe_thread(
        self, tenant_id: str, thread_id: str, *, now: datetime | None = None, force: bool = False
    ) -> list[str]:
        """Compress the turns older than the hot window that no observation covers yet.
        ``force`` observes even when fewer than a batch of new turns is waiting."""
        now = now or datetime.now(UTC)
        async with self.uow_factory() as uow:
            thread = await uow.threads.get(tenant_id, thread_id)
            if thread is None:
                return []
            latest = await uow.messages.list_thread(tenant_id, thread_id, limit=1)
            if not latest:
                return []
            cutoff = latest[-1].sequence - self.cfg.observer_hot_window_messages + 1
            existing = await self.observations(uow, thread)
            covered = max((int(m.system_metadata.get("sequence_to", 0)) for m in existing), default=0)
            pending = cutoff - 1 - covered
            if pending < (1 if force else self.cfg.observer_batch_messages):
                return []
            batch = min(pending, self.cfg.observer_batch_messages * 3)
            older = await uow.messages.list_thread(
                tenant_id, thread_id, limit=batch, before_sequence=cutoff
            )
            messages = [m for m in older if m.sequence > covered]
            if not messages:
                return []
            ctx = thread_context(thread)
            notes = extract_notes(messages, self.native, ctx, max_notes=self.cfg.observer_max_notes)
            if not notes:
                return []
            memory = await self._memory(ctx, thread, messages, notes, now=now)
            await uow.memories.add(
                memory, visibility_keys=keys_for(memory.scope, memory.visibility, ctx)
            )
            await uow.enqueue(
                JobSpec(
                    task_name=TASK_MEMORY_INDEX,
                    queue=Queue.EMBEDDING,
                    payload={"tenant_id": tenant_id, "memory_ids": [memory.memory_id]},
                    idempotency_key=f"memidx:observe:{memory.memory_id}",
                    tenant_id=tenant_id,
                )
            )
            await uow.revisions.bump(tenant_id, RevisionKind.THREAD, thread_id)
            await uow.commit()
        log.info(
            "memory.observed",
            tenant_id=tenant_id,
            thread_id=thread_id,
            turns=len(messages),
            notes=len(notes),
        )
        return [memory.memory_id]

    async def _memory(
        self,
        ctx: MemoryExecutionContext,
        thread: Thread,
        messages: Sequence[Message],
        notes: list[dict[str, Any]],
        *,
        now: datetime,
    ) -> CanonicalMemory:
        first, last = messages[0], messages[-1]
        header = (
            f"Observations {first.occurred_at.date().isoformat()} to "
            f"{last.occurred_at.date().isoformat()} (turns {first.sequence}-{last.sequence}):"
        )
        extractive = render_notes(notes, header=header)
        content = extractive
        refined = await self._refine(messages, notes)
        if refined:
            content = render_notes(refined, header=header)
        cited = {n["message_id"] for n in notes}
        candidate = MemoryCandidate(
            content=content,
            memory_type=MemoryType.OBSERVATION,
            lifetime=Lifetime.LONG_TERM,
            visibility=Visibility.THREAD,
            subject=f"thread:{thread.thread_id}",
            predicate="observation",
            evidence=[
                EvidenceRef(
                    source_type="message",
                    source_id=m.message_id,
                    message_id=m.message_id,
                    observed_at=m.occurred_at,
                )
                for m in messages
                if m.message_id in cited
            ]
            or [EvidenceRef(source_type="message", source_id=first.message_id, observed_at=now)],
            confidence=0.7,
            importance=0.4,
            category=OBSERVATION_CATEGORY,
            valid_from=first.occurred_at,
        )
        memory = build_memory(candidate, ctx, now=now)
        memory.system_metadata.update(
            {
                "sequence_from": first.sequence,
                "sequence_to": last.sequence,
                "observed_from": first.occurred_at.isoformat(),
                "observed_to": last.occurred_at.isoformat(),
                "message_count": len(messages),
                "notes": refined or notes,
                "extractive_notes": notes if refined else None,
                "refined": bool(refined),
            }
        )
        return memory

    async def _refine(
        self, messages: Sequence[Message], notes: list[dict[str, Any]]
    ) -> list[dict[str, Any]] | None:
        if not self.assist.wants("observation_refinement"):
            return None
        source = "\n".join(
            f"[turn {m.sequence}] {m.role.value.lower()}: {' '.join(m.content.split())}"
            for m in messages
            if m.kind is MessageKind.VISIBLE and m.content.strip()
        )[-_MAX_SOURCE_CHARS:]
        out = await self.assist.structured(
            "observation_refinement",
            system=_REFINE_SYSTEM.format(n=self.cfg.observer_max_notes, kinds=", ".join(NOTE_KINDS)),
            user="Deterministic notes:\n"
            + "\n".join(f"- {n['kind']}: {n['text']}" for n in notes)
            + f"\n\nSource turns:\n{source}",
            schema=_REFINE_SCHEMA,
            max_tokens=600,
        )
        raw = out.get("notes") if isinstance(out, dict) else None
        if not isinstance(raw, list) or not raw:
            return None
        by_turn = notes[0] if notes else {}
        refined: list[dict[str, Any]] = []
        for item in raw[: self.cfg.observer_max_notes]:
            if not isinstance(item, dict):
                continue
            kind, text = item.get("kind"), item.get("text")
            if kind not in NOTE_KINDS or not isinstance(text, str) or not text.strip():
                continue
            refined.append(
                {
                    "kind": kind,
                    "text": " ".join(text.split())[:_MAX_NOTE_CHARS],
                    "message_id": by_turn.get("message_id"),
                    "sequence": by_turn.get("sequence"),
                    "role": "REFINED",
                    "at": by_turn.get("at"),
                }
            )
        return refined or None


class ObservationReflector:
    """Observation notes -> memory candidates -> the pipeline's consolidation and admission."""

    def __init__(self, uow_factory: UnitOfWorkFactory, pipeline: ObservationPipeline) -> None:
        self.uow_factory = uow_factory
        self.pipeline = pipeline

    async def reflect_thread(
        self, tenant_id: str, thread_id: str, *, now: datetime | None = None
    ) -> list[ConsolidationOutcome]:
        now = now or datetime.now(UTC)
        outcomes: list[ConsolidationOutcome] = []
        async with self.uow_factory() as uow:
            thread = await uow.threads.get(tenant_id, thread_id)
            if thread is None:
                return []
            rows = await uow.memories.list_scope(
                tenant_id,
                scope_keys=[thread_scope(thread).key()],
                memory_types=[MemoryType.OBSERVATION.value],
                limit=500,
            )
            pending = [m for m in rows if "reflected_at" not in m.system_metadata]
            if not pending:
                return []
            ctx = thread_context(thread)
            affected: set[str] = set()
            for obs in sorted(pending, key=lambda m: int(m.system_metadata.get("sequence_to", 0))):
                candidates = await self._candidates(obs, ctx)
                affected |= await self.pipeline._apply_all(uow, ctx, candidates, outcomes)
                obs.system_metadata["reflected_at"] = now.isoformat()
                obs.updated_at = now
                await uow.memories.update(obs)
            if affected:
                await uow.enqueue(
                    JobSpec(
                        task_name=TASK_MEMORY_INDEX,
                        queue=Queue.EMBEDDING,
                        payload={"tenant_id": tenant_id, "memory_ids": sorted(affected)},
                        idempotency_key=f"memidx:reflect-obs:{pending[0].memory_id}",
                        tenant_id=tenant_id,
                    )
                )
                if ctx.user_id:
                    await uow.revisions.bump(tenant_id, RevisionKind.USER, ctx.user_id)
                await uow.revisions.bump(tenant_id, RevisionKind.THREAD, thread_id)
            await uow.commit()
        log.info(
            "memory.observations_reflected",
            tenant_id=tenant_id,
            thread_id=thread_id,
            observations=len(pending),
            decisions={o.decision.value: 1 for o in outcomes},
        )
        return outcomes

    async def _candidates(
        self, obs: CanonicalMemory, ctx: MemoryExecutionContext
    ) -> list[MemoryCandidate]:
        out: list[MemoryCandidate] = []
        for note in obs.system_metadata.get("notes") or []:
            if note.get("kind") not in _REFLECTABLE or note.get("role") == "ASSISTANT":
                continue
            observation = Observation(
                tenant_id=ctx.tenant_id,
                kind=ObservationKind.MESSAGE,
                content=str(note.get("text", "")),
                content_hash=content_hash(str(note.get("text", ""))),
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                thread_id=ctx.thread_id,
                principal_id=ctx.principal_id,
                message_id=note.get("message_id"),
                occurred_at=_parse_at(note.get("at")) or obs.temporal.observed_at,
            )
            provider = self.pipeline.provider
            for cand in await provider.extract(observation, ctx):
                cand = await provider.classify(cand, ctx)
                out.append(
                    cand.model_copy(
                        update={
                            "evidence": [
                                *cand.evidence,
                                EvidenceRef(
                                    source_type="memory",
                                    source_id=obs.memory_id,
                                    observed_at=obs.temporal.observed_at,
                                ),
                            ]
                        }
                    )
                )
        return out


def _parse_at(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
