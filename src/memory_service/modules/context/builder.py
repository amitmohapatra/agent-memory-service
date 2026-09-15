"""Thin ContextBuilder: retrieval result + recent conversation -> bounded ContextBundle.

write / select / compress / isolate: select ranked evidence within a token budget, compress
the conversation window to its most recent turns (a rolling summary is attached by M9 when
available), isolate by scope (already enforced upstream). Bundles are cached by
scope fingerprint + revision fingerprint + query hash + retrieval config fingerprint.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from memory_service.config.settings import ContextSettings, RetrievalSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.context_bundle import (
    ContextBundle,
    ContextItem,
    ConversationWindow,
    EvidenceReport,
)
from memory_service.domain.conversation import Message
from memory_service.domain.enums import EvidenceStatus, MessageKind, Representation
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.ids import stable_key
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.conversation.service import ConversationService
from memory_service.modules.ingestion.hierarchy import estimate_tokens
from memory_service.modules.memory.ephemeral import EphemeralMemory
from memory_service.modules.retrieval.engine import Candidate, RetrievalEngine, RetrievalResult
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import evidence_status_total, stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.cache import CacheProvider, CacheUnavailable
from memory_service.ports.uow import UnitOfWorkFactory

log = get_logger(__name__)


def candidate_to_item(c: Candidate) -> ContextItem:
    p = c.payload
    citation = f"chunk_id:{c.record_id}" if c.kind == "chunk" else f"memory_id:{c.record_id}"
    evidence = [
        EvidenceRef(
            source_type="document_chunk" if c.kind == "chunk" else "memory",
            source_id=c.record_id,
            document_id=p.get("document_id"),
            chunk_id=c.record_id if c.kind == "chunk" else None,
            node_id=p.get("node_id"),
            page=p.get("page"),
            observed_at=datetime.now(UTC),
        )
    ]
    return ContextItem(
        item_id=c.record_id,
        representation=c.representation
        if c.expansion_edge is None
        else Representation(p.get("representation", "CHUNK")),
        text=c.text,
        score=c.rerank_score if c.rerank_score is not None else c.score,
        retrievers=c.retrievers,
        evidence=evidence,
        citation=citation,
        document_id=p.get("document_id"),
        page=p.get("page"),
        section_path=p.get("section_path"),
        expanded_from=c.expanded_from,
        expansion_edge=c.expansion_edge,
        token_estimate=estimate_tokens(c.text),
    )


class ContextBuilder:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        engine: RetrievalEngine,
        conversation: ConversationService,
        cache: CacheProvider | None,
        *,
        settings: ContextSettings,
        retrieval: RetrievalSettings,
        cache_ttl_seconds: int = 300,
        working: EphemeralMemory | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.engine = engine
        self.conversation = conversation
        self.cache = cache
        self.working = working
        self.cfg = settings
        self.retrieval_cfg = retrieval
        self.cache_ttl = cache_ttl_seconds

    def _config_fingerprint(self) -> str:
        return stable_key(
            self.retrieval_cfg.model_dump_json(),
            self.cfg.model_dump_json(),
            self.engine.indexer.fingerprint,
        )

    async def _revision_fingerprint(self, ctx: MemoryExecutionContext) -> str:
        async with self.uow_factory() as uow:
            keys = [
                (RevisionKind.TENANT, ""),
                (RevisionKind.USER, ctx.user_id or ""),
                (RevisionKind.THREAD, ctx.thread_id or ""),
            ]
            values = await uow.revisions.get_many(ctx.tenant_id, keys)
        return stable_key(*(f"{k}={v}" for k, v in sorted(values.items())))

    async def build(
        self,
        ctx: MemoryExecutionContext,
        query: str,
        *,
        token_budget: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> ContextBundle:
        budget = token_budget or self.cfg.token_budget
        with (
            span("context.build", tenant_id=ctx.tenant_id),
            stage_seconds.labels("context.build").time(),
        ):
            revision_fp = await self._revision_fingerprint(ctx)
            cache_key = "ctx:" + stable_key(
                ctx.tenant_id,
                ctx.scope_fingerprint(),
                revision_fp,
                self._config_fingerprint(),
                query,
                str(budget),
                ",".join(document_ids or []),
            )
            if self.cache is not None:
                try:
                    raw = await self.cache.get(cache_key)
                except CacheUnavailable:
                    raw = None
                if raw is not None:
                    bundle = ContextBundle.model_validate_json(raw)
                    return bundle.model_copy(update={"cache_hit": True})
            result = await self.engine.retrieve(ctx, query, document_ids=document_ids)
            window = await self._conversation_window(ctx, result)
            if self.working is not None and result.routed.needs_memories:
                for i, item in enumerate(await self.working.recall(ctx)):
                    result.candidates.insert(
                        i,
                        Candidate(
                            record_id=f"wm_{i}",
                            kind="memory",
                            text=str(item.get("content", "")),
                            score=1.0,
                            retrievers=["working"],
                            payload={"memory_type": item.get("memory_type", "WORKING")},
                        ),
                    )
            bundle = self._assemble(query, result, window, budget, revision_fp)
            if self.cache is not None:
                with contextlib.suppress(CacheUnavailable):
                    await self.cache.set(
                        cache_key, bundle.model_dump_json().encode(), ttl_seconds=self.cache_ttl
                    )
        evidence_status_total.labels(bundle.evidence.status.value).inc()
        return bundle

    async def _conversation_window(
        self, ctx: MemoryExecutionContext, result: RetrievalResult
    ) -> ConversationWindow:
        if not ctx.thread_id or not result.routed.needs_conversation:
            return ConversationWindow(thread_id=ctx.thread_id)
        async with self.uow_factory() as uow:
            thread = await uow.threads.get(ctx.tenant_id, ctx.thread_id)
            if thread is None:
                return ConversationWindow(thread_id=ctx.thread_id)
            messages = await self.conversation.list_messages(
                uow, ctx, ctx.thread_id, limit=self.cfg.conversation_max_messages
            )
        return render_window(ctx.thread_id, messages, self.cfg.conversation_token_budget)

    def _assemble(
        self,
        query: str,
        result: RetrievalResult,
        window: ConversationWindow,
        budget: int,
        revision_fp: str,
    ) -> ContextBundle:
        remaining = budget - window.token_estimate
        memories: list[ContextItem] = []
        knowledge: list[ContextItem] = []
        graph_facts: list[ContextItem] = []
        summaries: list[ContextItem] = []
        for c in result.candidates:
            item = candidate_to_item(c)
            if item.token_estimate > remaining:
                continue
            if c.kind == "memory" and len(memories) < self.cfg.memories_max:
                memories.append(item)
            elif c.kind == "fact" and len(graph_facts) < self.cfg.graph_facts_max:
                graph_facts.append(item)
            elif c.kind == "summary" and len(summaries) < self.cfg.summaries_max:
                summaries.append(item)
            elif c.kind == "chunk" and len(knowledge) < self.cfg.knowledge_max:
                knowledge.append(item)
            else:
                continue
            remaining -= item.token_estimate
        evidence = result.diagnostics.get("evidence")
        report = (
            EvidenceReport.model_validate(evidence)
            if isinstance(evidence, dict)
            else EvidenceReport(
                status=EvidenceStatus.COMPLETE
                if (knowledge or memories or graph_facts)
                else EvidenceStatus.INSUFFICIENT
            )
        )
        diagnostics = {k: v for k, v in result.diagnostics.items() if k != "evidence"}
        return ContextBundle(
            query=query,
            query_type=result.routed.query_type,
            conversation=window,
            memories=memories,
            knowledge=knowledge,
            graph_facts=graph_facts,
            summaries=summaries,
            evidence=report,
            token_budget=budget,
            token_estimate=budget - remaining,
            revision_fingerprint=revision_fp,
            diagnostics=diagnostics,
        )


def render_window(
    thread_id: str, messages: Sequence[Message], token_budget: int
) -> ConversationWindow:
    """Most recent visible messages that fit the budget, oldest first."""
    chosen: list[Message] = []
    used = 0
    for m in reversed(messages):
        if m.kind is not MessageKind.VISIBLE:
            continue
        t = estimate_tokens(m.content) + 4
        if used + t > token_budget and chosen:
            break
        chosen.insert(0, m)
        used += t
    rendered = "\n".join(f"{m.role.value}: {m.content}" for m in chosen)
    return ConversationWindow(
        thread_id=thread_id,
        message_ids=[m.message_id for m in chosen],
        rendered=rendered,
        token_estimate=used,
    )


def bundle_to_api(bundle: ContextBundle) -> dict[str, Any]:
    data = json.loads(bundle.model_dump_json())
    data["rendered"] = bundle.render()
    return data
