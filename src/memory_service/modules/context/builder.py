"""Thin ContextBuilder: retrieval result + recent conversation -> bounded ContextBundle.

write / select / compress / isolate: select ranked evidence within a token budget, compress
the conversation window to its most recent turns (a rolling summary is attached by M9 when
available), isolate by scope (already enforced upstream). Bundles are cached by
scope fingerprint + revision fingerprint + query hash + retrieval config fingerprint.
"""

from __future__ import annotations

import contextlib
import json
import re
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
    UnusedEvidence,
)
from memory_service.domain.conversation import Message
from memory_service.domain.enums import EvidenceStatus, MessageKind
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.ids import stable_key
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.context.summaries import (
    SOURCE_CHARS,
    SUMMARY_SCHEMA,
    accept_abstractive,
)
from memory_service.modules.conversation.service import ConversationService
from memory_service.modules.ingestion.hierarchy import estimate_tokens
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.memory.ephemeral import EphemeralMemory
from memory_service.modules.retrieval.engine import (
    UNUSED_MAX,
    Candidate,
    RetrievalEngine,
    RetrievalResult,
)
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import evidence_status_total, stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.cache import CacheProvider, CacheUnavailable
from memory_service.ports.uow import UnitOfWorkFactory

log = get_logger(__name__)

_ROLLING_SYSTEM = (
    "You summarise the earlier part of a conversation that no longer fits the context "
    "window. Write at most {max_chars} characters capturing what the user asked for, the "
    "facts and decisions stated, and anything still open. Use only the conversation; no "
    'preamble. Return JSON only: {{"summary": "..."}}.'
)


#: Un-reranked items were never judged and ranked below everything that was, so their
#: relevance is mapped into a band strictly beneath the reranked floor. Order is preserved;
#: the number stops claiming a confidence nobody measured.
_TAIL_CEILING = 0.05


def _relevance(c: Candidate) -> tuple[float, str, float]:
    """The raw ranking number, what produced it, and a comparable 0..1 relevance.

    One field used to carry three incompatible scales: a cross-encoder logit (-11..+11) for
    items the reranker judged, an RRF fusion score (~0.001..0.25) for everything past
    ``candidate_k``, and a hardcoded 1.0 for exact identifier hits. In one measured response
    rank 27 scored +0.067 while rank 1 scored -4.14, so any client that sorted or thresholded
    on it got the ranking exactly backwards.
    """
    if c.rerank_score is not None:
        # a calibrated P(relevant): the cross-encoder applies its sigmoid
        value = min(max(float(c.rerank_score), 0.0), 1.0)
        return float(c.rerank_score), "cross_encoder", value
    if "exact" in (c.retrievers or []):
        return float(c.score), "exact", 1.0
    # RRF scores are sums of 1/(k+rank): bounded and monotone in rank, but not a probability
    return float(c.score), "fusion", min(float(c.score), 1.0) * _TAIL_CEILING


def candidate_to_item(c: Candidate) -> ContextItem:
    p = c.payload
    citation = {
        "chunk": f"chunk_id:{c.record_id}",
        "memory": f"memory_id:{c.record_id}",
        "fact": f"relation_id:{c.record_id}",
        "summary": f"summary_id:{c.record_id}",
    }.get(c.kind, f"{c.kind}:{c.record_id}")
    evidence = [
        EvidenceRef(
            source_type={"chunk": "document_chunk", "memory": "memory", "fact": "graph_fact"}.get(
                c.kind, c.kind
            ),
            source_id=c.record_id,
            document_id=p.get("document_id"),
            chunk_id=c.record_id if c.kind == "chunk" else p.get("chunk_id"),
            node_id=p.get("node_id"),
            page=p.get("page"),
            observed_at=datetime.now(UTC),
        )
    ]
    # _relevance existed, was correct, and was called by nothing but its own test, so every
    # item on the wire carried relevance=0.0 and score_kind="fusion" regardless of what
    # produced it — the incomparable-scale problem its docstring describes was still live.
    raw, kind, relevance = _relevance(c)
    return ContextItem(
        item_id=c.record_id,
        representation=c.representation,
        text=c.text,
        score=raw,
        score_kind=kind,
        relevance=relevance,
        retrievers=c.retrievers,
        evidence=evidence,
        citation=citation,
        document_id=p.get("document_id"),
        page=p.get("page"),
        section_path=p.get("section_path"),
        expanded_from=c.expanded_from,
        expansion_edge=c.expansion_edge,
        token_estimate=estimate_tokens(c.text),
        attributes={
            k: v
            for k, v in p.items()
            if k
            in (
                "predicate",
                "subject",
                "object",
                "attributes",
                "contradicts",
                "contributors",
                "memory_type",
                "visibility",
                "owner_principal",
                "confidence",
                "status",
            )
            and v not in (None, [], {})
        },
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
        assist: LLMAssist | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.engine = engine
        self.conversation = conversation
        self.cache = cache
        self.working = working
        self.cfg = settings
        self.retrieval_cfg = retrieval
        self.cache_ttl = cache_ttl_seconds
        self.assist = assist or LLMAssist.disabled()

    def _config_fingerprint(self) -> str:
        parts = [
            self.retrieval_cfg.model_dump_json(),
            self.cfg.model_dump_json(),
            self.engine.indexer.fingerprint,
        ]
        # bundles built with model assistance must not be served to a deployment without it
        llm_uses = [u for u in ("summaries", "query_expansion") if self.assist.wants(u)]
        if llm_uses:
            parts.append("llm:" + ",".join(llm_uses))
        return stable_key(*parts)

    async def _revision_fingerprint(self, ctx: MemoryExecutionContext) -> str:
        async with self.uow_factory() as uow:
            # Every revision a bundle's content depends on. AGENT and GRAPH were missing, so
            # an agent-scoped memory or a graph enrichment bumped a counter nobody read and
            # the bundle stayed cached regardless.
            keys = [
                (RevisionKind.TENANT, ""),
                (RevisionKind.USER, ctx.user_id or ""),
                (RevisionKind.THREAD, ctx.thread_id or ""),
                (RevisionKind.AGENT, ctx.agent_id or ""),
                (RevisionKind.GRAPH, ""),
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
            bundle_id = stable_key(
                ctx.tenant_id,
                ctx.scope_fingerprint(),
                revision_fp,
                self._config_fingerprint(),
                query,
                str(budget),
                ",".join(document_ids or []),
            )
            cache_key = self._cache_key(ctx.tenant_id, bundle_id)
            if self.cache is not None:
                try:
                    raw = await self.cache.get(cache_key)
                except CacheUnavailable:
                    raw = None
                if raw is not None:
                    bundle = ContextBundle.model_validate_json(raw)
                    return bundle.model_copy(update={"cache_hit": True})
            tokens_before = self.assist.tokens_used()
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
            spent = self.assist.tokens_used() - tokens_before
            bundle = bundle.model_copy(
                update={
                    "bundle_id": bundle_id,
                    "evidence": bundle.evidence.model_copy(update={"llm_tokens": spent}),
                }
            )
            if self.cache is not None:
                with contextlib.suppress(CacheUnavailable):
                    await self.cache.set(
                        cache_key, bundle.model_dump_json().encode(), ttl_seconds=self.cache_ttl
                    )
        evidence_status_total.labels(bundle.evidence.status.value).inc()
        await self._record_access(ctx, bundle)
        return bundle

    async def _record_access(self, ctx: MemoryExecutionContext, bundle: ContextBundle) -> None:
        """Count a memory as used when it actually reaches the caller.

        The forgetting policy scores
        ``importance * 0.5**(idle/half_life) * (1 - 0.5**(accesses + reinforcements))`` and
        reads ``access_count`` and ``last_accessed_at``. Migration 0006 added those columns for
        it. Nothing ever incremented them: ``bump_access`` existed on the repository and on the
        port and had no call site anywhere, so ``access_count`` stayed 0 for every memory and
        the access term collapsed to the constant 0.5. Decay was running on importance and
        recency alone, and the "frequently used memories survive" half of the policy silently
        did nothing — while looking, in the code and in the migration, entirely present.

        Served, not merely retrieved: candidates that were ranked but dropped by the token
        budget never reached anyone and must not count as use, or every query would reinforce
        memories nobody read.
        """
        ids = [item.item_id for item in bundle.memories if item.item_id]
        if not ids:
            return
        try:
            async with self.uow_factory() as uow:
                await uow.memories.bump_access(ctx.tenant_id, ids, at=datetime.now(UTC))
                await uow.commit()
        except Exception as exc:  # usage accounting must never fail a read
            log.warning("memory.access_bump_failed", error_message=str(exc), count=len(ids))

    async def cached(self, ctx: MemoryExecutionContext, bundle_id: str) -> ContextBundle | None:
        """A bundle built earlier under the caller's tenant, while it is still cached."""
        if self.cache is None or not bundle_id:
            return None
        try:
            raw = await self.cache.get(self._cache_key(ctx.tenant_id, bundle_id))
        except CacheUnavailable:
            return None
        return ContextBundle.model_validate_json(raw) if raw is not None else None

    @staticmethod
    def _cache_key(tenant_id: str, bundle_id: str) -> str:
        return f"ctx:{tenant_id}:{bundle_id}"

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
        window = render_window(ctx.thread_id, messages, self.cfg.conversation_token_budget)
        older = [m for m in messages if m.message_id not in set(window.message_ids)]
        if older:
            summary = rolling_summary(older)
            if self.assist.wants("summaries"):
                summary = await abstractive_rolling_summary(self.assist, older, summary)
            window = window.model_copy(update={"summary": summary})
        return window

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
        # a seed chunk travels with the companions its required evidence groups point at
        # (definition, footnote, cross-reference): either the whole unit fits the budget or
        # the seed is left out, so the bundle never carries a claim without its companion
        targets = result.diagnostics.get("evidence_targets") or {}
        seed_groups = result.diagnostics.get("evidence_seed_groups") or {}
        included_ids: set[str] = set()
        skipped_seeds: list[str] = []
        top_seed_skipped: list[str] = []  # groups of the best-ranked seed, if it was left out
        seen_seed = False
        items = [(c, candidate_to_item(c)) for c in result.candidates]
        by_node: dict[str, list[tuple[Any, ContextItem]]] = {}
        for c, item in items:
            node = c.payload.get("node_id")
            if node and c.kind in ("chunk", "summary"):
                by_node.setdefault(str(node), []).append((c, item))

        def companions_for(node: str) -> list[tuple[Any, ContextItem]]:
            out: list[tuple[Any, ContextItem]] = []
            for group in seed_groups.get(node, []):
                accepted = set(targets.get(group, []))
                if any(
                    str(cc.payload.get("node_id")) in accepted
                    for cc, _ in items
                    if cc.record_id in included_ids
                ):
                    continue  # already satisfied by something in the bundle
                for target in targets.get(group, []):
                    found = [
                        (cc, it)
                        for cc, it in by_node.get(target, [])
                        if cc.record_id not in included_ids
                    ]
                    if found:
                        out.append(found[0])
                        break
            return out

        for c, item in items:
            if c.record_id in included_ids:
                continue
            if c.kind == "chunk" and len(knowledge) < self.cfg.knowledge_max:
                node = str(c.payload.get("node_id") or "")
                unit = [(c, item)] + (companions_for(node) if node in seed_groups else [])
                cost = sum(it.token_estimate for _, it in unit)
                if cost > remaining:
                    if len(unit) > 1:
                        skipped_seeds.append(c.record_id)
                        if not seen_seed:
                            top_seed_skipped.extend(seed_groups.get(node, []))
                    seen_seed = seen_seed or node in seed_groups
                    continue
                seen_seed = seen_seed or node in seed_groups
                for cc, it in unit:
                    if cc.record_id in included_ids:
                        continue
                    if cc.kind == "summary":
                        summaries.append(it)
                    else:
                        knowledge.append(it)
                    included_ids.add(cc.record_id)
                remaining -= cost
                continue
            if item.token_estimate > remaining:
                continue
            if c.kind == "memory" and len(memories) < self.cfg.memories_max:
                memories.append(item)
            elif c.kind == "fact" and len(graph_facts) < self.cfg.graph_facts_max:
                graph_facts.append(item)
            elif c.kind == "summary" and len(summaries) < self.cfg.summaries_max:
                summaries.append(item)
            else:
                continue
            included_ids.add(c.record_id)
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
        # the report must describe what is IN the bundle: companions that were retrieved but
        # dropped by the token budget count as missing
        included_nodes = {
            ev.node_id for item in (*knowledge, *summaries) for ev in item.evidence if ev.node_id
        }
        if report.required_groups and report.status is not EvidenceStatus.INSUFFICIENT:
            # groups required only by seeds that were left out (with their companions) no
            # longer apply to what is in the bundle
            seeds_in = {
                str(c.payload.get("node_id"))
                for c in result.candidates
                if c.record_id in included_ids and c.kind == "chunk"
            }
            applicable = {
                g for node, names in seed_groups.items() if node in seeds_in for g in names
            } or set(report.required_groups)
            report = report.model_copy(
                update={
                    "required_groups": [g for g in report.required_groups if g in applicable],
                    "satisfied_groups": [g for g in report.satisfied_groups if g in applicable],
                    "missing_groups": [g for g in report.missing_groups if g in applicable],
                    "notes": [
                        *report.notes,
                        *(
                            [
                                f"{len(skipped_seeds)} lower-ranked passage(s) left out: their "
                                "companion evidence did not fit the token budget"
                            ]
                            if skipped_seeds
                            else []
                        ),
                    ],
                }
            )
            targets_by_group = dict(
                zip(
                    report.required_groups,
                    [list(targets.get(g, [])) for g in report.required_groups],
                    strict=False,
                )
            )
            dropped = [
                g for g, ids in targets_by_group.items() if ids and not (set(ids) & included_nodes)
            ]
            if dropped:
                missing = sorted(set(report.missing_groups) | set(dropped))
                report = report.model_copy(
                    update={
                        "status": EvidenceStatus.INCOMPLETE,
                        "missing_groups": missing,
                        "satisfied_groups": [
                            g for g in report.satisfied_groups if g not in missing
                        ],
                        "notes": [*report.notes, "companion evidence exceeded the token budget"],
                    }
                )
        if not (knowledge or memories or graph_facts or summaries):
            report = report.model_copy(update={"status": EvidenceStatus.INSUFFICIENT})
        # what the retriever found but this bundle does not carry (ranked out, over budget):
        # the grounding cascade checks answers against it for contradictions
        unused = [
            UnusedEvidence(item_id=c.record_id, kind=c.kind, text=c.text)
            for c, _ in items
            if c.record_id not in included_ids and c.kind in ("chunk", "memory", "summary")
        ]
        unused.extend(
            UnusedEvidence(item_id=str(u["record_id"]), kind=str(u["kind"]), text=str(u["text"]))
            for u in result.diagnostics.get("unused") or []
        )
        report = report.model_copy(update={"unused": unused[:UNUSED_MAX]})
        diagnostics = {
            k: v
            for k, v in result.diagnostics.items()
            if k not in ("evidence", "evidence_targets", "evidence_seed_groups", "unused")
        }
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


def rolling_summary(messages: Sequence[Message], *, max_chars: int = 600) -> str:
    """Deterministic digest of turns that fell out of the window: role + first sentence."""
    parts: list[str] = []
    used = 0
    for m in messages:
        if m.kind is not MessageKind.VISIBLE or not m.content.strip():
            continue
        first = re.split(r"(?<=[.!?])\s+", m.content.strip(), maxsplit=1)[0][:160]
        line = f"{m.role.value.lower()}: {first}"
        if used + len(line) > max_chars:
            parts.append("…")
            break
        parts.append(line)
        used += len(line) + 1
    return "\n".join(parts)


async def abstractive_rolling_summary(
    assist: LLMAssist,
    messages: Sequence[Message],
    extractive: str,
    *,
    max_chars: int = 600,
) -> str:
    """Model-written digest of the turns that fell out of the window, given the deterministic
    digest and the (bounded, most recent) source turns; the deterministic digest otherwise."""
    if not assist.wants("summaries"):
        return extractive
    lines = [
        f"{m.role.value.lower()}: {' '.join(m.content.split())}"
        for m in messages
        if m.kind is MessageKind.VISIBLE and m.content.strip()
    ]
    source = "\n".join(lines)[-SOURCE_CHARS:]
    result = await assist.structured(
        "summaries",
        system=_ROLLING_SYSTEM.format(max_chars=max_chars),
        user=f"Deterministic digest:\n{extractive}\n\nEarlier conversation:\n{source}",
        schema=SUMMARY_SCHEMA,
        max_tokens=max(128, max_chars // 2),
    )
    return accept_abstractive(result, max_chars=max_chars) or extractive


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
