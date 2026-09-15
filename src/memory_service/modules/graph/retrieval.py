"""GraphStage: a RetrievalEngine post-stage for entity/relation, multi-hop, temporal and
decision queries.

    query entities -> bounded neighbourhood (1 hop, 2 for multi-hop) -> ranked facts
    -> the facts' *evidence chunks* are pulled in as knowledge candidates (multi-hop
       retrieval: 'Adjusted EBITDA' --co_occurs_with--> 'Restructuring' -> the page-14 chunk)

Facts become ``kind="fact"`` candidates (bundle bucket ``graph_facts``, citation
``relation_id:rel_…``) and never displace the reranked evidence: they are appended, and the
expansion chunks are visibility-checked against the same specification the store used.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import QueryType
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.graph.service import GraphService
from memory_service.modules.memory.native import parse_date
from memory_service.modules.retrieval.engine import Candidate
from memory_service.modules.retrieval.router import RoutedQuery
from memory_service.observability.metrics import stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.intelligence import Entity, Relation
from memory_service.ports.uow import UnitOfWorkFactory

_MULTI_HOP_TYPES = {QueryType.DOCUMENT_MULTI_HOP, QueryType.ENTITY_RELATION}
_STRUCTURAL = {"mentioned_in", "co_occurs_with"}


def fact_candidate(r: Relation, names: dict[str, str]) -> Candidate:
    subject = names.get(r.subject_id, r.subject_id)
    obj = names.get(r.object_id, r.object_id)
    text = r.fact_text or f"{subject} {r.predicate.replace('_', ' ')} {obj}"
    ev = r.evidence[0] if r.evidence else None
    return Candidate(
        record_id=r.relation_id,
        kind="fact",
        text=text,
        score=r.confidence,
        retrievers=["graph"],
        payload={
            "subject": subject,
            "predicate": r.predicate,
            "object": obj,
            "valid_from": r.valid_from.isoformat() if r.valid_from else None,
            "valid_to": r.valid_to.isoformat() if r.valid_to else None,
            "status": r.status,
            "observed_at": r.observed_at.isoformat(),
            "memory_id": r.memory_id,
            "document_id": r.document_id or (ev.document_id if ev else None),
            "chunk_id": ev.chunk_id if ev else None,
            "node_id": ev.node_id if ev else None,
            "page": ev.page if ev else r.attributes.get("page"),
            "representation": "RELATION",
        },
    )


class GraphStage:
    name = "graph"

    def __init__(
        self,
        graph: GraphService,
        uow_factory: UnitOfWorkFactory,
        *,
        max_facts: int = 12,
        max_expansion_chunks: int = 6,
        max_visited: int = 80,
    ) -> None:
        self.graph = graph
        self.uow_factory = uow_factory
        self.max_facts = max_facts
        self.max_expansion_chunks = max_expansion_chunks
        self.max_visited = max_visited  # retrieval-time traversal is tighter than /v1/graph

    async def __call__(
        self,
        ctx: MemoryExecutionContext,
        routed: RoutedQuery,
        candidates: list[Candidate],
        visibility: VisibilitySpecification,
        diagnostics: dict[str, Any],
    ) -> list[Candidate]:
        if not routed.needs_graph:
            return candidates
        with span("retrieval.graph"), stage_seconds.labels("retrieval.graph").time():
            as_of = parse_date(routed.query) if routed.query_type is QueryType.TEMPORAL else None
            answer = await self.graph.query(
                ctx,
                query=routed.query,
                hops=2 if routed.query_type in _MULTI_HOP_TYPES else 1,
                as_of=as_of,
                visibility=visibility,
                max_visited=self.max_visited,
            )
            diagnostics["graph"] = {
                "matched": [e.canonical_name for e in answer.matched],
                "visited": answer.visited,
                "relations": len(answer.relations),
                "as_of": as_of.isoformat() if as_of else None,
            }
            if not answer.relations:
                return candidates
            names = {e.entity_id: e.name for e in answer.entities}
            seeds = {e.entity_id for e in answer.matched}
            # typed facts (from memories) first, then structural document facts touching a seed
            ranked = sorted(
                answer.relations,
                key=lambda r: (
                    r.predicate in _STRUCTURAL,
                    not (r.subject_id in seeds or r.object_id in seeds),
                    -r.confidence,
                    r.relation_id,
                ),
            )
            existing_ids = {c.record_id for c in candidates}
            facts = [fact_candidate(r, names) for r in ranked[: self.max_facts]]
            candidates.extend(f for f in facts if f.record_id not in existing_ids)
            # multi-hop expansion: pull the evidence chunks the facts point at
            chunk_ids: list[str] = []
            for r in ranked:
                for ev in r.evidence:
                    if (
                        ev.chunk_id
                        and ev.chunk_id not in existing_ids
                        and ev.chunk_id not in chunk_ids
                    ):
                        chunk_ids.append(ev.chunk_id)
                if len(chunk_ids) >= self.max_expansion_chunks:
                    break
            if chunk_ids:
                added = await self._expand(ctx, chunk_ids, visibility, names_of=answer.entities)
                # expansion chunks go right after the ranked evidence, before the facts, so a
                # caller's ``limit`` on evidence keeps the best-ranked items first
                first_fact = next(
                    (i for i, c in enumerate(candidates) if c.kind == "fact"), len(candidates)
                )
                candidates[first_fact:first_fact] = added
                diagnostics["graph"]["expansion_chunks"] = len(added)
        return candidates

    async def _expand(
        self,
        ctx: MemoryExecutionContext,
        chunk_ids: list[str],
        visibility: VisibilitySpecification,
        *,
        names_of: list[Entity],
    ) -> list[Candidate]:
        out: list[Candidate] = []
        async with self.uow_factory() as uow:
            chunks = await uow.documents.get_chunks(ctx.tenant_id, chunk_ids)
            keys_cache: dict[str, list[str]] = {}
            for c in chunks:
                if c.document_id not in keys_cache:
                    keys_cache[c.document_id] = await uow.documents.visibility_keys(
                        ctx.tenant_id, c.document_id
                    )
                if not visibility.allows(c.tenant_id, keys_cache[c.document_id]):
                    continue
                out.append(
                    Candidate(
                        record_id=c.chunk_id,
                        kind="chunk",
                        text=c.text,
                        score=0.4,
                        retrievers=["graph"],
                        payload={
                            "document_id": c.document_id,
                            "page": c.page,
                            "section_path": c.section_path,
                            "node_id": c.node_id,
                            "text": c.text,
                        },
                        expanded_from="graph",
                        expansion_edge="GRAPH_EVIDENCE",
                    )
                )
        return out


def as_of_from_query(query: str) -> datetime | None:
    return parse_date(query)
