"""ExpansionStage: context preservation over the Document Context Graph.

For the best-ranked chunks it follows structural edges and pulls in the text a reader would
need next to them: the *definition* of a term used (DEFINED_BY), the *footnote* a sentence
refers to (FOOTNOTE), the section a sentence points at (CROSS_REFERENCE), the *parent*
section's summary (PARENT), and the immediate *neighbours* (previous/next chunk). Every
expansion is marked (`expanded_from`, `expansion_edge`) so the bundle and the evidence report
can show why a passage is there, and the whole stage is bounded by
``retrieval.expansion_budget_items``.

Expansions never leave the source document, so they inherit its visibility; the source chunk
was already filtered by the store.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from memory_service.config.settings import RetrievalSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.documents import Chunk, ContextEdge
from memory_service.domain.enums import ContextGraphEdge, QueryType
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.retrieval.engine import Candidate
from memory_service.modules.retrieval.router import RoutedQuery
from memory_service.observability.metrics import stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.uow import UnitOfWorkFactory

# edge -> (score for the expansion, priority: lower first)
_EDGE_PRIORITY: dict[ContextGraphEdge, tuple[float, int]] = {
    ContextGraphEdge.DEFINED_BY: (0.55, 0),
    ContextGraphEdge.FOOTNOTE: (0.55, 1),
    ContextGraphEdge.CROSS_REFERENCE: (0.5, 2),
    ContextGraphEdge.PARENT: (0.35, 3),
    ContextGraphEdge.PREVIOUS: (0.3, 4),
    ContextGraphEdge.NEXT: (0.3, 4),
}


def chunk_candidate(c: Chunk, *, score: float, edge: str, source: str) -> Candidate:
    return Candidate(
        record_id=c.chunk_id,
        kind="chunk",
        text=c.text,
        score=score,
        retrievers=["expansion"],
        payload={
            "document_id": c.document_id,
            "page": c.page,
            "section_path": c.section_path,
            "node_id": c.node_id,
            "text": c.text,
            "text_hash": c.text_hash,
        },
        expanded_from=source,
        expansion_edge=edge,
    )


class ExpansionStage:
    name = "expansion"

    def __init__(
        self, uow_factory: UnitOfWorkFactory, *, settings: RetrievalSettings, seeds: int = 4
    ) -> None:
        self.uow_factory = uow_factory
        self.cfg = settings
        self.seeds = seeds

    def _kinds(self) -> list[ContextGraphEdge]:
        kinds = [ContextGraphEdge.FOOTNOTE, ContextGraphEdge.CROSS_REFERENCE]
        if self.cfg.definition_expansion:
            kinds.insert(0, ContextGraphEdge.DEFINED_BY)
        if self.cfg.parent_expansion:
            kinds.append(ContextGraphEdge.PARENT)
        if self.cfg.neighbor_expansion:
            kinds.extend([ContextGraphEdge.PREVIOUS, ContextGraphEdge.NEXT])
        return kinds

    async def __call__(
        self,
        ctx: MemoryExecutionContext,
        routed: RoutedQuery,
        candidates: list[Candidate],
        visibility: VisibilitySpecification,
        diagnostics: dict[str, Any],
    ) -> list[Candidate]:
        if routed.query_type is QueryType.EXACT_IDENTIFIER:
            return candidates  # an id lookup returns exactly what was asked for
        seeds = [c for c in candidates if c.kind == "chunk" and c.expansion_edge is None][
            : self.seeds
        ]
        if not seeds or self.cfg.expansion_budget_items <= 0:
            return candidates
        with span("retrieval.expansion"), stage_seconds.labels("retrieval.expansion").time():
            added = await self.expand(
                ctx, seeds, candidates, budget=self.cfg.expansion_budget_items
            )
        if added:
            first_fact = next(
                (i for i, c in enumerate(candidates) if c.kind in ("fact", "summary")),
                len(candidates),
            )
            candidates[first_fact:first_fact] = added
            diagnostics["expansion"] = {
                "added": len(added),
                "edges": sorted({c.expansion_edge or "" for c in added}),
            }
        return candidates

    async def expand(
        self,
        ctx: MemoryExecutionContext,
        seeds: Sequence[Candidate],
        existing: Sequence[Candidate],
        *,
        budget: int,
        kinds: Sequence[ContextGraphEdge] | None = None,
    ) -> list[Candidate]:
        present = {c.record_id for c in existing}
        present_nodes = {c.payload.get("node_id") for c in existing if c.kind == "chunk"}
        node_ids = [str(c.payload["node_id"]) for c in seeds if c.payload.get("node_id")]
        if not node_ids:
            return []
        wanted = list(kinds) if kinds is not None else self._kinds()
        added: list[Candidate] = []
        async with self.uow_factory() as uow:
            edges = await uow.documents.edges_from(ctx.tenant_id, node_ids, kinds=wanted)
            # neighbours: chunk-level PREVIOUS/NEXT inside the seed's own node come first
            seed_chunks = await uow.documents.get_chunks(
                ctx.tenant_id, [c.record_id for c in seeds]
            )
            ordered: list[
                tuple[int, float, str, str, str]
            ] = []  # (prio, score, edge, src, target node)
            for e in edges:
                score, prio = _EDGE_PRIORITY.get(e.edge, (0.3, 9))
                ordered.append((prio, score, e.edge.value, e.source_id, e.target_id))
            ordered.sort(key=lambda t: (t[0], -t[1]))
            targets = [t[4] for t in ordered]
            chunks_by_node: dict[str, list[Chunk]] = {}
            for c in await uow.documents.chunks_for_nodes(ctx.tenant_id, targets):
                chunks_by_node.setdefault(c.node_id, []).append(c)
            summaries = await uow.documents.node_summaries(
                ctx.tenant_id, [t[4] for t in ordered if t[2] == "PARENT"]
            )
            # same-node neighbours by ordinal
            own = await uow.documents.chunks_for_nodes(
                ctx.tenant_id, [c.node_id for c in seed_chunks]
            )
        by_node_own: dict[str, list[Chunk]] = {}
        for c in own:
            by_node_own.setdefault(c.node_id, []).append(c)
        if self.cfg.neighbor_expansion and (kinds is None or ContextGraphEdge.NEXT in kinds):
            for sc in seed_chunks:
                siblings = by_node_own.get(sc.node_id, [])
                idx = next((i for i, c in enumerate(siblings) if c.chunk_id == sc.chunk_id), -1)
                for j, edge in ((idx - 1, "PREVIOUS"), (idx + 1, "NEXT")):
                    if 0 <= j < len(siblings) and siblings[j].chunk_id not in present:
                        if len(added) >= budget:
                            break
                        cand = chunk_candidate(
                            siblings[j], score=0.3, edge=edge, source=sc.chunk_id
                        )
                        added.append(cand)
                        present.add(cand.record_id)
        for _prio, score, edge, source, target in ordered:
            if len(added) >= budget:
                break
            source_chunk = next(
                (c.record_id for c in seeds if c.payload.get("node_id") == source), source
            )
            if edge == "PARENT" and target in summaries and f"sum_{target}" not in present:
                added.append(
                    Candidate(
                        record_id=f"sum_{target}",
                        kind="summary",
                        text=summaries[target],
                        score=score,
                        retrievers=["expansion"],
                        payload={
                            "node_id": target,
                            "document_id": seeds[0].payload.get("document_id"),
                        },
                        expanded_from=source_chunk,
                        expansion_edge=edge,
                    )
                )
                present.add(f"sum_{target}")
                continue
            if target in present_nodes:
                continue
            for c in chunks_by_node.get(target, [])[:1]:
                if c.chunk_id in present:
                    continue
                added.append(chunk_candidate(c, score=score, edge=edge, source=source_chunk))
                present.add(c.chunk_id)
                present_nodes.add(target)
        return added[:budget]


def edges_to_groups(edges: Sequence[ContextEdge]) -> dict[str, str]:
    """Group name -> target node for the edges that define *required* companions."""
    out: dict[str, str] = {}
    for e in edges:
        if e.edge in (
            ContextGraphEdge.DEFINED_BY,
            ContextGraphEdge.FOOTNOTE,
            ContextGraphEdge.CROSS_REFERENCE,
        ):
            out[f"{e.edge.value.lower()}:{e.label or e.target_id}"] = e.target_id
    return out
