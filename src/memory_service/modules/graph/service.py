"""GraphService: enrich the graph from memories and documents; answer entity queries under
the caller's visibility."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from memory_service.config.settings import GraphSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.memory import Scope
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.graph.native import NativeGraphEnrichment
from memory_service.modules.ingestion.context_graph import canonical_entity, extract_entities
from memory_service.modules.memory.native import _STOP as _STOP_WORDS
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.intelligence import Entity, GraphNeighborhood, Relation
from memory_service.ports.uow import UnitOfWorkFactory

log = get_logger(__name__)

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9&/\-]*")
_QUESTION_TEXT = """
why what how when where who whom which did does do is are was were tell show explain
give list describe compare because despite
"""
_QUESTION_WORDS = frozenset(_QUESTION_TEXT.split())
_IGNORE = _STOP_WORDS | _QUESTION_WORDS


@dataclass
class GraphAnswer:
    entities: list[Entity]
    relations: list[Relation]
    matched: list[Entity] = field(default_factory=list)
    visited: int = 0


def query_terms(query: str, *, max_terms: int = 12) -> list[str]:
    """Canonical candidate entity names in a question: extracted entities, plus lowercased
    uni/bi/tri-grams so 'adjusted ebitda' matches even when not capitalised."""
    names: list[str] = [canonical_entity(e) for e in extract_entities(query)]
    words = [w for w in _WORD.findall(query) if w.casefold() not in _IGNORE]
    lowered = [w.casefold() for w in words]
    for n in (3, 2, 1):
        for i in range(len(lowered) - n + 1):
            gram = " ".join(lowered[i : i + n])
            if len(gram) >= 3 and gram not in names:
                names.append(gram)
    return names[: max_terms * 3]


class GraphService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        store: Any,
        provider: Any,
        authz: AuthorizationService,
        *,
        settings: GraphSettings,
    ) -> None:
        self.uow_factory = uow_factory
        self.store = store
        self.provider = provider or NativeGraphEnrichment()
        self.authz = authz
        self.cfg = settings

    # -- enrichment (called from index jobs) ------------------------------------------
    async def enrich_memories(self, tenant_id: str, memory_ids: Sequence[str]) -> int:
        async with self.uow_factory() as uow:
            memories = await uow.memories.get_many(tenant_id, memory_ids)
        now = datetime.now(UTC)
        found = {m.memory_id for m in memories}
        n = 0
        with (
            span("graph.enrich_memories", tenant_id=tenant_id),
            stage_seconds.labels("graph.enrich").time(),
        ):
            for mid in memory_ids:
                if mid not in found:
                    await self.store.supersede_for_memory(tenant_id, mid, at=now)
            for m in memories:
                if m.temporal.status.value != "CURRENT":
                    await self.store.supersede_for_memory(tenant_id, m.memory_id, at=now)
                    continue
                ctx = MemoryExecutionContext(tenant_id=tenant_id, user_id=m.scope.user_id)
                entities, relations = await self.provider.enrich_memory(m, ctx)
                await self.store.upsert_entities(entities)
                await self.store.upsert_relations(relations)
                n += len(relations)
        if n:
            async with self.uow_factory() as uow:
                await uow.revisions.bump(tenant_id, RevisionKind.GRAPH, "")
                await uow.commit()
        return n

    async def enrich_document(self, tenant_id: str, document_id: str) -> int:
        async with self.uow_factory() as uow:
            document = await uow.documents.get(tenant_id, document_id)
            if document is None or not document.current_version_id:
                return 0
            version = await uow.documents.get_version(tenant_id, document.current_version_id)
            if version is None:
                return 0
            nodes = await uow.documents.list_nodes(tenant_id, document_id)
            chunks = await uow.documents.list_chunks(tenant_id, document_id)
            keys = await uow.documents.visibility_keys(tenant_id, document_id)
        from memory_service.domain.enums import ScopeLevel

        scope = (
            Scope(level=ScopeLevel.THREAD, tenant_id=tenant_id, thread_id=document.thread_id)
            if document.thread_id
            else Scope(
                level=ScopeLevel.WORKSPACE, tenant_id=tenant_id, workspace_id=document.workspace_id
            )
            if document.workspace_id
            else Scope(level=ScopeLevel.TENANT, tenant_id=tenant_id)
        )
        ctx = MemoryExecutionContext(tenant_id=tenant_id, user_id=document.owner_user_id)
        with (
            span("graph.enrich_document", tenant_id=tenant_id),
            stage_seconds.labels("graph.enrich").time(),
        ):
            await self.store.delete_for_document(tenant_id, document_id)
            entities, relations = await self.provider.enrich_document(
                version,
                nodes,
                chunks,
                ctx,
                visibility_keys=keys,
                document_title=document.title,
                scope_key=scope.key(),
            )
            await self.store.upsert_entities(entities)
            await self.store.upsert_relations(relations)
        async with self.uow_factory() as uow:
            await uow.revisions.bump(tenant_id, RevisionKind.GRAPH, "")
            await uow.commit()
        log.info(
            "graph.document_enriched",
            tenant_id=tenant_id,
            document_id=document_id,
            entities=len(entities),
            relations=len(relations),
        )
        return len(relations)

    # -- queries ----------------------------------------------------------------------
    async def resolve(
        self, ctx: MemoryExecutionContext, names: Sequence[str], visibility: VisibilitySpecification
    ) -> list[Entity]:
        canon = [canonical_entity(n) for n in names if n.strip()]
        return await self.store.find_entities(
            ctx.tenant_id, canon, scope_keys=sorted(visibility.keys)
        )

    async def query(
        self,
        ctx: MemoryExecutionContext,
        *,
        query: str | None = None,
        entities: Sequence[str] = (),
        hops: int | None = None,
        as_of: datetime | None = None,
        visibility: VisibilitySpecification | None = None,
        max_visited: int | None = None,
    ) -> GraphAnswer:
        if visibility is None:
            async with self.uow_factory() as uow:
                visibility = await self.authz.visibility(ctx, revisions=uow.revisions)
        names = list(entities) + (query_terms(query) if query else [])
        matched = await self.resolve(ctx, names, visibility)
        if not matched:
            return GraphAnswer(entities=[], relations=[], matched=[], visited=0)
        # prefer the most specific matches: longer canonical names first, bounded
        matched.sort(key=lambda e: (-len(e.canonical_name), e.canonical_name))
        seeds = [e.entity_id for e in matched[:8]]
        hood: GraphNeighborhood = await self.store.neighborhood(
            ctx.tenant_id,
            seeds,
            scope_keys=sorted(visibility.keys),
            hops=hops or self.cfg.default_hops,
            max_visited=max_visited or self.cfg.max_visited,
            as_of=as_of,
        )
        return GraphAnswer(
            entities=hood.entities,
            relations=hood.relations,
            matched=matched[:8],
            visited=hood.visited,
        )
