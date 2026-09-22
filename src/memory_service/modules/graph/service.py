"""GraphService: enrich the graph from memories and documents; answer entity queries under
the caller's visibility."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from memory_service.config.constants import GraphSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.memory import Scope
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.graph.native import VALUE_TYPES, NativeGraphEnrichment
from memory_service.modules.ingestion.context_graph import canonical_entity, extract_entities
from memory_service.modules.llm.assist import LLMAssist
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
LLM_MAX_CANDIDATES = 200
LLM_MAX_QUERY_NAMES = 12
_RESOLUTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "query_name": {"type": "string"},
                    "entity_name": {"type": "string"},
                },
                "required": ["query_name", "entity_name"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["matches"],
    "additionalProperties": False,
}
_RESOLUTION_SYSTEM = (
    "You resolve names in a question to entities of a knowledge graph. For each query name "
    "that clearly refers to one of the candidate entities (a synonym, abbreviation, spelling "
    "or wording variant), return the candidate name exactly as listed. Skip query names that "
    "refer to nothing in the list; never invent entities."
)


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
        assist: LLMAssist | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.store = store
        self.provider = provider or NativeGraphEnrichment()
        self.authz = authz
        self.cfg = settings
        self.assist = assist or LLMAssist.disabled()

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
        scope_keys = sorted(visibility.keys)
        found = await self.store.find_entities(ctx.tenant_id, canon, scope_keys=scope_keys)
        if not canon or not self.assist.wants("entity_resolution"):
            return found
        known = {e.canonical_name for e in found} | {
            canonical_entity(a) for e in found for a in e.aliases
        }
        unmatched = [
            n for n in dict.fromkeys(canon) if not any(k and f" {k} " in f" {n} " for k in known)
        ][:LLM_MAX_QUERY_NAMES]
        if not unmatched:
            return found
        seen = {e.entity_id for e in found}
        extra = await self._resolve_with_model(ctx.tenant_id, unmatched, scope_keys)
        return found + [e for e in extra if e.entity_id not in seen]

    async def _resolve_with_model(
        self, tenant_id: str, names: Sequence[str], scope_keys: Sequence[str]
    ) -> list[Entity]:
        candidates = [
            e
            for e in await self.store.list_entities(
                tenant_id, scope_keys=scope_keys, limit=LLM_MAX_CANDIDATES
            )
            if e.entity_type not in VALUE_TYPES
        ]
        if not candidates:
            return []
        by_form: dict[str, Entity] = {}
        for e in candidates:
            for form in (
                e.canonical_name,
                canonical_entity(e.name),
                *map(canonical_entity, e.aliases),
            ):
                if form:
                    by_form.setdefault(form, e)
        lines = []
        for e in candidates:
            aka = [a for a in e.aliases if canonical_entity(a) != canonical_entity(e.name)][:3]
            lines.append(f"- {e.name[:80]}" + (f" (aka {', '.join(aka)})" if aka else ""))
        out = await self.assist.structured(
            "entity_resolution",
            system=_RESOLUTION_SYSTEM,
            user=f"Query names: {'; '.join(names)}\nCandidates:\n" + "\n".join(lines),
            schema=_RESOLUTION_SCHEMA,
            max_tokens=400,
        )
        asked = set(names)
        accepted: dict[str, None] = {}
        for m in (out or {}).get("matches", []):
            if not isinstance(m, dict):
                continue
            e = by_form.get(canonical_entity(str(m.get("entity_name", ""))))
            if e is not None and canonical_entity(str(m.get("query_name", ""))) in asked:
                accepted[e.canonical_name] = None
        if not accepted:
            return []
        return await self.store.find_entities(tenant_id, list(accepted), scope_keys=scope_keys)

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
