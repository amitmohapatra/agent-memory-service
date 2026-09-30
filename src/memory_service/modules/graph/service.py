"""GraphService: enrich the graph from memories and documents; answer entity queries under
the caller's visibility."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from memory_service.config.constants import GraphSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.errors import NotFound
from memory_service.domain.graph import GraphLayer
from memory_service.domain.memory import CanonicalMemory, Scope, unverified_representation
from memory_service.domain.revisions import RevisionKind
from memory_service.domain.text import unicode_tokens
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.graph.native import VALUE_TYPES, NativeGraphEnrichment
from memory_service.modules.graph.summaries import EntitySummaries
from memory_service.modules.ingestion.context_graph import canonical_entity, extract_entities
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.llm.policy import document_identity
from memory_service.modules.memory.native import _STOP as _STOP_WORDS
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.credentials import ModelIdentity
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


@dataclass
class EntityProfile:
    """One entity as a reader sees it: its current value per predicate (the newest current
    relation of each, with the entity as subject), every visible current relation, the
    history of facts that stopped holding, and the names of the entities they point at."""

    entity: Entity
    current: list[Relation]
    relations: list[Relation]
    history: list[Relation]
    names: dict[str, str]


def query_terms(query: str, *, max_terms: int = 12) -> list[str]:
    """Canonical candidate entity names in a question: extracted entities, plus lowercased
    uni/bi/tri-grams so 'adjusted ebitda' matches even when not capitalised."""
    if max_terms <= 0:
        return []
    names: dict[str, None] = {}
    for name in _query_names(query):
        names.setdefault(name, None)
        if len(names) >= max_terms * 3:
            break
    return list(names)


def _query_names(query: str) -> Iterator[str]:
    yield from (canonical_entity(e) for e in extract_entities(query))
    raw = (
        _WORD.findall(query)
        if query.isascii()
        else unicode_tokens(query, _WORD.findall, max_ngram=4)
    )
    lowered = [w.casefold() for w in raw if w.casefold() not in _IGNORE]
    # A Latin name inside CJK text has no Unicode word boundary (e.g. 誰がAcmeを).
    # Preserve whole script-separated words before bounded CJK n-grams exhaust the cap.
    if not query.isascii():
        yield from (word for word in lowered if word.isascii() and len(word) >= 3)
    # Direct script terms cannot be recovered by adding spaces between CJK characters.
    # Keep them before the ordinary word n-grams; never scan the whole entity table.
    yield from (word for word in lowered if not word.isascii() and len(word) >= 2)
    for n in (3, 2, 1):
        for i in range(len(lowered) - n + 1):
            gram = " ".join(lowered[i : i + n])
            if len(gram) >= 3:
                yield gram


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
        self.summaries = EntitySummaries(store, self.assist, settings)

    # -- enrichment (called from index jobs) ------------------------------------------
    async def enrich_memories(self, tenant_id: str, memory_ids: Sequence[str]) -> int:
        async with self.uow_factory() as uow:
            memories = await uow.memories.get_many(tenant_id, memory_ids)
        now = datetime.now(UTC)
        found = {m.memory_id for m in memories}
        n = 0
        removed = 0
        #: entities to re-summarise, per owner whose model identity may pay for it
        touched: dict[tuple[str, str | None], list[str]] = {}
        with (
            span("graph.enrich_memories", tenant_id=tenant_id),
            stage_seconds.labels("graph.enrich").time(),
        ):
            for mid in memory_ids:
                if mid not in found:
                    removed += await self.store.supersede_for_memory(tenant_id, mid, at=now)
            for m in memories:
                if m.temporal.status.value != "CURRENT" or unverified_representation(
                    m.system_metadata
                ):
                    removed += await self.store.supersede_for_memory(tenant_id, m.memory_id, at=now)
                    continue
                n += await self._enrich_memory(m, touched)
            for (principal, workspace_id), entity_ids in touched.items():
                async with self.assist.bound(ModelIdentity(tenant_id, principal, workspace_id)):
                    await self.summaries.refresh(tenant_id, entity_ids)
        if n or removed:
            async with self.uow_factory() as uow:
                await uow.revisions.bump(tenant_id, RevisionKind.GRAPH, "")
                await uow.commit()
        return n

    async def _enrich_memory(
        self, m: CanonicalMemory, touched: dict[tuple[str, str | None], list[str]]
    ) -> int:
        """One memory's entities and relations; records the entities it touched per owner."""
        ctx = MemoryExecutionContext(tenant_id=m.tenant_id, user_id=m.scope.user_id)
        owner = ModelIdentity(m.tenant_id, m.owner_principal, m.scope.workspace_id)
        async with self.assist.bound(owner):
            entities, relations = await self.provider.enrich_memory(m, ctx)
        await self.store.upsert_entities(entities)
        await self.store.upsert_relations(relations)
        if relations:
            owner = (m.owner_principal, m.scope.workspace_id)
            touched.setdefault(owner, []).extend(e.entity_id for e in entities)
        return len(relations)

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
            else Scope(level=ScopeLevel.TENANT, tenant_id=tenant_id)
        )
        ctx = MemoryExecutionContext(tenant_id=tenant_id, user_id=document.owner_user_id)
        async with self.assist.bound(document_identity(document)):
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
                busiest = sorted(entities, key=lambda e: (-e.mention_count, e.entity_id))
                await self.summaries.refresh(tenant_id, [e.entity_id for e in busiest])
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
            for e in await self.store.search_entities(
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
        valid_at: datetime | None = None,
        layers: Sequence[GraphLayer] | None = None,
        visibility: VisibilitySpecification | None = None,
        max_visited: int | None = None,
        budgeted: bool = False,
    ) -> GraphAnswer:
        """``budgeted`` is the retrieval-time traversal: names resolve lexically only (a model
        call has no place under the graph budget) and the store stops the walk past it
        (``GraphBudgetExceededError``)."""
        if visibility is None:
            visibility = await self._visibility(ctx)
        names = list(entities) + (query_terms(query) if query else [])
        matched = (
            await self.store.find_entities(
                ctx.tenant_id,
                [canonical_entity(n) for n in names if n.strip()],
                scope_keys=sorted(visibility.keys),
            )
            if budgeted
            else await self.resolve(ctx, names, visibility)
        )
        if not matched:
            return GraphAnswer(entities=[], relations=[], matched=[], visited=0)
        # prefer the most specific matches: longer canonical names first, bounded
        matched.sort(key=lambda e: (-len(e.canonical_name), e.canonical_name, e.entity_id))
        seeds = [e.entity_id for e in matched[:8]]
        hood: GraphNeighborhood = await self.store.neighborhood(
            ctx.tenant_id,
            seeds,
            scope_keys=sorted(visibility.keys),
            hops=hops or self.cfg.default_hops,
            max_visited=max_visited or self.cfg.max_visited,
            as_of=as_of,
            valid_at=valid_at,
            layers=layers,
            budgeted=budgeted,
        )
        return GraphAnswer(
            entities=hood.entities,
            relations=hood.relations,
            matched=matched[:8],
            visited=hood.visited,
        )

    async def search_entities(
        self,
        ctx: MemoryExecutionContext,
        *,
        query: str | None = None,
        entity_type: str | None = None,
        limit: int | None = None,
    ) -> list[Entity]:
        """Visible entities whose canonical name starts with ``query``, most mentioned first."""
        visibility = await self._visibility(ctx)
        return await self.store.search_entities(
            ctx.tenant_id,
            scope_keys=sorted(visibility.keys),
            prefix=canonical_entity(query) if query and query.strip() else None,
            entity_type=entity_type,
            limit=min(limit or self.cfg.entity_search_max, self.cfg.entity_search_max),
        )

    async def profile(self, ctx: MemoryExecutionContext, entity_id: str) -> EntityProfile:
        """The entity's profile, bounded; ``NotFound`` when it is absent or not visible."""
        visibility = await self._visibility(ctx)
        scope_keys = sorted(visibility.keys)
        found = await self.store.get_entities(ctx.tenant_id, [entity_id], scope_keys=scope_keys)
        if not found:
            raise NotFound(f"entity {entity_id} not found")
        relations = await self.store.entity_relations(
            ctx.tenant_id,
            entity_id,
            scope_keys=scope_keys,
            current=True,
            limit=self.cfg.profile_relations_max,
        )
        history = await self.store.entity_relations(
            ctx.tenant_id,
            entity_id,
            scope_keys=scope_keys,
            current=False,
            limit=self.cfg.profile_history_max,
        )
        # newest first, so the first relation seen per predicate is its current value
        current: dict[str, Relation] = {}
        for relation in relations:
            if relation.subject_id == entity_id and relation.layer != "structural":
                current.setdefault(relation.predicate, relation)
        others = {r.subject_id for r in (*relations, *history)} | {
            r.object_id for r in (*relations, *history)
        }
        visible = await self.store.get_entities(
            ctx.tenant_id, sorted(others), scope_keys=scope_keys
        )
        return EntityProfile(
            entity=found[0],
            current=[current[p] for p in sorted(current)],
            relations=relations,
            history=history,
            names={e.entity_id: e.name for e in visible},
        )

    async def _visibility(self, ctx: MemoryExecutionContext) -> VisibilitySpecification:
        async with self.uow_factory() as uow:
            return await self.authz.visibility(ctx, revisions=uow.revisions)
