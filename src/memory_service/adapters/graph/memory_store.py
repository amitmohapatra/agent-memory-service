"""In-memory GraphStore with the same semantics as the PostgreSQL store (tests, dev)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from memory_service.ports.intelligence import Entity, GraphNeighborhood, Relation


class MemoryGraphStore:
    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.relations: dict[str, Relation] = {}

    @staticmethod
    def _visible(keys: Sequence[str], audience: Sequence[str]) -> bool:
        return bool(set(keys) & set(audience))

    async def upsert_entities(self, entities: Sequence[Entity]) -> None:
        for e in entities:
            prev = self.entities.get(e.entity_id)
            if prev is None:
                self.entities[e.entity_id] = e.model_copy()
            else:
                self.entities[e.entity_id] = prev.model_copy(
                    update={
                        "mention_count": prev.mention_count + 1,
                        "aliases": sorted(set(prev.aliases) | set(e.aliases)),
                        "visibility_keys": sorted(
                            set(prev.visibility_keys) | set(e.visibility_keys)
                        ),
                        "revision": prev.revision + 1,
                    }
                )

    async def upsert_relations(self, relations: Sequence[Relation]) -> None:
        for r in relations:
            prev = self.relations.get(r.relation_id)
            if prev is None:
                self.relations[r.relation_id] = r.model_copy()
            else:
                self.relations[r.relation_id] = prev.model_copy(
                    update={
                        "confidence": max(prev.confidence, r.confidence),
                        "status": r.status,
                        "valid_to": r.valid_to,
                        "superseded_by": r.superseded_by,
                        "visibility_keys": list(r.visibility_keys),
                        "fact_text": r.fact_text,
                    }
                )

    async def find_entities(
        self, tenant_id: str, names: Sequence[str], *, scope_keys: Sequence[str]
    ) -> list[Entity]:
        wanted = set(names)
        return [
            e
            for e in self.entities.values()
            if e.tenant_id == tenant_id
            and (e.canonical_name in wanted or wanted.intersection(e.aliases))
            and self._visible(e.visibility_keys, scope_keys)
        ]

    async def get_entities(
        self, tenant_id: str, entity_ids: Sequence[str], *, scope_keys: Sequence[str]
    ) -> list[Entity]:
        return [
            e
            for i in entity_ids
            if (e := self.entities.get(i)) is not None
            and e.tenant_id == tenant_id
            and self._visible(e.visibility_keys, scope_keys)
        ]

    async def neighborhood(
        self,
        tenant_id: str,
        entity_ids: Sequence[str],
        *,
        scope_keys: Sequence[str],
        hops: int = 1,
        max_visited: int = 200,
        as_of: datetime | None = None,
    ) -> GraphNeighborhood:
        visited: dict[str, None] = dict.fromkeys(entity_ids)
        frontier = list(entity_ids)
        found: dict[str, Relation] = {}
        for _ in range(max(0, hops)):
            if not frontier or len(visited) >= max_visited:
                break
            fs = set(frontier)
            next_frontier: list[str] = []
            for r in sorted(self.relations.values(), key=lambda x: -x.confidence):
                if r.tenant_id != tenant_id or not self._visible(r.visibility_keys, scope_keys):
                    continue
                if r.subject_id not in fs and r.object_id not in fs:
                    continue
                if as_of is None:
                    if r.status != "CURRENT":
                        continue
                elif (
                    r.status == "RETRACTED"
                    or (r.valid_from and r.valid_from > as_of)
                    or (r.valid_to and r.valid_to <= as_of)
                ):
                    continue
                found.setdefault(r.relation_id, r)
                for eid in (r.subject_id, r.object_id):
                    if eid not in visited and len(visited) < max_visited:
                        visited[eid] = None
                        next_frontier.append(eid)
            frontier = next_frontier
        ents = await self.get_entities(tenant_id, list(visited), scope_keys=scope_keys)
        allowed = {e.entity_id for e in ents}
        rels = [r for r in found.values() if r.subject_id in allowed and r.object_id in allowed]
        return GraphNeighborhood(entities=ents, relations=rels, visited=len(visited))

    async def supersede(self, relation_id: str, *, by: str, at: datetime) -> None:
        r = self.relations.get(relation_id)
        if r is not None:
            self.relations[relation_id] = r.model_copy(
                update={"status": "SUPERSEDED", "superseded_by": by, "valid_to": at}
            )

    async def supersede_for_memory(self, tenant_id: str, memory_id: str, *, at: datetime) -> int:
        n = 0
        for rid, r in list(self.relations.items()):
            if r.tenant_id == tenant_id and r.memory_id == memory_id and r.status == "CURRENT":
                self.relations[rid] = r.model_copy(update={"status": "SUPERSEDED", "valid_to": at})
                n += 1
        return n

    async def delete_for_document(self, tenant_id: str, document_id: str) -> int:
        gone = [
            rid
            for rid, r in self.relations.items()
            if r.tenant_id == tenant_id and r.document_id == document_id
        ]
        for rid in gone:
            del self.relations[rid]
        return len(gone)

    async def relations_for_document(
        self, tenant_id: str, document_id: str, *, scope_keys: Sequence[str]
    ) -> list[Relation]:
        return [
            r
            for r in self.relations.values()
            if r.tenant_id == tenant_id
            and r.document_id == document_id
            and self._visible(r.visibility_keys, scope_keys)
        ]

    async def relations_for_memory(self, tenant_id: str, memory_id: str) -> list[Relation]:
        return [
            r
            for r in self.relations.values()
            if r.tenant_id == tenant_id and r.memory_id == memory_id
        ]

    async def count(self, tenant_id: str) -> tuple[int, int]:
        return (
            sum(1 for e in self.entities.values() if e.tenant_id == tenant_id),
            sum(1 for r in self.relations.values() if r.tenant_id == tenant_id),
        )

    async def ping(self) -> bool:
        return True
