"""In-memory GraphStore with the same semantics as the PostgreSQL store (tests, dev)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from memory_service.domain.graph import INVALIDATED_BY, GraphLayer
from memory_service.modules.graph.invalidation import invalidation_edge, passes_time
from memory_service.ports.intelligence import Entity, EntityAlias, GraphNeighborhood, Relation


class MemoryGraphStore:
    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.relations: dict[str, Relation] = {}
        self.aliases: dict[tuple[str, str, str], EntityAlias] = {}

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
                        "layer": r.layer,
                        "valid_to": r.valid_to,
                        "invalidated_at": r.invalidated_at,
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

    async def list_entities(
        self, tenant_id: str, *, scope_keys: Sequence[str], limit: int = 200
    ) -> list[Entity]:
        if not scope_keys or limit <= 0:
            return []
        visible = [
            e
            for e in self.entities.values()
            if e.tenant_id == tenant_id and self._visible(e.visibility_keys, scope_keys)
        ]
        visible.sort(key=lambda e: (-e.mention_count, e.canonical_name))
        return visible[:limit]

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
        valid_at: datetime | None = None,
        layers: Sequence[GraphLayer] | None = None,
    ) -> GraphNeighborhood:
        visited: dict[str, None] = dict.fromkeys(entity_ids)
        frontier = list(entity_ids)
        found: dict[str, Relation] = {}
        wanted_layers = set(layers) if layers else None
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
                if wanted_layers is not None and r.layer not in wanted_layers:
                    continue
                if not passes_time(r, as_of=as_of, valid_at=valid_at):
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
                update={
                    "status": "SUPERSEDED",
                    "superseded_by": by,
                    "valid_to": at,
                    "invalidated_at": at,
                }
            )

    async def supersede_for_memory(self, tenant_id: str, memory_id: str, *, at: datetime) -> int:
        n = 0
        for rid, r in list(self.relations.items()):
            if r.tenant_id == tenant_id and r.memory_id == memory_id and r.status == "CURRENT":
                self.relations[rid] = r.model_copy(
                    update={"status": "SUPERSEDED", "valid_to": at, "invalidated_at": at}
                )
                n += 1
        return n

    async def invalidate(
        self,
        relation_id: str,
        *,
        reason: str,
        at: datetime,
        by: str | None = None,
        status: str = "INVALIDATED",
        attributes: dict[str, Any] | None = None,
    ) -> Relation | None:
        r = self.relations.get(relation_id)
        if r is None:
            return None
        update: dict[str, Any] = {
            "status": status,
            "invalidated_at": r.invalidated_at or at,
            "attributes": {
                **r.attributes,
                "invalidation": {"reason": reason, "at": at.isoformat(), "by": by},
            },
        }
        if status == "SUPERSEDED":
            update["valid_to"] = r.valid_to or at
            update["superseded_by"] = by or r.superseded_by
        self.relations[relation_id] = r.model_copy(update=update)
        winner = self.relations.get(by) if by else None
        if winner is None:
            return None
        edge = invalidation_edge(
            self.relations[relation_id], winner, reason=reason, at=at, attributes=attributes
        )
        self.relations[edge.relation_id] = edge
        return edge

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
        self,
        tenant_id: str,
        document_id: str,
        *,
        scope_keys: Sequence[str],
        include_invalidated: bool = False,
    ) -> list[Relation]:
        return [
            r
            for r in self.relations.values()
            if r.tenant_id == tenant_id
            and r.document_id == document_id
            and (include_invalidated or r.status != "INVALIDATED")
            and self._visible(r.visibility_keys, scope_keys)
        ]

    async def count(self, tenant_id: str) -> tuple[int, int]:
        return (
            sum(1 for e in self.entities.values() if e.tenant_id == tenant_id),
            sum(
                1
                for r in self.relations.values()
                if r.tenant_id == tenant_id and r.predicate != INVALIDATED_BY
            ),
        )

    async def ping(self) -> bool:
        return True
