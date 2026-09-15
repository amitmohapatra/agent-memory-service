"""Graphiti (temporal knowledge graph on Neo4j) as an optional GraphEnrichmentProvider.

Graphiti extracts entities/edges with an LLM and stores bitemporal facts in Neo4j. This
adapter feeds it *episodes* (memory content, document chunks) namespaced per tenant via
``group_id`` and reads back the resulting nodes/edges as Entity/Relation objects so they can
be mirrored into the canonical graph store with evidence. It requires ``graphiti-core``, a
reachable Neo4j (``graph_enrichment.graphiti_neo4j_*``) and an LLM; it refuses to start
otherwise. The client is injectable for tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from memory_service.config.settings import Settings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.documents import Chunk, DocumentNode, DocumentVersion
from memory_service.domain.errors import DependencyUnavailable
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.memory import CanonicalMemory
from memory_service.modules.graph.native import make_entity, relation_id_for
from memory_service.ports.intelligence import Entity, Relation
from memory_service.ports.models import ProviderInfo


class GraphitiEnrichment:
    info = ProviderInfo(
        name="graphiti",
        license="Apache-2.0",
        origin="getzep/graphiti",
        locality="local",
        requires_llm=True,
    )

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client

    async def _graphiti(self) -> Any:
        if self._client is None:
            cfg = self.settings.graph_enrichment
            if not (
                cfg.graphiti_neo4j_url and cfg.graphiti_neo4j_user and cfg.graphiti_neo4j_password
            ):
                raise DependencyUnavailable(
                    "graphiti requires graph_enrichment.graphiti_neo4j_* settings"
                )
            try:
                from graphiti_core import Graphiti
            except ImportError as exc:
                raise DependencyUnavailable(
                    "graphiti-core is required (install [graphiti])"
                ) from exc
            self._client = Graphiti(
                cfg.graphiti_neo4j_url,
                cfg.graphiti_neo4j_user,
                cfg.graphiti_neo4j_password.get_secret_value(),
            )
            await self._client.build_indices_and_constraints()
        return self._client

    @staticmethod
    def _group(tenant_id: str, scope_key: str) -> str:
        return f"{tenant_id}::{scope_key}"

    async def _episode(
        self,
        *,
        tenant_id: str,
        scope_key: str,
        name: str,
        body: str,
        keys: Sequence[str],
        evidence: list[EvidenceRef],
        memory_id: str | None = None,
        document_id: str | None = None,
    ) -> tuple[list[Entity], list[Relation]]:
        client = await self._graphiti()
        result = await client.add_episode(
            name=name,
            episode_body=body,
            source_description="memory-service",
            reference_time=datetime.now(UTC),
            group_id=self._group(tenant_id, scope_key),
        )
        entities: dict[str, Entity] = {}
        relations: list[Relation] = []
        by_uuid: dict[str, Entity] = {}
        for node in getattr(result, "nodes", []) or []:
            e = make_entity(
                tenant_id, scope_key, str(node.name), visibility_keys=keys, evidence=evidence
            )
            entities.setdefault(e.entity_id, e)
            by_uuid[str(node.uuid)] = e
        for edge in getattr(result, "edges", []) or []:
            s, o = by_uuid.get(str(edge.source_node_uuid)), by_uuid.get(str(edge.target_node_uuid))
            if s is None or o is None:
                continue
            predicate = str(getattr(edge, "name", "related_to")).lower()
            relations.append(
                Relation(
                    relation_id=relation_id_for(
                        tenant_id, s.entity_id, predicate, o.entity_id, str(edge.uuid)
                    ),
                    tenant_id=tenant_id,
                    subject_id=s.entity_id,
                    predicate=predicate,
                    object_id=o.entity_id,
                    scope_key=scope_key,
                    visibility_keys=list(keys),
                    valid_from=getattr(edge, "valid_at", None),
                    valid_to=getattr(edge, "invalid_at", None),
                    observed_at=datetime.now(UTC),
                    status="CURRENT" if getattr(edge, "invalid_at", None) is None else "SUPERSEDED",
                    confidence=0.7,
                    evidence=evidence,
                    memory_id=memory_id,
                    document_id=document_id,
                    fact_text=str(getattr(edge, "fact", "")),
                    attributes={"graphiti_uuid": str(edge.uuid)},
                )
            )
        return list(entities.values()), relations

    async def enrich_memory(
        self, memory: CanonicalMemory, ctx: MemoryExecutionContext
    ) -> tuple[list[Entity], list[Relation]]:
        return await self._episode(
            tenant_id=memory.tenant_id,
            scope_key=memory.scope.key(),
            name=memory.memory_id,
            body=memory.content,
            keys=list(memory.system_metadata.get("visibility_keys", [])),
            evidence=list(memory.evidence),
            memory_id=memory.memory_id,
        )

    async def enrich_document(
        self,
        version: DocumentVersion,
        nodes: Sequence[DocumentNode],
        chunks: Sequence[Chunk],
        ctx: MemoryExecutionContext,
        *,
        visibility_keys: Sequence[str] = (),
        document_title: str = "",
        scope_key: str = "",
    ) -> tuple[list[Entity], list[Relation]]:
        entities: dict[str, Entity] = {}
        relations: list[Relation] = []
        now = datetime.now(UTC)
        for c in chunks:
            ev = [
                EvidenceRef(
                    source_type="document_chunk",
                    source_id=c.chunk_id,
                    document_id=version.document_id,
                    chunk_id=c.chunk_id,
                    node_id=c.node_id,
                    page=c.page,
                    observed_at=now,
                )
            ]
            es, rs = await self._episode(
                tenant_id=version.tenant_id,
                scope_key=scope_key,
                name=c.chunk_id,
                body=c.contextual_text,
                keys=visibility_keys,
                evidence=ev,
                document_id=version.document_id,
            )
            for e in es:
                entities.setdefault(e.entity_id, e)
            relations.extend(rs)
        return list(entities.values()), relations
