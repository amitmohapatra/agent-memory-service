"""Docling Graph as an optional GraphEnrichmentProvider for documents.

``docling-graph`` turns a Docling document into a typed knowledge graph (entities + relations
with page provenance) using an LLM-backed extractor. This adapter converts its output into
Entity/Relation objects with chunk/page evidence; memories fall back to the native provider.
Requires ``docling-graph`` and an LLM. The extractor is injectable for tests.
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
from memory_service.modules.graph.native import NativeGraphEnrichment, make_entity, relation_id_for
from memory_service.ports.intelligence import Entity, Relation
from memory_service.ports.models import ProviderInfo


class DoclingGraphEnrichment:
    info = ProviderInfo(
        name="docling-graph",
        license="MIT",
        origin="docling-project/docling-graph",
        locality="local",
        requires_llm=True,
    )

    def __init__(self, settings: Settings, extractor: Any | None = None) -> None:
        self.settings = settings
        self._extractor = extractor
        self._native = NativeGraphEnrichment()

    def _get_extractor(self) -> Any:
        if self._extractor is None:
            try:
                import docling_graph  # noqa: F401
            except ImportError as exc:
                raise DependencyUnavailable(
                    "docling-graph is required (pip install docling-graph)"
                ) from exc
            raise DependencyUnavailable(
                "docling-graph extractor must be configured explicitly (inject an extractor)"
            )
        return self._extractor

    async def enrich_memory(
        self, memory: CanonicalMemory, ctx: MemoryExecutionContext
    ) -> tuple[list[Entity], list[Relation]]:
        return await self._native.enrich_memory(memory, ctx)

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
        extractor = self._get_extractor()
        # extractor(chunks) -> iterable of dicts with keys
        #   subject, predicate, object, chunk_id, page, text, confidence
        triples = await extractor(chunks)
        keys = list(visibility_keys)
        tenant = version.tenant_id
        now = datetime.now(UTC)
        entities: dict[str, Entity] = {}
        relations: list[Relation] = []
        for t in triples:
            ev = [
                EvidenceRef(
                    source_type="document_chunk",
                    source_id=str(t.get("chunk_id") or version.document_id),
                    document_id=version.document_id,
                    chunk_id=t.get("chunk_id"),
                    page=t.get("page"),
                    observed_at=now,
                )
            ]
            s = make_entity(tenant, scope_key, str(t["subject"]), visibility_keys=keys, evidence=ev)
            o = make_entity(tenant, scope_key, str(t["object"]), visibility_keys=keys, evidence=ev)
            entities.setdefault(s.entity_id, s)
            entities.setdefault(o.entity_id, o)
            predicate = str(t["predicate"]).lower().replace(" ", "_")
            relations.append(
                Relation(
                    relation_id=relation_id_for(
                        tenant, s.entity_id, predicate, o.entity_id, str(t.get("chunk_id") or "doc")
                    ),
                    tenant_id=tenant,
                    subject_id=s.entity_id,
                    predicate=predicate,
                    object_id=o.entity_id,
                    scope_key=scope_key,
                    visibility_keys=keys,
                    observed_at=now,
                    confidence=float(t.get("confidence", 0.7)),
                    evidence=ev,
                    document_id=version.document_id,
                    fact_text=str(
                        t.get("text") or f"{t['subject']} {t['predicate']} {t['object']}"
                    ),
                    attributes={"page": t.get("page")},
                )
            )
        return list(entities.values()), relations
