"""Native (LLM-free) graph enrichment: entities and temporal relations from memories and
from parsed documents.

Memories contribute *typed* facts (subject —predicate→ object from the consolidated triple)
plus MENTIONS edges for entities named in the content. Documents contribute structural,
deterministic facts: which document/page an entity is mentioned in, which entities co-occur
in the same chunk (bounded), and where a term is defined. Every relation carries evidence
(chunk/node/page or message) so the retrieval stage can pull the *source text* — the graph
is a router to evidence, never a substitute for it.

Ids are deterministic (``ent_<hash(tenant, scope, canonical)>``, ``rel_<hash(...)>``) so
re-running enrichment upserts instead of duplicating.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from datetime import UTC, datetime

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.documents import Chunk, DocumentNode, DocumentVersion
from memory_service.domain.enums import Representation
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.ids import stable_key
from memory_service.domain.memory import CanonicalMemory
from memory_service.modules.ingestion.context_graph import (
    canonical_entity,
    extract_definitions,
    extract_entities,
)
from memory_service.ports.intelligence import Entity, Relation
from memory_service.ports.models import ProviderInfo

MAX_ENTITIES_PER_CHUNK = 6
_PRINCIPAL_TYPES = {"user": "USER", "agent": "AGENT", "thread": "THREAD", "work": "WORK"}
_GENERIC_WORDS = """
table figure section page note appendix chapter total item revenue change fy25 fy26
eur usd million billion
"""
_GENERIC = frozenset(_GENERIC_WORDS.split())


def entity_id_for(tenant_id: str, scope_key: str, canonical: str) -> str:
    return "ent_" + stable_key(tenant_id, scope_key, canonical)


def relation_id_for(
    tenant_id: str, subject_id: str, predicate: str, object_id: str, source: str
) -> str:
    return "rel_" + stable_key(tenant_id, subject_id, predicate, object_id, source)


def _entity_type(name: str, canonical: str) -> str:
    prefix = canonical.split(":", 1)[0] if ":" in canonical else ""
    if prefix in _PRINCIPAL_TYPES:
        return _PRINCIPAL_TYPES[prefix]
    if canonical.startswith("doc:"):
        return "DOCUMENT"
    if name.isupper() and 2 <= len(name) <= 6:
        return "ACRONYM"
    if re.search(r"\b(inc|corp|ltd|gmbh|plc|llc|ag|sa|co)\b\.?$", canonical):
        return "ORG"
    return "THING"


def make_entity(
    tenant_id: str,
    scope_key: str,
    name: str,
    *,
    visibility_keys: Sequence[str],
    evidence: Sequence[EvidenceRef] = (),
    canonical: str | None = None,
) -> Entity:
    canon = canonical or canonical_entity(name)
    return Entity(
        entity_id=entity_id_for(tenant_id, scope_key, canon),
        tenant_id=tenant_id,
        name=name.strip(),
        canonical_name=canon,
        entity_type=_entity_type(name, canon),
        aliases=[name.strip()] if name.strip().casefold() != canon else [],
        scope_key=scope_key,
        visibility_keys=list(visibility_keys),
        evidence=list(evidence)[:5],
    )


_SECTION_NUMBER = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[A-Z]\.|[IVX]+\.)\s+")


def _section_title(section_path: str, document_title: str) -> str | None:
    """Last element of 'Doc > 3. Financial Results > 3.2 Adjusted EBITDA' without numbering."""
    if not section_path:
        return None
    parts = [p.strip() for p in section_path.split(">") if p.strip()]
    if not parts:
        return None
    last = _SECTION_NUMBER.sub("", parts[-1]).strip()
    if not last or last.casefold() == (document_title or "").casefold():
        return None
    return last


def _usable_entity(name: str) -> bool:
    canon = canonical_entity(name)
    if canon in _GENERIC or len(canon) < 3:
        return False
    return not re.fullmatch(r"[\d.,%]+", canon)


class NativeGraphEnrichment:
    info = ProviderInfo(
        name="native-graph", license="Apache-2.0", origin="memory-service", locality="local"
    )

    # -- memories ---------------------------------------------------------------------
    async def enrich_memory(
        self, memory: CanonicalMemory, ctx: MemoryExecutionContext
    ) -> tuple[list[Entity], list[Relation]]:
        keys = list(memory.system_metadata.get("visibility_keys", []))
        scope_key = memory.scope.key()
        tenant = memory.tenant_id
        evidence = list(memory.evidence)
        entities: dict[str, Entity] = {}
        relations: list[Relation] = []

        def ent(name: str, canonical: str | None = None) -> Entity:
            e = make_entity(
                tenant,
                scope_key,
                name,
                visibility_keys=keys,
                evidence=evidence,
                canonical=canonical,
            )
            return entities.setdefault(e.entity_id, e)

        subject: Entity | None = None
        if memory.subject:
            subject = ent(memory.subject, canonical_entity(memory.subject))
        obj_text = (memory.object or "").strip()
        if subject is not None and memory.predicate and obj_text and len(obj_text) <= 80:
            obj = ent(obj_text, canonical_entity(obj_text))
            relations.append(
                Relation(
                    relation_id=relation_id_for(
                        tenant, subject.entity_id, memory.predicate, obj.entity_id, memory.memory_id
                    ),
                    tenant_id=tenant,
                    subject_id=subject.entity_id,
                    predicate=memory.predicate,
                    object_id=obj.entity_id,
                    scope_key=scope_key,
                    visibility_keys=keys,
                    valid_from=memory.temporal.valid_from,
                    valid_to=memory.temporal.valid_to,
                    observed_at=memory.temporal.observed_at,
                    status="CURRENT" if memory.temporal.status.value == "CURRENT" else "SUPERSEDED",
                    confidence=memory.confidence,
                    evidence=evidence,
                    memory_id=memory.memory_id,
                    fact_text=memory.content,
                    attributes={
                        "memory_type": memory.memory_type.value,
                        "category": memory.system_metadata.get("category"),
                    },
                )
            )
        for name in extract_entities(memory.content, max_entities=8):
            if not _usable_entity(name):
                continue
            e = ent(name)
            if subject is None or e.entity_id == subject.entity_id:
                continue
            if any(r.object_id == e.entity_id for r in relations):
                continue
            relations.append(
                Relation(
                    relation_id=relation_id_for(
                        tenant, subject.entity_id, "mentions", e.entity_id, memory.memory_id
                    ),
                    tenant_id=tenant,
                    subject_id=subject.entity_id,
                    predicate="mentions",
                    object_id=e.entity_id,
                    scope_key=scope_key,
                    visibility_keys=keys,
                    observed_at=memory.temporal.observed_at,
                    status="CURRENT" if memory.temporal.status.value == "CURRENT" else "SUPERSEDED",
                    confidence=min(memory.confidence, 0.6),
                    evidence=evidence,
                    memory_id=memory.memory_id,
                    fact_text=memory.content,
                    attributes={"memory_type": memory.memory_type.value},
                )
            )
        return list(entities.values()), relations

    # -- documents --------------------------------------------------------------------
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
        tenant = version.tenant_id
        keys = list(visibility_keys)
        now = datetime.now(UTC)
        entities: dict[str, Entity] = {}
        relations: dict[str, Relation] = {}
        doc_canon = f"doc:{version.document_id}"
        doc_entity = make_entity(
            tenant,
            scope_key,
            document_title or version.document_id,
            visibility_keys=keys,
            canonical=doc_canon,
        )
        entities[doc_entity.entity_id] = doc_entity

        def ent(name: str, ev: Sequence[EvidenceRef]) -> Entity:
            e = make_entity(tenant, scope_key, name, visibility_keys=keys, evidence=ev)
            return entities.setdefault(e.entity_id, e)

        def rel(
            subject: Entity,
            predicate: str,
            obj: Entity,
            source: str,
            ev: list[EvidenceRef],
            *,
            text: str,
            confidence: float,
            page: int | None,
        ) -> None:
            rid = relation_id_for(tenant, subject.entity_id, predicate, obj.entity_id, source)
            prev = relations.get(rid)
            if prev is not None:
                return
            relations[rid] = Relation(
                relation_id=rid,
                tenant_id=tenant,
                subject_id=subject.entity_id,
                predicate=predicate,
                object_id=obj.entity_id,
                scope_key=scope_key,
                visibility_keys=keys,
                observed_at=now,
                confidence=confidence,
                evidence=ev,
                document_id=version.document_id,
                fact_text=text,
                attributes={"page": page},
            )

        # definitions: term -defined_in-> document (evidence: the defining node)
        for n in nodes:
            if not n.text or n.representation is Representation.DOCUMENT:
                continue
            for term in extract_definitions(n.text):
                if not _usable_entity(term):
                    continue
                ev = [
                    EvidenceRef(
                        source_type="document_chunk",
                        source_id=n.node_id,
                        document_id=version.document_id,
                        document_version_id=version.document_version_id,
                        node_id=n.node_id,
                        page=n.page_start,
                        observed_at=now,
                    )
                ]
                t = ent(term, ev)
                rel(
                    t,
                    "defined_in",
                    doc_entity,
                    n.node_id,
                    ev,
                    text=n.text.strip().split("\n", 1)[0][:300],
                    confidence=0.9,
                    page=n.page_start,
                )
        # mentions + bounded co-occurrence per chunk (evidence: the chunk); the chunk's
        # section title is an entity too ("Restructuring Programme" -discusses-> "Annualised")
        for c in chunks:
            names = [x for x in (c.entities or extract_entities(c.text)) if _usable_entity(x)]
            names = names[:MAX_ENTITIES_PER_CHUNK]
            section = _section_title(c.section_path, document_title)
            if section and _usable_entity(section) and section not in names:
                names.append(section)
            if not names:
                continue
            ev = [
                EvidenceRef(
                    source_type="document_chunk",
                    source_id=c.chunk_id,
                    document_id=version.document_id,
                    document_version_id=version.document_version_id,
                    chunk_id=c.chunk_id,
                    node_id=c.node_id,
                    page=c.page,
                    observed_at=now,
                )
            ]
            ents = [ent(n, ev) for n in names]
            snippet = c.text.strip().replace("\n", " ")[:200]
            for e in ents:
                rel(
                    e,
                    "mentioned_in",
                    doc_entity,
                    c.chunk_id,
                    ev,
                    text=(
                        f"{e.name} is mentioned in {document_title or version.document_id} "
                        f"(page {c.page}): {snippet}"
                    ),
                    confidence=0.7,
                    page=c.page,
                )
            section_entity = ent(section, ev) if section and _usable_entity(section) else None
            for a, b in itertools.combinations(ents, 2):
                if a.entity_id == b.entity_id:
                    continue
                if section_entity is not None and section_entity.entity_id in (
                    a.entity_id,
                    b.entity_id,
                ):
                    other = b if a.entity_id == section_entity.entity_id else a
                    rel(
                        section_entity,
                        "discusses",
                        other,
                        c.chunk_id,
                        ev,
                        text=(
                            f"Section '{section_entity.name}' discusses {other.name} "
                            f"(page {c.page}): {snippet}"
                        ),
                        confidence=0.6,
                        page=c.page,
                    )
                    continue
                first, second = sorted((a, b), key=lambda x: x.canonical_name)
                rel(
                    first,
                    "co_occurs_with",
                    second,
                    c.chunk_id,
                    ev,
                    text=(
                        f"{first.name} and {second.name} appear together (page {c.page}): {snippet}"
                    ),
                    confidence=0.5,
                    page=c.page,
                )
        return list(entities.values()), list(relations.values())
