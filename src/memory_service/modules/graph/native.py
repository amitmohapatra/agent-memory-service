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
from typing import Any

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.documents import Chunk, DocumentNode, DocumentVersion
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.ids import stable_key
from memory_service.domain.memory import CanonicalMemory
from memory_service.modules.graph.document_facts import DocumentIE, Fact, LexEntity
from memory_service.modules.ingestion.context_graph import canonical_entity, extract_entities
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
        """Typed entities with resolved aliases and factual relations (see
        :mod:`memory_service.modules.graph.document_facts`), plus the structural layer
        (``defined_in``, one ``mentioned_in`` per entity and document with page evidence,
        ``discusses`` per section, bounded ``co_occurs_with``) that keeps the graph a router
        to evidence."""
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
        ie = DocumentIE(nodes, chunks, document_title=document_title)
        ie.run()

        def chunk_ref(c: Chunk) -> EvidenceRef:
            return EvidenceRef(
                source_type="document_chunk",
                source_id=c.chunk_id,
                document_id=version.document_id,
                document_version_id=version.document_version_id,
                chunk_id=c.chunk_id,
                node_id=c.node_id,
                page=c.page,
                observed_at=now,
            )

        def node_ref(n: DocumentNode) -> EvidenceRef:
            return EvidenceRef(
                source_type="document_chunk",
                source_id=n.node_id,
                document_id=version.document_id,
                document_version_id=version.document_version_id,
                node_id=n.node_id,
                page=n.page_start,
                observed_at=now,
            )

        by_node = {n.node_id: n for n in nodes}

        def ent(lex: LexEntity) -> Entity:
            mentions = ie.mentions.get(lex.canonical, [])
            ev: list[EvidenceRef] = []
            seen_chunks: set[str] = set()
            for c, _ in mentions:
                if c.chunk_id not in seen_chunks:
                    seen_chunks.add(c.chunk_id)
                    ev.append(chunk_ref(c))
            if not ev and lex.definition_node_id and lex.definition_node_id in by_node:
                ev.append(node_ref(by_node[lex.definition_node_id]))
            e = Entity(
                entity_id=entity_id_for(tenant, scope_key, lex.canonical),
                tenant_id=tenant,
                name=lex.name,
                canonical_name=lex.canonical,
                entity_type=lex.type,
                aliases=sorted(lex.aliases)[:20],
                scope_key=scope_key,
                visibility_keys=keys,
                evidence=ev[:5],
                mention_count=max(1, len(seen_chunks)),
            )
            prev = entities.get(e.entity_id)
            if prev is not None:
                return prev
            entities[e.entity_id] = e
            return e

        def rel(
            subject: Entity,
            predicate: str,
            obj: Entity,
            source: str,
            ev: list[EvidenceRef],
            *,
            text: str,
            confidence: float,
            attributes: dict[str, Any] | None = None,
        ) -> None:
            rid = relation_id_for(tenant, subject.entity_id, predicate, obj.entity_id, source)
            if rid in relations:
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
                attributes={k: v for k, v in (attributes or {}).items() if v is not None},
            )

        # 1. facts (the semantic layer)
        for f in ie.facts:
            subj = ent(f.subject)
            if isinstance(f.object, LexEntity):
                obj = ent(f.object)
            else:  # a value that never entered the lexicon (rare)
                obj = ent(ie._value_entity(f.object))
            ev = (
                [chunk_ref(f.chunk)]
                if f.chunk is not None
                else (subj.evidence or [node_ref(nodes[0])])[:1]
            )
            page = f.chunk.page if f.chunk is not None else None
            source = (
                (f.chunk.chunk_id if f.chunk is not None else "lexicon")
                + ":"
                + str(f.attributes.get("period") or "")
                + ":"
                + str(f.attributes.get("table") or "")
            )
            rel(
                subj,
                f.predicate,
                obj,
                source,
                ev,
                text=fact_summary(f),
                confidence=f.confidence,
                attributes={"page": page, **f.attributes},
            )
        # 2. definitions -> defined_in (with the definition text)
        for lex in list(ie.lexicon.values()):
            if lex.definition and lex.definition_node_id in by_node:
                n = by_node[lex.definition_node_id]
                rel(
                    ent(lex),
                    "defined_in",
                    doc_entity,
                    n.node_id,
                    [node_ref(n)],
                    text=lex.definition[:300],
                    confidence=0.9,
                    attributes={"page": n.page_start, "definition": lex.definition[:400]},
                )
        # 3. one mentioned_in per entity (pages as attributes), discusses per section,
        #    bounded co-occurrence among named (non-value, non-section) entities per chunk
        for lex in list(ie.lexicon.values()):
            mentions = ie.mentions.get(lex.canonical, [])
            if not mentions or lex.type in VALUE_TYPES:
                continue
            e = ent(lex)
            pages = sorted({c.page for c, _ in mentions if c.page is not None})
            first_chunk = mentions[0][0]
            rel(
                e,
                "mentioned_in",
                doc_entity,
                "doc",
                e.evidence or [chunk_ref(first_chunk)],
                text=f"{e.name} is mentioned in {document_title or version.document_id} "
                f"(pages {', '.join(map(str, pages)) or '?'})",
                confidence=0.7,
                attributes={
                    "pages": pages,
                    "mentions": len(mentions),
                    "page": pages[0] if pages else None,
                },
            )
        for c in chunks:
            names = [
                ie.forms.get(m.entity.canonical) or m.entity
                for _, m in [
                    (cc, mm)
                    for cc, mm in itertools.chain.from_iterable(ie.mentions.values())
                    if cc.chunk_id == c.chunk_id
                ]
            ]
            named = []
            seen: set[str] = set()
            for lex in names:
                if lex.type in VALUE_TYPES or lex.type == "SECTION" or lex.canonical in seen:
                    continue
                seen.add(lex.canonical)
                named.append(lex)
            section = ie._section_of(c)
            ev = [chunk_ref(c)]
            snippet = c.text.strip().replace("\n", " ")[:200]
            if section is not None and section.type in ("SECTION", "EVENT"):
                se = ent(section)
                for lex in named[:8]:
                    if lex is section:
                        continue
                    rel(
                        se,
                        "discusses",
                        ent(lex),
                        c.chunk_id,
                        ev,
                        text=f"Section '{se.name}' discusses {lex.name} (page {c.page}): {snippet}",
                        confidence=0.6,
                        attributes={"page": c.page},
                    )
            for a, b in list(itertools.combinations(named[:6], 2))[:8]:
                first, second = sorted((a, b), key=lambda x: x.canonical)
                rel(
                    ent(first),
                    "co_occurs_with",
                    ent(second),
                    c.chunk_id,
                    ev,
                    text=f"{first.name} and {second.name} appear together (page {c.page}): {snippet}",
                    confidence=0.4,
                    attributes={"page": c.page},
                )
        return list(entities.values()), list(relations.values())


VALUE_TYPES = frozenset({"MONEY", "PERCENT", "COUNT", "DATE", "NUMBER", "PERIOD"})


def fact_summary(f: Fact) -> str:
    """Human-readable statement with the sentence it came from."""
    a = f.attributes
    obj = f.object.name if isinstance(f.object, LexEntity) else f.object.display
    period = f" ({a['period']})" if a.get("period") else ""
    if f.predicate in ("has_value", "would_have_value"):
        head = f"{f.subject.name}{period} {'would have been' if f.predicate == 'would_have_value' else '='} {obj}"
        extras = []
        if a.get("change"):
            extras.append(f"{a['change']} YoY")
        if a.get("previous_value"):
            extras.append(f"from {a['previous_value']}")
        if a.get("estimate"):
            extras.append("estimate")
        if a.get("per"):
            extras.append(f"per {a['per']}")
        if a.get("condition"):
            extras.append(f"if {a['condition']}")
        if a.get("table"):
            extras.append(a["table"])
        if extras:
            head += " (" + "; ".join(extras) + ")"
    elif f.predicate == "driven_by":
        head = f"{f.subject.name}{period} {'increase' if a.get('direction') == 'up' else 'decrease' if a.get('direction') == 'down' else 'change'} driven by {obj}"
    elif f.predicate == "excludes":
        head = f"{f.subject.name} excludes {obj}" + (f" ({a['value']})" if a.get("value") else "")
    elif f.predicate in ("reduced", "cut", "lowered", "increased", "raised", "grew", "expanded"):
        head = (
            f"{f.subject.name} {f.predicate} {obj}"
            + (f" by {a['by']}" if a.get("by") else "")
            + period
        )
    elif f.predicate in (
        "consolidated",
        "closed",
        "opened",
        "added",
        "divested",
        "sold",
        "acquired",
    ) and a.get("count"):
        head = f"{f.subject.name} {f.predicate} {a['count']} {obj}{period}"
    elif f.predicate == "segment_of":
        head = f"{f.subject.name} is a segment of {obj}"
    elif f.predicate == "refers_to":
        head = f"{f.subject.name} refers to {obj} ({a.get('reference', '')})"
    else:
        head = f"{f.subject.name} {f.predicate.replace('_', ' ')} {obj}{period}"
    sentence = f.text.strip()
    if sentence and not sentence.startswith(f.subject.name + " (") and sentence[:40] != head[:40]:
        return f"{head} — {sentence[:240]}"
    return head
