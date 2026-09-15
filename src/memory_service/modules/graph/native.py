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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.documents import Chunk, DocumentNode, DocumentVersion
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.ids import stable_key
from memory_service.domain.memory import CanonicalMemory
from memory_service.modules.graph.document_facts import DocumentIE, Fact, LexEntity, sentences
from memory_service.modules.ingestion.context_graph import canonical_entity, extract_entities
from memory_service.modules.llm.assist import LLMAssist
from memory_service.ports.intelligence import Entity, Relation
from memory_service.ports.models import ProviderInfo

MAX_ENTITIES_PER_CHUNK = 6
_PRINCIPAL_TYPES = {"user": "USER", "agent": "AGENT", "thread": "THREAD", "work": "WORK"}
_GENERIC_WORDS = """
table figure section page note appendix chapter total item revenue change fy25 fy26
eur usd million billion
"""
_GENERIC = frozenset(_GENERIC_WORDS.split())

LLM_RELATION_MAX_CONFIDENCE = 0.8
LLM_MAX_RELATIONS_PER_MEMORY = 8
LLM_MAX_PAIRS_PER_DOCUMENT = 12
_LLM_MAX_TEXT = 1200
_LLM_MAX_SENTENCE = 300
_STRUCTURAL_PREDICATES = frozenset(
    {"mentions", "mentioned_in", "co_occurs_with", "discusses", "defined_in"}
)
_PREDICATE_RE = re.compile(r"[a-z][a-z0-9_]{1,39}")
_RELATIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["subject", "predicate", "object", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["relations"],
    "additionalProperties": False,
}
_MEMORY_RELATIONS_SYSTEM = (
    "You extract relations for a knowledge graph. Given a text and the entities already "
    "found in it, return the typed relations the text states explicitly between two of those "
    "entities. Use entity names exactly as listed; never introduce other entities. Predicates "
    "are short lowercase snake_case verb phrases (works_at, leads, reports_to, located_in, "
    "part_of, uses). Return an empty list when the text states no relation."
)
_DOCUMENT_RELATIONS_SYSTEM = (
    "You extract relations for a knowledge graph. For each numbered pair of entities and the "
    "sentence(s) where they appear together, return a typed relation only when a sentence "
    "states one explicitly between the two entities of that pair (either direction). Use entity "
    "names exactly as listed; never introduce other entities. Predicates are short lowercase "
    "snake_case verb phrases (acquired, part_of, led_by, headquartered_in, measures, "
    "reduced). Return nothing for a pair that merely co-occurs."
)


def llm_predicate(raw: object) -> str | None:
    """Model predicate normalised like native ones (``works at`` -> ``works_at``); ``None``
    when it is empty, malformed or one of the structural predicates the native layer owns."""
    pred = re.sub(r"[^a-z0-9]+", "_", str(raw).strip().casefold()).strip("_")
    if not _PREDICATE_RE.fullmatch(pred) or pred in _STRUCTURAL_PREDICATES:
        return None
    return pred


def llm_confidence(raw: object) -> float:
    value = float(raw) if isinstance(raw, int | float) else 0.5
    return min(LLM_RELATION_MAX_CONFIDENCE, max(0.1, value))


def accept_relations[E](
    out: dict[str, Any] | None,
    resolve: Callable[[str], E | None],
    *,
    allowed_pairs: set[frozenset[str]] | None = None,
    key: Callable[[E], str],
    max_relations: int,
) -> list[tuple[E, str, E, float]]:
    """Keep only model relations whose ends resolve to entities the native code already
    found (never invent entities), with a well-formed predicate and capped confidence."""
    accepted: list[tuple[E, str, E, float]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in (out or {}).get("relations", []):
        if not isinstance(item, dict) or len(accepted) >= max_relations:
            continue
        subject = resolve(str(item.get("subject", "")))
        obj = resolve(str(item.get("object", "")))
        pred = llm_predicate(item.get("predicate", ""))
        if subject is None or obj is None or pred is None or key(subject) == key(obj):
            continue
        if allowed_pairs is not None and frozenset((key(subject), key(obj))) not in allowed_pairs:
            continue
        sig = (key(subject), pred, key(obj))
        if sig in seen:
            continue
        seen.add(sig)
        accepted.append((subject, pred, obj, llm_confidence(item.get("confidence"))))
    return accepted


@dataclass
class _PairContext:
    first: LexEntity
    second: LexEntity
    chunk: Chunk
    count: int = 0
    sentences: list[str] = field(default_factory=list)


def _has_form(low: str, lex: LexEntity) -> bool:
    return any(
        re.search(r"(?<![a-z0-9\-])" + re.escape(f) + r"(?:s|es)?(?![a-z0-9\-])", low)
        for f in lex.all_forms()
        if len(f) >= 2
    )


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

    def __init__(self, *, assist: LLMAssist | None = None) -> None:
        self.assist = assist or LLMAssist.disabled()

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
        if (
            len(entities) >= 2
            and all(r.predicate == "mentions" for r in relations)
            and self.assist.wants("relation_extraction")
        ):
            known = {r.relation_id for r in relations}
            for r in await self._memory_relations(memory, list(entities.values()), keys):
                if r.relation_id not in known:
                    known.add(r.relation_id)
                    relations.append(r)
        return list(entities.values()), relations

    async def _memory_relations(
        self, memory: CanonicalMemory, entities: Sequence[Entity], keys: Sequence[str]
    ) -> list[Relation]:
        by_form: dict[str, Entity] = {}
        for e in entities:
            for form in (
                e.canonical_name,
                canonical_entity(e.name),
                *map(canonical_entity, e.aliases),
            ):
                by_form.setdefault(form, e)
        out = await self.assist.structured(
            "relation_extraction",
            system=_MEMORY_RELATIONS_SYSTEM,
            user=f"Text: {memory.content[:_LLM_MAX_TEXT]}\n"
            f"Entities: {'; '.join(e.name for e in entities)}",
            schema=_RELATIONS_SCHEMA,
            max_tokens=400,
        )
        accepted = accept_relations(
            out,
            lambda name: by_form.get(canonical_entity(name)),
            key=lambda e: e.entity_id,
            max_relations=LLM_MAX_RELATIONS_PER_MEMORY,
        )
        status = "CURRENT" if memory.temporal.status.value == "CURRENT" else "SUPERSEDED"
        return [
            Relation(
                relation_id=relation_id_for(
                    memory.tenant_id, subject.entity_id, pred, obj.entity_id, memory.memory_id
                ),
                tenant_id=memory.tenant_id,
                subject_id=subject.entity_id,
                predicate=pred,
                object_id=obj.entity_id,
                scope_key=memory.scope.key(),
                visibility_keys=list(keys),
                valid_from=memory.temporal.valid_from,
                valid_to=memory.temporal.valid_to,
                observed_at=memory.temporal.observed_at,
                status=status,
                confidence=min(confidence, memory.confidence),
                evidence=list(memory.evidence),
                memory_id=memory.memory_id,
                fact_text=memory.content,
                attributes={"memory_type": memory.memory_type.value, "extraction": "llm"},
            )
            for subject, pred, obj, confidence in accepted
        ]

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
        llm_pairs: dict[tuple[str, str], _PairContext] = {}
        llm_on = self.assist.wants("relation_extraction")
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
            chunk_sentences = sentences(c.text) if llm_on else []
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
                if llm_on:
                    pc = llm_pairs.setdefault(
                        (first.canonical, second.canonical), _PairContext(first, second, c)
                    )
                    pc.count += 1
                    if len(pc.sentences) < 2:
                        for s in chunk_sentences:
                            low = s.lower()
                            if _has_form(low, first) and _has_form(low, second):
                                if not pc.sentences:
                                    pc.chunk = c
                                pc.sentences.append(s[:_LLM_MAX_SENTENCE])
                                if len(pc.sentences) >= 2:
                                    break
        if llm_pairs:
            for first, pred, second, confidence, pc in await self._document_relations(
                ie, llm_pairs
            ):
                text = f"{first.name} {pred.replace('_', ' ')} {second.name}"
                if pc.sentences:
                    text += f" — {pc.sentences[0][:240]}"
                rel(
                    ent(first),
                    pred,
                    ent(second),
                    "llm:" + pc.chunk.chunk_id,
                    [chunk_ref(pc.chunk)],
                    text=text,
                    confidence=confidence,
                    attributes={"page": pc.chunk.page, "extraction": "llm"},
                )
        return list(entities.values()), list(relations.values())

    async def _document_relations(
        self, ie: DocumentIE, pairs: dict[tuple[str, str], _PairContext]
    ) -> list[tuple[LexEntity, str, LexEntity, float, _PairContext]]:
        top = sorted(
            pairs.values(), key=lambda p: (-p.count, p.first.canonical, p.second.canonical)
        )
        top = top[:LLM_MAX_PAIRS_PER_DOCUMENT]
        lines = []
        for i, pc in enumerate(top, start=1):
            context = " ".join(pc.sentences) or pc.chunk.text.strip().replace("\n", " ")[:200]
            lines.append(f"{i}. {pc.first.name} | {pc.second.name}\n   context: {context}")
        out = await self.assist.structured(
            "relation_extraction",
            system=_DOCUMENT_RELATIONS_SYSTEM,
            user="Pairs:\n" + "\n".join(lines),
            schema=_RELATIONS_SCHEMA,
            max_tokens=600,
        )
        by_pair = {frozenset((pc.first.canonical, pc.second.canonical)): pc for pc in top}
        accepted = accept_relations(
            out,
            lambda name: ie.forms.get(canonical_entity(name)),
            allowed_pairs=set(by_pair),
            key=lambda lex: lex.canonical,
            max_relations=LLM_MAX_PAIRS_PER_DOCUMENT,
        )
        return [
            (subject, pred, obj, confidence, by_pair[frozenset((subject.canonical, obj.canonical))])
            for subject, pred, obj, confidence in accepted
        ]


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
