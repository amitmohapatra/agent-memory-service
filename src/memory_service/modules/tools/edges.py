"""Knowledge-graph edges from recorded tool calls (layer ``procedural``).

A catalog entry says which arguments name which kind of entity (``supplier: ORG``). A call
with such an argument ``used_entity`` it (tool -> entity); when the call succeeded and its
one typed argument came back with an id field in the output, the entity is
``identified_by`` that id (entity -> identifier). That is what lets a later task that names
"Acme" have its ``supplier_id`` filled in.

Deterministic; the entities and edges carry the call's audience, so they are read by exactly
those who could read the call.
"""

from __future__ import annotations

import re
from typing import Final

from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.graph import IDENTIFIED_BY, USED_ENTITY
from memory_service.domain.tools import ToolDescriptor, ToolInvocation
from memory_service.modules.graph.native import entity_id_for, relation_id_for
from memory_service.modules.ingestion.context_graph import canonical_entity
from memory_service.ports.intelligence import Entity, Relation

TOOL_ENTITY_TYPE: Final = "TOOL"
IDENTIFIER_ENTITY_TYPE: Final = "IDENTIFIER"
#: Identifier fields of one output that become edges.
IDENTIFIERS_MAX: Final = 3
_REDACTED: Final = "[redacted]"
_INDEX = re.compile(r"\[\d+\]")


def _is_identifier(path: str) -> bool:
    leaf = _INDEX.sub("", path).rsplit(".", 1)[-1].casefold()
    return leaf == "id" or leaf.endswith("_id")


def _entity(call: ToolInvocation, name: str, entity_type: str) -> Entity:
    audience = call.visibility_keys[0]
    canonical = canonical_entity(name)
    return Entity(
        entity_id=entity_id_for(call.tenant_id, audience, canonical),
        tenant_id=call.tenant_id,
        name=name,
        canonical_name=canonical,
        entity_type=entity_type,
        scope_key=audience,
        visibility_keys=list(call.visibility_keys),
        evidence=[_evidence(call)],
    )


def _evidence(call: ToolInvocation) -> EvidenceRef:
    return EvidenceRef(
        source_type="tool_result",
        source_id=call.invocation_id,
        agent_id=call.agent_id,
        agent_run_id=call.run_id,
        observed_at=call.occurred_at,
    )


def _relation(
    call: ToolInvocation, subject: Entity, predicate: str, obj: Entity, detail: str
) -> Relation:
    return Relation(
        relation_id=relation_id_for(
            call.tenant_id, subject.entity_id, predicate, obj.entity_id, call.invocation_id
        ),
        tenant_id=call.tenant_id,
        subject_id=subject.entity_id,
        predicate=predicate,
        object_id=obj.entity_id,
        scope_key=subject.scope_key,
        visibility_keys=list(call.visibility_keys),
        observed_at=call.occurred_at,
        confidence=1.0 if call.succeeded else 0.5,
        evidence=[_evidence(call)],
        fact_text=f"{subject.name} {predicate.replace('_', ' ')} {obj.name}",
        attributes={"detail": detail},
    )


def _typed_arguments(call: ToolInvocation, entry: ToolDescriptor) -> list[tuple[str, str, str]]:
    """``(argument, value, entity type)`` for every typed argument the call carried."""
    out = []
    for argument, entity_type in sorted(entry.argument_entity_types.items()):
        value = call.args_redacted.get(argument)
        if isinstance(value, str | int) and not isinstance(value, bool):
            text = str(value).strip()
            if text and text != _REDACTED:
                out.append((argument, text, entity_type))
    return out


def tool_edges(call: ToolInvocation, entry: ToolDescriptor) -> tuple[list[Entity], list[Relation]]:
    """The entities and edges one recorded call contributes."""
    typed = _typed_arguments(call, entry) if call.visibility_keys else []
    if not typed:
        return [], []
    tool = _entity(call, call.tool_name, TOOL_ENTITY_TYPE)
    entities, relations = [tool], []
    for argument, value, entity_type in typed:
        used = _entity(call, value, entity_type)
        entities.append(used)
        relations.append(_relation(call, tool, USED_ENTITY, used, argument))
    if len(typed) == 1 and call.succeeded:
        subject = entities[1]
        ids = [(p, v) for p, v in call.output_fields.items() if _is_identifier(p)]
        for path, value in ids[:IDENTIFIERS_MAX]:
            if isinstance(value, str | int) and not isinstance(value, bool) and str(value).strip():
                identifier = _entity(call, str(value).strip(), IDENTIFIER_ENTITY_TYPE)
                entities.append(identifier)
                relations.append(_relation(call, subject, IDENTIFIED_BY, identifier, path))
    return entities, relations
