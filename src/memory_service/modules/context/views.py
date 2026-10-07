"""The context as the API sends it.

Two forms, each holding only what its reader uses and nothing twice:

- **prompt** (what an agent puts in front of its model): ``rendered``, the ``bundle_id`` that
  ``/v1/verify`` and the handles refer to, its size, whether there is evidence to answer from,
  and - when tools were given - the tools that fit, with a confidence the caller can narrow on;
- **full** (for a caller that builds its own prompt): the same content as structured data -
  the conversation window, the pinned sections, the memories, document passages, graph facts
  and summaries, the procedures and the tool choices - and no ``rendered`` copy of it.

Every number a caller sees is in 0..1: ``relevance`` for an item (its similarity to the
question, comparable across the bundle), ``confidence`` for a tool. Ranking internals (the raw
fusion score, which retrievers found an item, citation strings that repeat the id), empty
fields and what the bundle would only repeat (``ContextBundle.redundant``) are left out. The
build's ``diagnostics`` are added by the caller only when asked for.
"""

from __future__ import annotations

from typing import Any

from memory_service.domain.context_bundle import ContextBundle, ContextItem
from memory_service.domain.enums import Representation
from memory_service.domain.tools import SkillView, ToolHints

#: Decimal places of every 0..1 number on the wire.
PLACES = 2


def prompt_view(bundle: ContextBundle) -> dict[str, Any]:
    """The prompt form of a bundle."""
    body: dict[str, Any] = {
        "bundle_id": bundle.bundle_id,
        "rendered": bundle.render(),
        "token_estimate": bundle.token_estimate,
        "evidence_status": str(bundle.evidence.status),
    }
    if bundle.tools is not None:
        body["tools"] = [
            {"name": c.name, "confidence": c.confidence} for c in bundle.tools.candidates
        ]
    return body


def full_view(bundle: ContextBundle) -> dict[str, Any]:
    """The full form of a bundle: its content as structured data."""
    repeated = bundle.redundant()
    body: dict[str, Any] = {
        "bundle_id": bundle.bundle_id,
        "evidence_status": str(bundle.evidence.status),
        "token_estimate": bundle.token_estimate,
    }
    _put(body, "missing_evidence", list(bundle.evidence.missing_groups))
    window = bundle.conversation
    if window.messages:
        body["conversation"] = {
            "thread_id": window.thread_id,
            "messages": [
                {"id": m.message_id, "role": m.role, "text": m.text} for m in window.messages
            ],
        }
    if bundle.thread_summary is not None:
        body["thread_summary"] = bundle.thread_summary.text
    _put(body, "profile", [{"block": b.block, "text": b.text} for b in bundle.profile])
    _put(body, "skills", [skill_body(p) for p in bundle.procedures])
    if bundle.tools is not None:
        _put(body, "tools", tool_choices(bundle.tools))
    _put(body, "memories", [_memory(m) for m in bundle.memories if m.item_id not in repeated])
    _put(body, "knowledge", [_passage(k) for k in bundle.knowledge])
    _put(body, "graph_facts", [_fact(f) for f in bundle.graph_facts if f.item_id not in repeated])
    _put(body, "summaries", [_summary(s) for s in bundle.summaries])
    return body


def tool_choices(hints: ToolHints) -> list[dict[str, Any]]:
    """Each candidate with its confidence, its track record, whether it is the learned
    plan's next step, the arguments already found and the required ones nothing found."""
    out: list[dict[str, Any]] = []
    for c in hints.candidates:
        prefix = f"{c.name}."
        choice: dict[str, Any] = {"name": c.name, "confidence": c.confidence}
        if c.success_rate is not None:
            choice["success_rate"] = round(c.success_rate, PLACES)
        if c.name == hints.next:
            choice["next"] = True
        args = {
            key.removeprefix(prefix): p.value
            for key, p in hints.prefill.items()
            if key.startswith(prefix)
        }
        _put(choice, "args", args)
        missing = [
            {"arg": m.arg, "question": m.question}
            | ({"entity_type": m.entity_type} if m.entity_type else {})
            for m in hints.missing
            if m.tool == c.name
        ]
        _put(choice, "missing", missing)
        out.append(choice)
    return out


def hints_view(hints: ToolHints) -> dict[str, Any]:
    """``/v1/tools/hints`` and the ``tool_search`` memory tool: the choices and the plan."""
    body: dict[str, Any] = {"tools": tool_choices(hints)}
    if hints.plan is not None:
        body["plan"] = skill_body(hints.plan)
    return body


def skill_body(skill: SkillView) -> dict[str, Any]:
    """A learned skill: its name, its steps, the agent's own skill it adds to, what fixed a
    failing step, and how often it worked."""
    body: dict[str, Any] = {"id": skill.id, "name": skill.name, "steps": skill.steps}
    _put(body, "with_skill", skill.with_skill)
    _put(body, "fixes", skill.fixes)
    body["success_rate"] = round(skill.success_rate, PLACES)
    body["runs"] = skill.support
    return body


def _memory(item: ContextItem) -> dict[str, Any]:
    attributes = item.attributes
    body = _item(item)
    for key in ("observed_at", "subject"):
        if attributes.get(key):
            body[key] = attributes[key]
    _put(body, "dates", [dict(d) for d in attributes.get("dated_mentions") or []])
    sources = [e.source_id for e in item.evidence]
    _put(body, "sources", sources)
    return body


def _passage(item: ContextItem) -> dict[str, Any]:
    body = _item(item)
    if item.representation is not Representation.CHUNK:
        body["kind"] = str(item.representation).lower()
    for key, value in (
        ("document_id", item.document_id),
        ("page", item.page),
        ("section", item.section_path),
    ):
        if value is not None:
            body[key] = value
    return body


def _fact(item: ContextItem) -> dict[str, Any]:
    attributes = item.attributes
    body: dict[str, Any] = {"id": item.item_id}
    for key in ("subject", "predicate", "object"):
        body[key] = attributes.get(key, "")
    body["relevance"] = round(item.relevance, PLACES)
    for key in ("observed_at", "valid_from", "valid_to"):
        if attributes.get(key):
            body[key] = attributes[key]
    if item.document_id:
        body["document_id"] = item.document_id
    return body


def _summary(item: ContextItem) -> dict[str, Any]:
    return _item(item)


def _item(item: ContextItem) -> dict[str, Any]:
    return {"id": item.item_id, "text": item.text, "relevance": round(item.relevance, PLACES)}


def _put(body: dict[str, Any], key: str, value: list[Any] | dict[str, Any] | str | None) -> None:
    """Set ``key`` only when there is something in it: an absent key means none."""
    if value:
        body[key] = value
