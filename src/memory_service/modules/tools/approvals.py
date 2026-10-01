"""Approval suggestions: the rules an agent's approve / reject / edit decisions support, and
accepting one into the catalog as the tool's ``approve_when``.

A suggestion is never applied by the service on its own: it is listed, and a person accepts
it. Accepting composes the rule with what the tool already asks for (its ``approve_when``, or
its risk tier when it has none), so an accepted rule narrows or widens exactly the calls of
that argument shape and nothing else.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Final

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.errors import Conflict, NotFound
from memory_service.domain.learning import ApprovalCounts, Suggestion
from memory_service.domain.revisions import RevisionKind
from memory_service.domain.tools import SideEffects, ToolDescriptor
from memory_service.ports.uow import UnitOfWork
from trellis.memory.approval import parse, when_shape

#: The only field accepting a suggestion changes.
APPROVE_WHEN: Final = frozenset({"approve_when"})


def suggestion_id(counts: ApprovalCounts) -> str:
    """An opaque, stable id that names the (agent, tool, argument shape) it is about."""
    raw = json.dumps([counts.agent_id, counts.tool, counts.arg_shape], separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def _decode(identifier: str) -> tuple[str, str, str]:
    try:
        padded = identifier + "=" * (-len(identifier) % 4)
        agent_id, tool, shape = json.loads(base64.urlsafe_b64decode(padded.encode()))
    except (binascii.Error, ValueError, TypeError) as exc:
        raise NotFound("no such approval suggestion") from exc
    return str(agent_id), str(tool), str(shape)


def compose(existing: str | None, risk: SideEffects, suggestion: Suggestion, shape: str) -> str:
    """The tool's ``approve_when`` with the rule for calls of ``shape`` added: always ask for
    them, or stop asking for them. Without an expression the tier is the base: irreversible
    tools ask for every call, the others for none."""
    base = existing or ("true" if risk == "irreversible" else "false")
    clause = when_shape(shape)
    if suggestion == "always_ask":
        return clause if base == "false" else f"({base}) or ({clause})"
    return f"not ({clause})" if base == "true" else f"({base}) and not ({clause})"


def offered(counts: ApprovalCounts, entry: ToolDescriptor | None) -> Suggestion | None:
    """The rule these decisions support that may be offered for this tool. Never "approve
    automatically" for an irreversible tool: however often a deletion or a payment was
    approved, the next one is still asked about. "Always ask" is offered for any tier."""
    suggestion = counts.suggestion()
    if suggestion == "auto_approve" and entry is not None and entry.risk == "irreversible":
        return None
    return suggestion


def accepted(entry: ToolDescriptor | None, shape: str) -> bool:
    """Whether the rule for this shape is already part of the tool's expression."""
    return bool(entry and entry.approve_when and when_shape(shape) in entry.approve_when)


async def accept(uow: UnitOfWork, ctx: MemoryExecutionContext, identifier: str) -> ToolDescriptor:
    """Write the suggestion into the tool's catalog entry. Only the agent whose decisions it
    was learned from may accept it, and only while the decisions still support it."""
    agent_id, tool, shape = _decode(identifier)
    if agent_id != (ctx.agent_id or ""):
        raise NotFound("no such approval suggestion")
    counts = await uow.tools.approval_pattern(ctx.tenant_id, agent_id, tool, shape)
    if counts is None or counts.suggestion() is None:
        raise Conflict("the decisions no longer support this suggestion")
    entry = await uow.tools.by_name(
        ctx.tenant_id, tool, workspace_id=ctx.workspace_id
    ) or await uow.tools.ensure(ctx.tenant_id, tool)
    suggestion = offered(counts, entry)
    if suggestion is None:
        raise Conflict(f"{tool} is irreversible: its calls are always asked about")
    if accepted(entry, shape):
        return entry
    expression = compose(entry.approve_when, entry.risk, suggestion, shape)
    parse(expression)  # composed from parsed parts; a failure here is a bug, not input
    stored, _ = await uow.tools.upsert(
        entry.model_copy(update={"approve_when": expression}), fields=APPROVE_WHEN
    )
    await uow.revisions.bump(ctx.tenant_id, RevisionKind.TENANT, "")
    return stored
