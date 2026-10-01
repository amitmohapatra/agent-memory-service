"""An irreversible tool is never offered "approve automatically", however its calls were
decided; "always ask" is offered for every tier, and accepting enforces the same rule."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.errors import Conflict
from memory_service.domain.learning import ApprovalCounts
from memory_service.domain.tools import ToolAnnotations, ToolDescriptor
from memory_service.modules.tools import approvals

SHAPE = "path:str"
AUTO = ApprovalCounts(agent_id="bot", tool="t", arg_shape=SHAPE, approvals=20)
ASK = ApprovalCounts(agent_id="bot", tool="t", arg_shape=SHAPE, approvals=1, rejections=5)
THIN = ApprovalCounts(agent_id="bot", tool="t", arg_shape=SHAPE, approvals=4)


def _tool(side_effects: Any = None, **annotations: bool) -> ToolDescriptor:
    return ToolDescriptor(
        tenant_id="acme",
        name="t",
        side_effects=side_effects,
        annotations=ToolAnnotations(**annotations),
    )


@pytest.mark.parametrize(
    ("entry", "counts", "expected"),
    [
        # read and write tools: whatever the decisions support
        (_tool("read"), AUTO, "auto_approve"),
        (_tool("write"), AUTO, "auto_approve"),
        (_tool(), AUTO, "auto_approve"),  # no tier declared: write
        (None, AUTO, "auto_approve"),  # not in the catalog yet: its tier is unknown
        # irreversible, declared or from the MCP destructive hint: never auto-approve
        (_tool("irreversible"), AUTO, None),
        (_tool(destructive=True), AUTO, None),
        # "always ask" only ever adds a question, so every tier may be offered it
        (_tool("irreversible"), ASK, "always_ask"),
        (_tool("read"), ASK, "always_ask"),
        # too little support is nothing, whatever the tier
        (_tool("irreversible"), THIN, None),
        (_tool("read"), THIN, None),
    ],
)
def test_what_may_be_offered(
    entry: ToolDescriptor | None, counts: ApprovalCounts, expected: str | None
) -> None:
    assert approvals.offered(counts, entry) == expected


class _Tools:
    def __init__(self, counts: ApprovalCounts | None, entry: ToolDescriptor | None) -> None:
        self.counts, self.entry = counts, entry
        self.ensured = False
        self.upserted: ToolDescriptor | None = None

    async def approval_pattern(self, *_: Any) -> ApprovalCounts | None:
        return self.counts

    async def by_name(self, *_: Any, **__: Any) -> ToolDescriptor | None:
        return self.entry

    async def ensure(self, tenant_id: str, name: str) -> ToolDescriptor:
        self.ensured = True
        return ToolDescriptor(tenant_id=tenant_id, name=name)

    async def upsert(self, entry: ToolDescriptor, **_: Any) -> tuple[ToolDescriptor, bool]:
        self.upserted = entry
        return entry, True


class _Revisions:
    async def bump(self, *_: Any) -> int:
        return 1


CTX = MemoryExecutionContext(tenant_id="acme", user_id="u", workspace_id="ws1", agent_id="bot")


async def _accept(counts: ApprovalCounts | None, entry: ToolDescriptor | None) -> _Tools:
    tools = _Tools(counts, entry)
    uow: Any = SimpleNamespace(tools=tools, revisions=_Revisions())
    await approvals.accept(uow, CTX, approvals.suggestion_id(AUTO))
    return tools


async def test_accepting_auto_approve_for_an_irreversible_tool_is_refused() -> None:
    for entry in (_tool("irreversible"), _tool(destructive=True)):
        with pytest.raises(Conflict, match="irreversible"):
            await _accept(AUTO, entry)


async def test_accepting_auto_approve_for_a_write_tool_writes_the_rule() -> None:
    tools = await _accept(AUTO, _tool("write"))
    assert tools.upserted is not None and tools.upserted.approve_when


async def test_accepting_always_ask_for_an_irreversible_tool_is_allowed() -> None:
    tools = await _accept(ASK, _tool("irreversible"))
    assert tools.upserted is not None and tools.upserted.approve_when


async def test_a_suggestion_with_no_support_touches_no_catalog_entry() -> None:
    for counts in (None, THIN):
        tools = _Tools(counts, None)
        uow: Any = SimpleNamespace(tools=tools, revisions=_Revisions())
        with pytest.raises(Conflict, match="no longer support"):
            await approvals.accept(uow, CTX, approvals.suggestion_id(AUTO))
        assert not tools.ensured and tools.upserted is None
