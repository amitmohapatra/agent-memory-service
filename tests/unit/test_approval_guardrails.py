"""An irreversible tool is never offered "approve automatically", however its calls were
decided; "always ask" is offered for every tier, and accepting enforces the same rule."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.errors import Conflict
from memory_service.domain.feedback import (
    Feedback,
    FeedbackProjection,
    FeedbackTargetKind,
    FeedbackVerdict,
    ProjectionAction,
)
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


def _approval(reviewer: str, n: int, args: dict[str, Any] | None = None) -> Feedback:
    return Feedback(
        feedback_id=f"fb_{n}",
        tenant_id="acme",
        agent_id="bot",
        target_kind=FeedbackTargetKind.TOOL_CALL,
        target_id=f"call_{n}",
        verdict=FeedbackVerdict.APPROVE,
        reviewer=reviewer,
        metadata={"tool": "t", "args": args or {"path": f"/p{n}"}},
        projection=FeedbackProjection(action=ProjectionAction.TOOL_CALL_COUNTED),
    )


class _Decisions:
    """The counted decisions on the tool: by default from two reviewers."""

    def __init__(self, *reviewers: str) -> None:
        names = reviewers or ("user:lead", "user:ops")
        self.decisions = [_approval(name, n) for n, name in enumerate(names * 10)]

    async def counted_decisions(self, *_: Any, **__: Any) -> list[Feedback]:
        return self.decisions


CTX = MemoryExecutionContext(tenant_id="acme", user_id="u", workspace_id="ws1", agent_id="bot")


async def _accept(counts: ApprovalCounts | None, entry: ToolDescriptor | None) -> _Tools:
    tools = _Tools(counts, entry)
    uow: Any = SimpleNamespace(tools=tools, revisions=_Revisions(), feedback=_Decisions())
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


async def test_a_learned_rule_never_relaxes_an_administrators_approve_when() -> None:
    """B6: an administrator's ``approve_when: "true"`` on a tool became "never ask for this
    shape" once the learned suggestion was accepted. Approving automatically is offered only
    while the tool has no approve_when; asking more still is."""
    ruled = _tool("write").model_copy(update={"approve_when": "true"})
    assert approvals.offered(AUTO, ruled) is None
    assert approvals.offered(ASK, ruled) == "always_ask"
    with pytest.raises(Conflict, match="approve_when"):
        await _accept(AUTO, ruled)
    # the rule this very suggestion wrote is still listed (as accepted)
    own = approvals.compose(None, "write", "auto_approve", SHAPE)
    assert approvals.offered(AUTO, _tool("write").model_copy(update={"approve_when": own}))


async def test_approve_automatically_needs_two_different_reviewers() -> None:
    """B6: twenty approvals by one reviewer made "approve automatically" a suggestion. It is
    offered only once two different reviewers approved calls of that shape, as the counted
    decisions name them; approvals of another shape are not counted."""
    entry = _tool("write")

    async def offered(decisions: _Decisions) -> str | None:
        uow: Any = SimpleNamespace(feedback=decisions)
        return await approvals.offer(uow, "acme", AUTO, entry)

    assert await offered(_Decisions("user:lead")) is None
    assert await offered(_Decisions("user:lead", "user:ops")) == "auto_approve"
    other_shape = _Decisions("user:lead")
    other_shape.decisions.append(_approval("user:ops", 99, args={"path": 3}))
    assert await offered(other_shape) is None
    # "always ask" only adds a question: one reviewer is enough
    one: Any = SimpleNamespace(feedback=_Decisions("user:lead"))
    assert await approvals.offer(one, "acme", ASK, entry) == "always_ask"
    tools = _Tools(AUTO, entry)
    uow: Any = SimpleNamespace(tools=tools, revisions=_Revisions(), feedback=_Decisions("u:1"))
    with pytest.raises(Conflict, match="reviewers"):
        await approvals.accept(uow, CTX, approvals.suggestion_id(AUTO))
    assert tools.upserted is None
