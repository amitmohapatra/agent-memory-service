"""The service accepts the rule people's decisions support on its own (ADR 0042): once, for a
tool the catalog holds whose risk is not the strictest tier; everything else waits."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
import structlog
from structlog.testing import capture_logs

from memory_service.domain.learning import ApprovalCounts
from memory_service.domain.tools import ToolAnnotations, ToolDescriptor
from memory_service.modules.tools import approvals
from trellis.memory.approval import evaluate, when_shape

SHAPE = "path:str"
NOW = datetime(2026, 10, 21, tzinfo=UTC)
AUTO = ApprovalCounts(agent_id="bot", tool="t", arg_shape=SHAPE, approvals=20)
ASK = ApprovalCounts(agent_id="bot", tool="t", arg_shape=SHAPE, approvals=1, rejections=5)
THIN = ApprovalCounts(agent_id="bot", tool="t", arg_shape=SHAPE, approvals=4)
ONCE = AUTO.model_copy(update={"auto_accepted_at": NOW})


def _tool(side_effects: Any = None, approve_when: str | None = None, **hints: bool):
    return ToolDescriptor(
        tenant_id="acme",
        name="t",
        side_effects=side_effects,
        approve_when=approve_when,
        annotations=ToolAnnotations(**hints),
    )


class _Tools:
    def __init__(self, counts: ApprovalCounts | None, entry: ToolDescriptor | None) -> None:
        self.counts, self.entry = counts, entry
        self.upserted: ToolDescriptor | None = None
        self.marked: list[tuple[Any, ...]] = []
        self.ensured = False

    async def approval_pattern(self, *_: Any) -> ApprovalCounts | None:
        return self.counts

    async def by_name(self, *_: Any, **__: Any) -> ToolDescriptor | None:
        return self.entry

    async def ensure(self, *_: Any) -> ToolDescriptor:  # pragma: no cover - must not be called
        self.ensured = True
        raise AssertionError("an uncatalogued tool is never created on the service's word")

    async def upsert(self, entry: ToolDescriptor, **_: Any) -> tuple[ToolDescriptor, bool]:
        self.upserted = entry
        return entry, True

    async def mark_auto_accepted(self, *args: Any) -> None:
        self.marked.append(args)


class _UoW:
    def __init__(self, tools: _Tools) -> None:
        self.tools = tools
        self.serialized: list[str] = []
        self.bumped = 0
        self.revisions = SimpleNamespace(bump=self._bump)

    async def _bump(self, *_: Any) -> int:
        self.bumped += 1
        return self.bumped

    async def serialize(self, *keys: str) -> None:
        self.serialized.extend(keys)


async def _auto(counts: ApprovalCounts | None, entry: ToolDescriptor | None):
    uow = _UoW(_Tools(counts, entry))
    stored = await approvals.auto_accept(
        uow,  # type: ignore[arg-type]
        tenant_id="acme",
        agent_id="bot",
        tool="t",
        shape=SHAPE,
        workspace_id="ws1",
        now=NOW,
    )
    return stored, uow


@pytest.mark.parametrize("risk", ["read", "write", None])
async def test_a_supported_auto_approve_is_written_logged_and_marked_once(risk) -> None:
    structlog.reset_defaults()  # capture_logs sees events on the default configuration only
    with capture_logs() as events:
        stored, uow = await _auto(AUTO, _tool(risk, approve_when="true"))
    [said] = [e for e in events if e["event"] == "approval_rule.auto_accepted"]
    assert said == {
        "event": "approval_rule.auto_accepted",
        "log_level": "info",
        "tenant_id": "acme",
        "agent_id": "bot",
        "tool": "t",
        "arg_shape": SHAPE,  # names and value kinds: no argument value is logged
        "suggestion": "auto_approve",
        "support": 20,
    }
    assert stored is not None and stored.approve_when == f"not ({when_shape(SHAPE)})"
    # the rule stops the pause for calls of the learned shape, and only for them
    assert evaluate(stored.approve_when, {"path": "/tmp/x"}) is False
    assert evaluate(stored.approve_when, {"path": "/tmp/x", "force": True}) is True
    assert uow.tools.marked == [("acme", "bot", "t", SHAPE, "auto_approve", NOW)]
    assert uow.bumped == 1 and uow.serialized == ["approval-rule:acme:t"]


async def test_always_ask_tightens_a_write_tool_on_its_own() -> None:
    stored, uow = await _auto(ASK, _tool("write"))
    assert stored is not None and evaluate(stored.approve_when or "", {"path": "/x"}) is True
    assert uow.tools.marked[0][4] == "always_ask"


@pytest.mark.parametrize(
    "entry",
    [
        _tool("irreversible", approve_when="amount > 100"),  # declared strictest: never
        _tool(destructive=True),  # strictest from the MCP destructive hint
        None,  # not in the catalog: its risk is unknown
    ],
)
@pytest.mark.parametrize("counts", [AUTO, ASK])
async def test_the_strictest_or_an_unknown_tool_waits_for_a_person(entry, counts) -> None:
    stored, uow = await _auto(counts, entry)
    assert stored is None and uow.tools.upserted is None and not uow.tools.marked
    assert uow.bumped == 0


@pytest.mark.parametrize(
    ("counts", "entry"),
    [
        (None, _tool("write")),  # no decisions recorded
        (THIN, _tool("write")),  # too few to support a rule
        (ONCE, _tool("write")),  # accepted once already: a revoked rule stays revoked
        (AUTO, _tool("write", approve_when=f"not ({when_shape(SHAPE)})")),  # in effect
    ],
)
async def test_nothing_is_accepted_without_support_or_twice(counts, entry) -> None:
    stored, uow = await _auto(counts, entry)
    assert stored is None and uow.tools.upserted is None and not uow.tools.marked
