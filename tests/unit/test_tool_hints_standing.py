"""The never-rule guard on the direct paths: ``POST /v1/tools/hints`` and the ``tool_search``
agent tool build no context, so the profile they read is the stored blocks alone, which
never list the ``standing_rules`` section. Given the caller's visibility, the hints read
the rules that apply to the task themselves (``standing.standing_blocks``, on the vectors
the task is already encoded into); the context, whose profile lists them, reads them once.
"""

from __future__ import annotations

from typing import Any

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.tools import ToolHints
from memory_service.modules.agent_tools.service import AgentTools, ToolSearchArgs
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.context import builder as builder_module
from memory_service.modules.context.sections import ContextSections, Pinned, ToolsRequest
from memory_service.modules.context.views import hints_view
from memory_service.modules.tools import hints as hints_module
from memory_service.modules.tools.hints import ToolHintsService
from memory_service.modules.tools.index import ToolMatch
from memory_service.modules.tools.patterns import task_pattern
from memory_service.ports.search import SearchHit
from tests.unit.test_context_datapath import _builder
from tests.unit.test_standing_rules import _Engine, _hit, _Vectors
from tests.unit.test_tool_learning import (
    NEVER_AT_S40,
    RULE,
    STOCK_RULE_TASK,
    _HintsUow,
    _procedure,
    _rules,
)

pytestmark = pytest.mark.unit

CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1")
VISIBILITY = VisibilitySpecification(tenant_id="acme", keys=frozenset({"user:acme/u1"}))
#: the rule as stored: a memory of the user's, read back as the section lists it
STORED_NEVER = f"{NEVER_AT_S40}."


class _Index:
    """The tool index: the task's vectors (counted) and one match."""

    def __init__(self) -> None:
        self.encoded = 0

    async def encode(self, task: str) -> _Vectors:
        self.encoded += 1
        return _Vectors()

    async def search(self, *_: object, **__: object) -> list[ToolMatch]:
        return [ToolMatch("create_stock_rule", 0.9)]


class _Memories(_Engine):
    """The memories' search, recording the visibility each read was scoped to."""

    def __init__(self, hits: list[SearchHit]) -> None:
        super().__init__(hits)
        self.scopes: list[Any] = []
        self.ensured = 0

    async def ensure_collections(self) -> None:
        self.ensured += 1

    def memory_scope(self, visibility: Any) -> Any:
        self.scopes.append(visibility)
        return super().memory_scope(visibility)


def _service(hits: list[SearchHit], *, wired: bool = True) -> tuple[ToolHintsService, _Memories]:
    """The F6 setting (a skill learned at S12 makes ``create_stock_rule`` the next step),
    with ``hits`` the user's memories the standing rules are read from."""
    learned = _procedure(
        task_pattern("Create a stock rule for ST-101 at S12, min 2 max 6"),
        ["create_stock_rule"],
        success_rate=1.0,
        support=3,
    )
    engine = _Memories(hits)
    service = ToolHintsService(
        lambda: _HintsUow([RULE], [learned]),  # type: ignore[arg-type,return-value]
        _Index(),  # type: ignore[arg-type]
        None,
        engine=engine if wired else None,  # type: ignore[arg-type]
    )
    return service, engine


async def _direct(
    hits: list[SearchHit],
    task: str = STOCK_RULE_TASK,
    *,
    visibility: VisibilitySpecification | None = VISIBILITY,
    wired: bool = True,
) -> tuple[ToolHints, _Memories]:
    service, engine = _service(hits, wired=wired)
    hints = await service.hints(
        CTX,
        task,
        available=["create_stock_rule"],
        k=5,
        scope_keys=["user:acme/u1"],
        visibility=visibility,
    )
    return hints, engine


def _never_at_s40(day: str = "2026-10-01", **extra: Any) -> SearchHit:
    return _hit("never", STORED_NEVER, "rule", day=day, **extra)


async def test_a_direct_request_is_guarded_by_the_user_s_stored_never_rule() -> None:
    """The stored profile has no ``standing_rules`` section: before, the hints listed no rule
    and handed over ``create_stock_rule`` at S40 as the next step, prefilled."""
    hints, engine = await _direct([_never_at_s40()])
    assert hints.forbidden == {"create_stock_rule": STORED_NEVER}
    assert hints.next is None
    assert not [k for k in hints.prefill if k.startswith("create_stock_rule.")]
    assert not [m for m in hints.missing if m.tool == "create_stock_rule"]
    (candidate,) = hints.candidates
    assert "next step" not in candidate.why
    assert engine.scopes and all(s is VISIBILITY for s in engine.scopes), "the caller's scope"
    assert engine.ensured == 1, "a deployment with no memory indexed yet has the collection"
    (choice,) = hints_view(hints)["tools"]
    assert choice["forbidden"] == STORED_NEVER
    assert "next" not in choice and "args" not in choice and "missing" not in choice


@pytest.mark.parametrize(
    "rule",
    [
        "Never create a stock rule at store S12.",  # another store
        "Always create a stock rule at store S40 with min 1.",  # not a "never" rule
        "Never delete a stock rule at store S40.",  # another action
    ],
)
async def test_a_stored_rule_about_another_call_forbids_nothing(rule: str) -> None:
    hints, _ = await _direct([_hit("r", rule, "rule")])
    assert not hints.forbidden and hints.next == "create_stock_rule"
    assert hints.prefill["create_stock_rule.store"].value == "S40"
    (choice,) = hints_view(hints)["tools"]
    assert "forbidden" not in choice and choice["next"] is True


async def test_a_later_rule_on_the_subject_lifts_the_never_rule() -> None:
    newer = "Create stock rules at store S40 only with my approval from now on."
    hints, _ = await _direct([_never_at_s40("2026-09-01"), _hit("b", newer, "rule")])
    assert not hints.forbidden and hints.next == "create_stock_rule"
    # a never rule a later statement replaced (an earlier value) is not current either
    hints, _ = await _direct([_never_at_s40(earlier_value=True)])
    assert not hints.forbidden and hints.next == "create_stock_rule"


async def test_with_no_rule_stored_a_direct_request_is_answered_as_before() -> None:
    read, engine = await _direct([])
    assert engine.store.filters, "the rules were read"
    before, _ = await _direct([], visibility=None, wired=False)
    # the learned skill is minted afresh per service: its id is the only difference
    assert read.plan is not None and before.plan is not None
    assert read.plan.model_copy(update={"id": ""}) == before.plan.model_copy(update={"id": ""})
    assert read.model_copy(update={"plan": None}) == before.model_copy(update={"plan": None})
    assert read.next == "create_stock_rule" and not read.forbidden


async def test_the_rules_are_read_on_the_task_s_one_encoding() -> None:
    """One encode for the catalog search, the procedures and the standing rules: the tool
    index and the memories share the indexer, so the vectors are the memories' too."""
    service, engine = _service([_never_at_s40()])
    await service.hints(
        CTX, STOCK_RULE_TASK, available=None, k=5, scope_keys=["k"], visibility=VISIBILITY
    )
    assert service.index.encoded == 1  # type: ignore[attr-defined]
    assert len(engine.store.filters) == 3, "rules, preferences, the setup: one dense search each"


async def test_without_a_visibility_the_hints_read_no_rule() -> None:
    """The context passes its profile, which already lists the rules: nothing is read
    again, and a profile that lists none forbids nothing."""
    hints, engine = await _direct([_never_at_s40()], visibility=None)
    assert not engine.store.filters and not engine.ensured and not hints.forbidden
    assert hints.next == "create_stock_rule"


async def test_a_deployment_without_the_memories_search_reads_no_rule() -> None:
    hints, engine = await _direct([_never_at_s40()], wired=False)
    assert not engine.store.filters and hints.next == "create_stock_rule"


class _Authz:
    async def visibility(self, ctx: MemoryExecutionContext) -> VisibilitySpecification:
        return VISIBILITY


class _Profile:
    async def blocks(self, uow: object, ctx: MemoryExecutionContext) -> list:
        return []  # the stored blocks: never a standing_rules section


async def test_the_tool_search_agent_tool_is_guarded_by_the_stored_never_rule() -> None:
    service, _ = _service([_never_at_s40()])
    tools = AgentTools(
        lambda: _HintsUow([]),
        {"authz": _Authz(), "profile": _Profile(), "tool_hints": service},
    )
    found = await tools._tool_search(
        CTX, ToolSearchArgs(task=STOCK_RULE_TASK), ["create_stock_rule"]
    )
    (choice,) = found["tools"]
    assert choice["name"] == "create_stock_rule" and choice["forbidden"] == STORED_NEVER
    assert "next" not in choice and "args" not in choice


class _Sections:
    """The pinned sections read nothing; the tool hints are the real ones."""

    tools = ContextSections.tools

    def __init__(self, hints: ToolHintsService) -> None:
        self.services = {"tool_hints": hints}

    async def gather(self, *_: object, **__: object) -> Pinned:
        return Pinned()


async def test_a_context_build_reads_the_standing_rules_once(monkeypatch) -> None:
    """The builder reads them for the profile; the hints read that profile, not the store."""
    reads: list[str] = []

    async def standing_blocks(
        engine: Any, query: str, visibility: Any, vectors: Any, *_: Any
    ) -> list:
        reads.append(query)
        return _rules(NEVER_AT_S40)

    monkeypatch.setattr(builder_module, "standing_blocks", standing_blocks)
    monkeypatch.setattr(hints_module, "standing_blocks", standing_blocks)
    service, _ = _service([])
    builder = _builder()
    builder.sections = _Sections(service)  # type: ignore[assignment]
    try:
        bundle = await builder.build(CTX, STOCK_RULE_TASK, tools=ToolsRequest())
    finally:
        await builder.close()
    assert reads == [STOCK_RULE_TASK]
    assert bundle.tools is not None
    assert bundle.tools.forbidden == {"create_stock_rule": NEVER_AT_S40}
    assert bundle.tools.next is None
