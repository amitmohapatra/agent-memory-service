"""Pure mapping tests: config -> lineage -> scope; messages; idempotency keys."""

from __future__ import annotations

import pytest

from universal_memory_langgraph import (
    Lineage,
    Segment,
    as_view,
    lineage_from_config,
    new_messages,
    safe_id,
    scope_fields,
    trailing_human,
)

pytestmark = pytest.mark.unit
DEFAULTS = {"tenant_id": "acme", "user_id": "u1", "workspace_id": "ws1", "group_ids": []}


def _cfg(ns: str, thread: str = "chat 42", step: int = 3, **memory):
    return {
        "configurable": {"thread_id": thread, "checkpoint_ns": ns, "memory": memory},
        "metadata": {"langgraph_step": step, "langgraph_node": ns.split("|")[-1].split(":")[0]},
    }


def test_root_node_acts_as_the_user() -> None:
    lin = lineage_from_config(_cfg("answer:aaa-1"))
    assert lin.segments == (Segment("answer", "aaa-1"),) and lin.subgraphs == ()
    assert lin.node == "answer" and lin.step == 3
    scope = scope_fields(lin, defaults=DEFAULTS)
    assert scope["thread_id"] == "chat-42"  # coerced to the id alphabet
    assert scope["session_id"] == "chat-42-session" and scope["turn_id"] == "chat-42-step3"
    assert "agent_id" not in scope and "agent_run_id" not in scope
    assert scope["tenant_id"] == "acme" and scope["user_id"] == "u1"


def test_subgraphs_become_agent_runs_with_stable_ids() -> None:
    lin = lineage_from_config(_cfg("supervisor:A|research:B|worker:C"))
    assert [s.node for s in lin.subgraphs] == ["supervisor", "research"]
    scope = scope_fields(lin, defaults=DEFAULTS)
    assert scope["agent_id"] == "research"
    assert scope["agent_run_id"] == "lg-B" and scope["parent_agent_run_id"] == "lg-A"
    # the same namespace (a retried superstep) maps to the same run ids
    assert (
        scope_fields(
            lineage_from_config(_cfg("supervisor:A|research:B|worker:C")), defaults=DEFAULTS
        )
        == scope
    )
    # a different invocation of the subgraph is a different run
    other = scope_fields(
        lineage_from_config(_cfg("supervisor:A|research:Z|worker:C")), defaults=DEFAULTS
    )
    assert other["agent_run_id"] == "lg-Z" and other["parent_agent_run_id"] == "lg-A"


def test_explicit_agent_on_a_root_node_uses_the_task_id() -> None:
    scope = scope_fields(lineage_from_config(_cfg("plan:T1")), defaults=DEFAULTS, agent="planner")
    assert scope["agent_id"] == "planner" and scope["agent_run_id"] == "lg-T1"
    assert "parent_agent_run_id" not in scope
    nested = scope_fields(
        lineage_from_config(_cfg("crew:A|plan:T1")), defaults=DEFAULTS, agent="planner"
    )
    assert nested["agent_run_id"] == "lg-T1" and nested["parent_agent_run_id"] == "lg-A"


def test_config_memory_overrides_and_turn_hint() -> None:
    lin = lineage_from_config(
        _cfg("answer:1", user_id="u2", session_id="ses_9", turn_id="trn_9", work_id="w1")
    )
    scope = scope_fields(lin, defaults=DEFAULTS, turn_hint="ignored-when-explicit")
    assert scope["user_id"] == "u2" and scope["session_id"] == "ses_9"
    assert scope["turn_id"] == "trn_9" and scope["work_id"] == "w1"
    hinted = scope_fields(lineage_from_config(_cfg("answer:1")), defaults=DEFAULTS, turn_hint="m-7")
    assert hinted["turn_id"] == "turn-m-7"
    assert scope_fields(Lineage(thread_id=None), defaults=DEFAULTS) == {
        "tenant_id": "acme",
        "user_id": "u1",
        "workspace_id": "ws1",
        "group_ids": [],
    }


def test_safe_id() -> None:
    assert safe_id("a b/c|d") == "a-b-c-d" and safe_id("") == "x" and safe_id("-.-") == "x"
    assert len(safe_id("x" * 500)) == 200


class _Msg:
    def __init__(self, type_, content, id_=None):
        self.type, self.content, self.id = type_, content, id_


def test_message_views_and_new_messages() -> None:
    assert as_view(("user", "hi")).type == "human"
    assert as_view({"role": "assistant", "content": "yo", "id": "m1"}).id == "m1"
    multi = as_view(_Msg("ai", [{"type": "text", "text": "a"}, {"type": "image", "url": "x"}, "b"]))
    assert multi.content == "a\nb"
    assert as_view({"role": "weird", "content": "?"}) is None and as_view(42) is None
    result = {"messages": [_Msg("ai", "answer", "m2"), _Msg("tool", ""), ("system", "s")]}
    assert [m.type for m in new_messages({}, result)] == ["ai", "system"]
    assert new_messages({}, {"messages": _Msg("ai", "one")})[0].content == "one"
    assert new_messages({}, "not a mapping") == [] and new_messages({}, {}) == []
    state = {"messages": [_Msg("ai", "old"), _Msg("human", "q1", "h1"), _Msg("human", "q2", "h2")]}
    assert [m.id for m in trailing_human(state)] == ["h1", "h2"]
    assert trailing_human({"messages": [_Msg("ai", "old")]}) == [] and trailing_human({}) == []
