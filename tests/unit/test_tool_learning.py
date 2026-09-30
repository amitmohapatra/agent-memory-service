"""Tool learning without a database: argument shapes and approval rules, task slots, graph
edges from a call, the learning job's admission and merge rules, and the hint resolvers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.graph import IDENTIFIED_BY, USED_ENTITY, layer_for
from memory_service.domain.learning import ApprovalCounts
from memory_service.domain.tools import RunOutcome, StoredProcedure, ToolDescriptor, ToolInvocation
from memory_service.modules.tools.edges import tool_edges
from memory_service.modules.tools.hints import (
    _candidates,
    _from_memories,
    _from_procedure,
    _from_profile,
    _from_task,
    _missing,
    _Resolution,
    next_tool,
    rank_procedures,
)
from memory_service.modules.tools.learning import (
    accept_distilled,
    merge,
    succeeded,
)
from memory_service.modules.tools.patterns import task_pattern, task_slots
from memory_service.modules.tools.procedures import mine_procedure
from memory_service.modules.tools.trajectories import build_trajectory
from trellis.memory.approval import arg_shape

NOW = datetime(2026, 9, 30, tzinfo=UTC)


def _call(tool: str, step: int, run: str = "run_1", **kw) -> ToolInvocation:
    base = {
        "tenant_id": "acme",
        "tool_id": f"tol_{tool}",
        "tool_name": tool,
        "run_id": run,
        "step": step,
        "visibility_keys": ["principal:acme/agent:u1/bot"],
        "principal_id": "agent:u1/bot",
        "occurred_at": NOW,
    }
    base.update(kw)
    return ToolInvocation(**base)


# ------------------------------------------------------------------ approvals


def test_an_argument_shape_names_kinds_and_magnitudes_never_values() -> None:
    shape = arg_shape({"supplier": "Acme", "amount": 12_500, "urgent": True, "lines": [1]})
    assert shape == "amount:num:1e4,lines:list,supplier:str,urgent:bool:true"
    assert arg_shape({"amount": 0.5}) == "amount:num:1e-1"
    assert arg_shape(None) == arg_shape({}) == ""


@pytest.mark.parametrize(
    ("approvals", "rejections", "edits", "expected"),
    [
        (5, 0, 0, "auto_approve"),
        (19, 1, 0, "auto_approve"),
        (1, 3, 1, "always_ask"),
        (3, 1, 1, None),  # mixed: nothing to suggest
        (4, 0, 0, None),  # too little support
    ],
)
def test_approvals_suggest_a_rule_only_with_support_and_a_clear_rate(
    approvals: int, rejections: int, edits: int, expected: str | None
) -> None:
    counts = ApprovalCounts(
        agent_id="bot",
        tool="erp-create_po",
        arg_shape="amount:num:1e3",
        approvals=approvals,
        rejections=rejections,
        edits=edits,
    )
    assert counts.suggestion() == expected


# ------------------------------------------------------------------ task slots


def test_the_task_s_values_come_back_in_the_order_it_names_them() -> None:
    task = "email bob@example.com the invoice INV-2201 for EUR 1200 by 2026-10-01"
    slots = task_slots(task)
    assert slots == [
        ("email", "bob@example.com"),
        ("id", "INV-2201"),
        ("money", "EUR 1200"),
        ("date", "2026-10-01"),
    ]
    assert task_pattern(task).count("{") == len(slots)
    assert task_slots("") == []


# ------------------------------------------------------------------ graph edges


def test_a_typed_argument_is_used_and_its_returned_id_identifies_it() -> None:
    entry = ToolDescriptor(
        tenant_id="acme", name="erp-find_supplier", argument_entity_types={"supplier": "ORG"}
    )
    call = _call(
        "erp-find_supplier",
        0,
        args_redacted={"supplier": "Acme Paper"},
        output_fields={"supplier_id": "SUP-42", "rows[0].id": 7, "name": "Acme Paper"},
    )
    entities, relations = tool_edges(call, entry)
    assert [e.entity_type for e in entities] == ["TOOL", "ORG", "IDENTIFIER", "IDENTIFIER"]
    assert [r.predicate for r in relations] == [USED_ENTITY, IDENTIFIED_BY, IDENTIFIED_BY]
    assert {layer_for(r.predicate) for r in relations} == {"procedural"}
    assert all(r.visibility_keys == call.visibility_keys for r in relations)


def test_an_untyped_redacted_or_failed_call_adds_nothing_it_cannot_stand_behind() -> None:
    entry = ToolDescriptor(tenant_id="acme", name="t", argument_entity_types={"who": "PERSON"})
    assert tool_edges(_call("t", 0, args_redacted={"other": "x"}), entry) == ([], [])
    assert tool_edges(_call("t", 0, args_redacted={"who": "[redacted]"}), entry) == ([], [])
    failed = _call("t", 0, args_redacted={"who": "Ann"}, output_fields={"id": 1}, status="error")
    _, relations = tool_edges(failed, entry)
    assert [r.predicate for r in relations] == [USED_ENTITY]


# ------------------------------------------------------------------ learning


def test_an_unlabelled_run_is_a_weak_positive_only_once_old_and_clean() -> None:
    clean = [_call("a", 0, occurred_at=NOW - timedelta(hours=25))]
    assert succeeded(None, clean, NOW) is True
    assert succeeded(None, [_call("a", 0)], NOW) is False
    assert succeeded(None, [*clean, _call("b", 1, status="error")], NOW) is False
    label = RunOutcome(tenant_id="acme", run_id="run_1", success=False)
    assert succeeded(label, clean, NOW) is False


def _trajectories(n: int, *, success: bool = True):
    runs = []
    for i in range(n):
        calls = [
            _call("lookup", 0, run=f"r{i}", output_fields={"quote": f"Q-{i}"}),
            _call("update", 1, run=f"r{i}", args_redacted={"quote": f"Q-{i}"}),
        ]
        runs.append(build_trajectory(f"r{i}", calls, succeeded=success))
    return runs


def test_a_procedure_is_admitted_with_support_and_kept_on_a_delta() -> None:
    owner = _call("lookup", 0)
    one = merge(
        None,
        mine_procedure("p", _trajectories(1)),
        tenant_id="acme",
        audience="k",
        owner=owner,
    )
    assert one is not None and one.status == "candidate" and one.title == "p"
    two = merge(
        one, mine_procedure("p", _trajectories(2)), tenant_id="acme", audience="k", owner=owner
    )
    assert two is not None and two.status == "active" and two.procedure_id == one.procedure_id
    assert {b["argument"] for b in two.bindings} == {"quote"}
    distilled = two.model_copy(update={"title": "Reprice", "distilled": two.steps_hash})
    again = merge(
        distilled,
        mine_procedure("p", _trajectories(3)),
        tenant_id="acme",
        audience="k",
        owner=owner,
    )
    assert again is not None and again.title == "Reprice" and again.distilled == again.steps_hash
    rejected = two.model_copy(update={"status": "rejected"})
    assert (
        merge(
            rejected,
            mine_procedure("p", _trajectories(3)),
            tenant_id="acme",
            audience="k",
            owner=owner,
        ).status
        == "rejected"
    )  # type: ignore[union-attr]
    assert merge(two, None, tenant_id="acme", audience="k", owner=owner).status == "retired"  # type: ignore[union-attr]
    assert merge(None, None, tenant_id="acme", audience="k", owner=owner) is None


def test_only_a_complete_distillation_is_accepted() -> None:
    assert accept_distilled(None) is None
    assert accept_distilled({"title": "", "strategy": "x", "avoid": ""}) is None
    title, strategy = accept_distilled(
        {"title": " Reprice  a quote ", "strategy": "Do it.", "avoid": "Locks."}
    )  # type: ignore[misc]
    assert title == "Reprice a quote" and strategy == "Do it.\nAvoid: Locks."


# ------------------------------------------------------------------ hints


def _procedure(pattern: str, tools: list[str], **kw) -> StoredProcedure:
    steps = [{"ordinal": i, "tool": t} for i, t in enumerate(tools)]
    return StoredProcedure(
        tenant_id="acme", scope_key="k", pattern=pattern, steps=steps, status="active", **kw
    )


def test_procedures_rank_by_pattern_similarity_and_stay_about_the_task() -> None:
    task = "update quote Q-1 with EMEA price for SKU-2"
    close = _procedure(task_pattern(task), ["a"], success_rate=0.9)
    far = _procedure("send the weekly newsletter", ["b"])
    assert rank_procedures(task, [far, close], 3) == [close]


def test_the_next_step_is_the_first_the_run_has_not_completed() -> None:
    plan = _procedure("p", ["lookup", "update", "notify"])
    assert next_tool(plan, []) == "lookup"
    done = [_call("lookup", 0), _call("update", 1, status="error")]
    assert next_tool(plan, done) == "update"
    assert next_tool(plan, [_call("lookup", 0), _call("update", 1), _call("notify", 2)]) is None
    assert next_tool(None, done) is None


def test_candidates_stay_callable_and_the_plan_s_next_step_leads() -> None:
    plan = _procedure("p", ["lookup", "update"])
    found = [("update", 1.0), ("search", 0.9), ("lookup", 0.4)]
    ranked = _candidates(found, plan, "lookup", {}, ["lookup", "update", "extra"], 5)
    assert [c.name for c in ranked] == ["lookup", "update", "extra"]
    assert "next step" in ranked[0].why


def _job(tool: ToolDescriptor, **kw) -> _Resolution:
    return _Resolution(
        ctx=MemoryExecutionContext(tenant_id="acme", user_id="u1"),
        task=kw.pop("task", ""),
        tool=tool,
        plan=kw.pop("plan", None),
        done=kw.pop("done", []),
        scope_keys=["k"],
        **kw,
    )


def test_each_resolver_names_where_its_value_came_from() -> None:
    tool = ToolDescriptor(tenant_id="acme", name="update", required=["quote", "amount"])
    plan = _procedure(
        "p",
        ["lookup", "update"],
        bindings=[
            {"step": 1, "argument": "quote", "source_step": 0, "source_field": "quote"},
            {"step": 1, "argument": "currency", "source_step": None, "literal": "EUR"},
        ],
    )
    done = [_call("lookup", 0, output_fields={"quote": "Q-7"})]
    job = _job(tool, plan=plan, done=done)
    earlier = _from_procedure(job, "quote")
    assert earlier is not None and earlier.value == "Q-7" and earlier.evidence_id
    literal = _from_procedure(job, "currency")
    assert literal is not None and literal.value == "EUR" and literal.source == "procedure"

    class Block:
        block, text = "user", "- Delivery address: Hauptstr. 1\nname: Ann"

    assert _from_profile(_job(tool, profile=[Block()]), "delivery_address").value == "Hauptstr. 1"  # type: ignore[union-attr]

    class Memory:
        item_id, attributes = "mem_1", {"predicate": "cost_center", "object": "CC-9"}

    found = _from_memories(_job(tool, memories=[Memory()]), "costCenter")
    assert found is not None and found.value == "CC-9" and found.evidence_id == "mem_1"

    by_task = _job(tool, task="bill 1200 to bob@example.com and 1300 to ann@example.com")
    by_task.slots = task_slots(by_task.task)
    first, second = _from_task(by_task, "email"), _from_task(by_task, "email")
    assert (first.value, second.value) == ("bob@example.com", "ann@example.com")  # type: ignore[union-attr]
    assert _from_task(by_task, "email") is None, "each value fills one argument"
    question = _missing(tool, "amount").question
    assert "amount" in question and "update" in question
