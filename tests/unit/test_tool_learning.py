"""Tool learning without a database: argument shapes and approval rules, task slots, graph
edges from a call, the learning job's admission and merge rules, and the hint resolvers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.graph import IDENTIFIED_BY, USED_ENTITY, layer_for
from memory_service.domain.learning import ApprovalCounts
from memory_service.domain.tools import (
    Prefill,
    RunOutcome,
    StoredProcedure,
    ToolDescriptor,
    ToolInvocation,
)
from memory_service.modules.tools.edges import tool_edges
from memory_service.modules.tools.hints import (
    ToolHintsService,
    _as_declared,
    _candidates,
    _from_memories,
    _from_procedure,
    _from_profile,
    _from_task,
    _labelled_slots,
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


def test_an_unlabelled_clean_run_is_no_evidence_however_old() -> None:
    """B5: an unlabelled run whose calls all worked is neither a success nor - while it is
    fresh - a failure; one with a failed call is a failure."""
    old = [_call("a", 0, occurred_at=NOW - timedelta(hours=25))]
    assert succeeded(None, old) is None
    assert succeeded(None, [_call("a", 0)]) is None
    assert succeeded(None, [*old, _call("b", 1, status="error")]) is False
    label = RunOutcome(tenant_id="acme", run_id="run_1", success=False, source="human")
    assert succeeded(label, old) is False


def test_a_failure_the_system_reported_on_a_run_whose_calls_worked_is_no_evidence() -> None:
    """B3: the harness reports a time limit, a model error or a stall as the run's ``system``
    reject; with every call of the run working that is the run's failure, not its
    procedure's. A failed call, or a judge's or a person's reject, still counts."""
    clean, failed = [_call("a", 0)], [_call("a", 0, status="timeout")]

    def label(source: str) -> RunOutcome:
        return RunOutcome(tenant_id="acme", run_id="run_1", success=False, source=source)

    assert succeeded(label("system"), clean) is None
    assert succeeded(label("system"), failed) is False
    assert succeeded(label("judge"), clean) is False
    assert succeeded(label("human"), clean) is False


@pytest.mark.parametrize(
    ("source", "success", "expected"),
    [
        ("system", True, None),  # the run's own word: its outcome only
        ("system", False, False),  # its own failure is evidence
        ("judge", True, True),
        ("human", True, True),
        ("human", False, False),
    ],
)
def test_a_runs_own_unverified_success_is_no_evidence(
    source: str, success: bool, expected: bool | None
) -> None:
    calls = [_call("a", 0, status="ok" if success else "error")]
    label = RunOutcome(tenant_id="acme", run_id="run_1", success=success, source=source)
    assert succeeded(label, calls) is expected


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


def test_a_failed_call_and_its_retry_are_one_step() -> None:
    """B10: lookup, reset (TimeoutError), reset (ok) in two confirmed runs is the two-step
    procedure lookup -> reset with both runs' support, not support 0."""
    runs = [
        build_trajectory(
            f"r{i}",
            [
                _call("lookup", 0, run=f"r{i}", output_fields={"user": f"u-{i}"}),
                _call("reset", 1, run=f"r{i}", status="error", error_class="TimeoutError"),
                _call("reset", 2, run=f"r{i}", args_redacted={"user": f"u-{i}"}),
            ],
            succeeded=True,
        )
        for i in range(2)
    ]
    mined = mine_procedure("p", runs)
    assert mined is not None and mined.tools == ["lookup", "reset"]
    assert (mined.support, mined.success_rate) == (2, 1.0)
    [binding] = mined.steps[1].bindings
    assert (binding.source_step, binding.source_field) == (0, "user"), "the retry that worked"


def _style_runs(*styles: str) -> list:
    return [
        build_trajectory(
            f"r{i}",
            [
                _call(
                    "style_stock",
                    0,
                    run=f"r{i}",
                    args_redacted={"style": "ST-101", "warehouse": "W1"},
                    task=f"How many units of style {style} are there?",
                )
            ],
            succeeded=True,
        )
        for i, style in enumerate(styles)
    ]


def test_a_value_is_a_learned_literal_only_when_constant_across_different_tasks() -> None:
    """B11: three runs about ST-101 all used style='ST-101': the task's value, not the
    procedure's, so no literal. 'W1', the same in runs about two different styles and named
    by none of them, is a constant."""
    same = mine_procedure("p", _style_runs("ST-101", "ST-101", "ST-101"))
    assert same is not None
    assert {b.argument: b.literal for b in same.steps[0].bindings} == {
        "style": None,
        "warehouse": None,
    }, "runs about one style teach no constant"
    mixed = mine_procedure("p", _style_runs("ST-101", "ST-205", "ST-101"))
    assert mixed is not None
    literals = {b.argument: b.literal for b in mixed.steps[0].bindings}
    assert literals == {"style": None, "warehouse": "W1"}


def test_code_mode_meta_tools_are_no_procedure_steps() -> None:
    """B17: a Code Mode run records the meta-tools it called and the nested calls the gateway
    logged; the procedure is the nested tools, the ones the task needed."""
    calls = [
        _call("list_tool_files", 0),
        _call("execute_tool_code", 1),
        _call("wiki-search_pages", 2, output_fields={"page": "Hooks"}),
        _call("wiki-read_page", 3, args_redacted={"page": "Hooks"}),
    ]
    runs = [build_trajectory(f"r{i}", calls, succeeded=True) for i in range(2)]
    mined = mine_procedure("p", runs)
    assert mined is not None and mined.tools == ["wiki-search_pages", "wiki-read_page"]


def test_a_learned_literal_never_fills_an_argument_the_task_names_another_value_for() -> None:
    """B11: the plan learned style='ST-101'; a task about ZZ999999 gets no 'ST-101'."""
    tool = ToolDescriptor(tenant_id="acme", name="style_stock", required=["style"])
    plan = _procedure(
        "p",
        ["style_stock"],
        bindings=[{"step": 0, "argument": "style", "source_step": None, "literal": "ST-101"}],
    )
    other = _job(tool, plan=plan, task="How many units of style ZZ999999 are there?")
    assert _from_procedure(other, "style") is None
    same = _job(tool, plan=plan, task="How many units of style ST-101 are there?")
    assert _from_procedure(same, "style").value == "ST-101"  # type: ignore[union-attr]
    unnamed = _job(tool, plan=plan, task="How many units are there?")
    assert _from_procedure(unnamed, "style").value == "ST-101"  # type: ignore[union-attr]


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
    by_task.slots = _labelled_slots(by_task.task, ["email"])
    first, second = _from_task(by_task, "email"), _from_task(by_task, "email")
    assert (first.value, second.value) == ("bob@example.com", "ann@example.com")  # type: ignore[union-attr]
    assert _from_task(by_task, "email") is None, "each value fills one argument"
    question = _missing(tool, "amount").question
    assert "amount" in question and "update" in question


# ------------------------------------------------------------------ arguments from the task


def _create_po() -> ToolDescriptor:
    return ToolDescriptor(
        tenant_id="acme",
        name="erp-create_po",
        input_schema={
            "type": "object",
            "properties": {
                "supplier_id": {"type": "string"},
                "amount": {"type": "number"},
                "cost_centre": {"type": "string"},
            },
        },
        required=["supplier_id", "amount", "cost_centre"],
    )


async def test_a_value_fills_the_argument_the_words_before_it_name() -> None:
    """ "cost centre CC-7" is the cost centre: it never fills supplier_id because both look
    like identifiers, and supplier_id, which the task does not give, is asked for."""
    task = "Order 700 EUR of supplies from Acme Steel for cost centre CC-7"
    service = ToolHintsService(None, None, None)  # type: ignore[arg-type]
    prefill, missing = await service.arguments(_job(_create_po(), task=task))
    assert prefill["erp-create_po.cost_centre"].value == "CC-7"
    assert prefill["erp-create_po.amount"].value == 700, "the schema declares a number"
    assert "erp-create_po.supplier_id" not in prefill
    assert [(m.arg, m.question) for m in missing] == [
        ("supplier_id", "erp-create_po needs 'supplier id': what should it be?")
    ]


async def test_a_memory_that_names_the_value_fills_what_the_task_does_not() -> None:
    """The user said "their supplier id is SUP-40" earlier: the memory of it fills
    supplier_id. The task's own values still win: the memory's older cost centre does not."""

    class Said:
        item_id = "mem_said"
        attributes: dict[str, str] = {}
        text = "We buy from Acme Steel; their supplier id is SUP-40, cost centre CC-1."

    task = "Order 700 EUR of supplies from Acme Steel for cost centre CC-7"
    service = ToolHintsService(None, None, None)  # type: ignore[arg-type]
    prefill, missing = await service.arguments(_job(_create_po(), task=task, memories=[Said()]))
    supplier = prefill["erp-create_po.supplier_id"]
    assert (supplier.value, supplier.source, supplier.evidence_id) == (
        "SUP-40",
        "memory",
        "mem_said",
    )
    assert prefill["erp-create_po.cost_centre"].value == "CC-7" and not missing


def test_slots_are_labelled_by_the_argument_words_just_before_them() -> None:
    task = "Order 700 EUR of supplies from Acme Steel for cost centre CC-7"
    labelled = _labelled_slots(task, ["supplier_id", "amount", "cost_centre"])
    by_value = {value: label for _, value, label in labelled}
    assert by_value["CC-7"] == "cost_centre"
    assert by_value["Acme Steel"] is None, "a name is no supplier_id: free for one that takes it"
    assert by_value["700 EUR"] is None, "a currency after the amount is the amount's"
    assert "Order" not in by_value, "the word that opens the sentence is not a name"


def test_an_identifier_that_spells_its_argument_is_labelled_by_it() -> None:
    task = "update quote Q-1183 with EMEA price for SKU-22"
    labelled = _labelled_slots(task, ["quote_id", "sku", "region"])
    by_value = {value: label for _, value, label in labelled}
    assert by_value["SKU-22"] == "sku"
    assert by_value["Q-1183"] == "quote_id", "named for another argument, never the sku"


async def test_numbers_are_cast_only_for_number_arguments() -> None:
    tool = ToolDescriptor(
        tenant_id="acme",
        name="pay",
        input_schema={
            "properties": {
                "total": {"type": "number"},
                "count": {"type": "integer"},
                "ref": {"type": "string"},
            }
        },
        required=["total", "count", "ref"],
    )
    service = ToolHintsService(None, None, None)  # type: ignore[arg-type]
    prefill, _ = await service.arguments(
        _job(tool, task="pay EUR 1,200.50 total for quantity count 3 ref INV-42")
    )
    assert prefill["pay.total"].value == 1200.5
    assert prefill["pay.count"].value == 3
    assert prefill["pay.ref"].value == "INV-42", "a string argument keeps its text"


# ------------------------------------------------------------------ which tools, and the next step


def test_nothing_matched_and_no_plan_is_no_candidates() -> None:
    """B13: an index not built yet scores every tool 0: no candidate (they were the toolbox's
    first names in alphabetical order), and so no next step. A learned plan still leads."""
    found = [("billing-list", 0.0)]
    assert _candidates(found, None, None, {}, ["allocatable_stock", "app_help"], 5) == []
    plan = _procedure("p", ["store_demand"])
    ranked = _candidates([], plan, "store_demand", {}, ["store_demand", "app_help"], 5)
    assert [c.name for c in ranked][:1] == ["store_demand"]


def test_a_general_view_is_no_candidate_below_a_specific_tool() -> None:
    """S3: the dashboard ranked under the specific tools the task asks for is dropped; ranked
    first it is what the task asked for; a step of the learned plan always stays."""
    found = [("tickets_by_priority", 1.0), ("sla_breaches", 0.8), ("desk_dashboard", 0.6)]
    names = [n for n, _ in found]
    assert [c.name for c in _candidates(found, None, None, {}, names, 5)] == names[:2]
    first = [("desk_dashboard", 1.0), ("tickets_by_priority", 0.8)]
    assert _candidates(first, None, None, {}, names, 5)[0].name == "desk_dashboard"
    plan = _procedure("p", ["tickets_by_priority", "desk_dashboard"])
    kept = _candidates(found, plan, None, {}, names, 5)
    assert "desk_dashboard" in [c.name for c in kept]


class _HintsUow:
    def __init__(self, entries: list[ToolDescriptor]) -> None:
        self.entries = entries
        self.tools = self
        self.procedures = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def visible(self, *_: object, **__: object) -> list:
        return []

    async def catalog(self, tenant_id, *, workspace_id, names, limit):
        return [e for e in self.entries if e.name in names]

    async def stats(self, tenant_id, names) -> dict:
        return {}


class _Index:
    def __init__(self, found: list[tuple[str, float]]) -> None:
        self.found = found

    async def search(self, *_: object, **__: object) -> list[tuple[str, float]]:
        return self.found


async def test_a_tool_that_changes_something_is_no_next_step_for_a_question() -> None:
    """S3: "Before I create any stock rule: how many units ... a week?" ranks the rule first;
    the next step is the read that answers it. Asked to create one, the rule is next."""
    rule = ToolDescriptor(tenant_id="acme", name="create_stock_rule", side_effects="irreversible")
    demand = ToolDescriptor(tenant_id="acme", name="store_demand", side_effects="read")
    service = ToolHintsService(
        lambda: _HintsUow([rule, demand]),  # type: ignore[arg-type,return-value]
        _Index([("create_stock_rule", 1.0), ("store_demand", 0.8)]),  # type: ignore[arg-type]
        None,
    )
    ctx = MemoryExecutionContext(tenant_id="acme", user_id="u1")
    names = ["create_stock_rule", "store_demand"]
    asked = await service.hints(
        ctx,
        "Before I create any stock rule: how many units of ST-101 does S12 sell a week?",
        available=names,
        k=5,
        scope_keys=["k"],
    )
    assert asked.candidates[0].name == "create_stock_rule" and asked.next == "store_demand"
    told = await service.hints(
        ctx, "Create a stock rule for ST-101 at S12.", available=names, k=5, scope_keys=["k"]
    )
    assert told.next == "create_stock_rule"


# ------------------------------------------------------------------ which value, for which argument


def _described(name: str, **args: tuple[str, str]) -> ToolDescriptor:
    """A tool whose arguments are (type, description), all required."""
    return ToolDescriptor(
        tenant_id="acme",
        name=name,
        input_schema={
            "properties": {a: {"type": t, "description": d} for a, (t, d) in args.items()}
        },
        required=list(args),
    )


ASSET_OWNER = _described("asset_owner", asset_tag=("string", "The tag, such as LT-2231."))
RESET = _described("reset_password", user_id=("string", "The user's id, such as u-77."))
DEMAND = _described(
    "store_demand",
    store=("string", "The store's id, such as S12."),
    style=("string", "The style's id, such as ST-101."),
)
RULE = _described(
    "create_stock_rule",
    store=("string", "The store's id, such as S12."),
    style=("string", "The style's id, such as ST-101."),
    min_units=("integer", "The fewest units."),
    max_units=("integer", "The most units."),
)
TICKETS = _described(
    "tickets_by_priority",
    team=("string", "The team's name, such as network or desktop."),
    priority=("string", "high, medium or low."),
)


async def _hinted(task: str, *tools: ToolDescriptor) -> tuple[dict, dict]:
    service = ToolHintsService(None, None, None)  # type: ignore[arg-type]
    prefill, missing = await service._every_argument([_job(t, task=task) for t in tools])
    values = {k: p.value for k, p in prefill.items()}
    return values, {(m.tool, m.arg) for m in missing}


async def test_an_identifier_fills_only_an_argument_whose_own_words_it_matches() -> None:
    """S1: the asset tag of the task is the asset owner's, never the password reset's
    ``user_id`` because that argument asks for an identifier."""
    task = "Before anyone resets anything: who is laptop LT-2290 assigned to?"
    values, missing = await _hinted(task, ASSET_OWNER, RESET)
    assert values == {"asset_owner.asset_tag": "LT-2290"}
    assert ("asset_owner", "asset_tag") not in missing


async def test_a_value_of_the_argument_s_example_shape_fills_it() -> None:
    """S2: "S12" and "ST-101", named with no argument word before them, and the S7 of
    "Store S7", fill the arguments whose examples they look like: no false "missing"."""
    task = "Before I create any stock rule: how many units of ST-101 does S12 sell a week?"
    values, missing = await _hinted(task, DEMAND)
    assert values == {"store_demand.store": "S12", "store_demand.style": "ST-101"}
    assert not missing
    values, _ = await _hinted("Store S7: how is it doing overall?", DEMAND)
    assert values["store_demand.store"] == "S7"


async def test_a_number_is_never_read_out_of_an_identifier() -> None:
    """S1: "units of ST-101" is no max_units of -101."""
    task = "Before I create any stock rule: how many units of ST-101 does S12 sell a week?"
    values, missing = await _hinted(task, RULE)
    assert "create_stock_rule.max_units" not in values
    assert values["create_stock_rule.style"] == "ST-101"
    assert {("create_stock_rule", "min_units"), ("create_stock_rule", "max_units")} <= missing
    found = Prefill(tool="create_stock_rule", value="ST-101", source="task")
    assert _as_declared(found, RULE, "max_units").value == "ST-101"


async def test_described_free_text_is_the_model_s_to_word_never_missing() -> None:
    """S2: a team or a priority the request words ("the network team", "high-priority") is
    no value the hints can extract, so the described argument is not asked for as missing;
    an undescribed one still is."""
    task = "Which high-priority tickets does the network team have open?"
    _, missing = await _hinted(task, TICKETS)
    assert missing == set()
    approve = ToolDescriptor(
        tenant_id="acme",
        name="approve",
        input_schema={"properties": {"approver": {"type": "string"}}},
        required=["approver"],
    )
    _, missing = await _hinted(task, approve)
    assert missing == {("approve", "approver")}
