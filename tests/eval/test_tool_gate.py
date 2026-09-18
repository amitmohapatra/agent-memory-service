"""Tool memory gate (TOOL_MEMORY.md §30.7).

Replays ``tests/fixtures/tool_trajectories.json`` through the real service and measures what
the advice is actually worth. The held-out runs of each pattern are never replayed, so the
first-step and next-step numbers are measured on trajectories the miner has not seen,
read straight off the mined procedure rather than through an advice endpoint.

Thresholds are hard, like every other gate. Writes ``benchmark/results/tool_gate.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmark.common import RESULTS, provenance

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Visibility
from memory_service.domain.tools import ToolDescriptor, ToolPolicy

pytestmark = pytest.mark.eval

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tool_trajectories.json"

SUGGESTION_HIT_RATE_MIN = 0.95
NEXT_STEP_HIT_RATE_MIN = 0.90
PLAN_VALIDITY_MIN = 1.0
MAX_ISOLATION_VIOLATIONS = 0
MAX_UNDECLARED_SUGGESTIONS = 0


def _ctx(run_id: str, agent_id: str = "ops-agent") -> MemoryExecutionContext:
    return MemoryExecutionContext(
        tenant_id="acme",
        user_id="u1",
        workspace_id="ws1",
        thread_id="thr_gate",
        agent_id=agent_id,
        agent_run_id=run_id,
    )


def _declared(tools: set[str]) -> list[dict[str, str]]:
    return [{"name": t} for t in sorted(tools)]


async def _replay(container, service, run: dict) -> None:
    ctx = _ctx(run["run_id"], run.get("agent_id", "ops-agent"))
    async with container.services["uow_factory"]() as uow:
        for call in run["invocations"]:
            await service.record(
                uow,
                ctx,
                tool=call["tool"],
                args=call["args"],
                output=call.get("output"),
                status=call.get("status", "ok"),
                error_class=call.get("error_class"),
                latency_ms=call.get("latency_ms"),
                cost=call.get("cost"),
                task=run["task"],
                step=call["step"],
                visibility=Visibility.RUN,
            )
        await service.set_outcome(uow, ctx, run_id=run["run_id"], success=run["success"])
        await uow.commit()


async def test_tool_gate(container, uow_factory) -> None:
    payload = json.loads(FIXTURE.read_text())
    runs = payload["runs"]
    held_out = set(payload["held_out"])
    service = container.services["tool_memory"]
    ctx = _ctx("run_probe")
    keys = list((await container.services["authz"].visibility(ctx)).keys)

    training = [r for r in runs if r["run_id"] not in held_out]
    evaluation = [r for r in runs if r["run_id"] in held_out]
    for run in training:
        await _replay(container, service, run)

    all_tools = {c["tool"] for r in runs for c in r["invocations"]}
    ops_tools = {
        c["tool"] for r in runs if r.get("agent_id") == "ops-agent" for c in r["invocations"]
    }

    # --- first-step hit rate: does the mined procedure open the way the real run did? ---
    #
    # This used to be measured through service.suggest() and service.next_step(). Those
    # endpoints are gone — the model plans better than a support count can — but the signal
    # they measured is procedure *quality*, and that is a property of the mined procedure
    # itself. Reading it straight off the procedure keeps the measurement and drops the API.
    async def _procedure(task: str):
        async with uow_factory() as uow:
            found = await service.procedures(uow, ctx, task=task, scope_keys=keys)
            await uow.commit()
        return found[0] if found else None

    suggestion_hits = 0
    undeclared = 0
    for run in evaluation:
        expected = run["invocations"][0]["tool"]
        declared = {d["name"] for d in _declared({c["tool"] for c in run["invocations"]})}
        procedure = await _procedure(run["task"])
        if procedure is None or not procedure.steps:
            continue
        undeclared += len({s.tool for s in procedure.steps} - declared)
        if procedure.steps[0].tool == expected:
            suggestion_hits += 1
    suggestion_rate = suggestion_hits / len(evaluation)

    # --- next-step hit rate over every prefix of every held-out trajectory ---
    next_total = next_hits = 0
    for run in evaluation:
        successful = [c for c in run["invocations"] if c.get("status") == "ok"]
        procedure = await _procedure(run["task"])
        steps = [s.tool for s in procedure.steps] if procedure else []
        for cut in range(1, len(successful)):
            expected = successful[cut]["tool"]
            next_total += 1
            # the procedure's own continuation after the same prefix
            if len(steps) > cut and steps[cut] == expected:
                next_hits += 1
    next_rate = next_hits / next_total if next_total else 0.0

    # --- plan validity: every returned plan's bindings must resolve ---
    plans = valid_plans = 0
    for run in evaluation:
        declared = _declared({c["tool"] for c in run["invocations"]})
        async with uow_factory() as uow:
            plan = await service.plan(
                uow, ctx, task=run["task"], available_tools=declared, scope_keys=keys
            )
            await uow.commit()
        if not plan.get("steps"):
            continue
        plans += 1
        resolvable = all(
            b.get("source_step") is None or b["source_step"] < step["ordinal"]
            for step in plan["steps"]
            for b in step["bindings"]
        )
        if plan["valid"] and resolvable and not plan.get("problems"):
            valid_plans += 1
    plan_validity = valid_plans / plans if plans else 0.0

    # --- isolation: another agent's unshared calls must not be visible or suggestible ---
    rival_tools = {
        c["tool"] for r in runs if r.get("agent_id") == "rival-agent" for c in r["invocations"]
    }
    isolation_violations = 0
    async with uow_factory() as uow:
        visible = await uow.tools.recent("acme", scope_keys=keys, limit=500)
        isolation_violations += sum(1 for i in visible if i.tool_name in rival_tools)
        mined = await service.procedures(
            uow,
            ctx,
            task="update quote Q-9990 with EMEA price for SKU-990",
            scope_keys=keys,
        )
        await uow.commit()
    # a mined procedure must never name a tool this caller could not see being used
    isolation_violations += sum(
        1 for procedure in mined for step in procedure.steps if step.tool in rival_tools
    )

    report = {
        "gate": "tool_memory",
        "fixture": payload["name"],
        "runs_replayed": len(training),
        "runs_held_out": len(evaluation),
        "patterns": len({r["pattern"] for r in runs}),
        "tools_seen": sorted(ops_tools),
        "suggestion_hit_rate": round(suggestion_rate, 4),
        "next_step_hit_rate": round(next_rate, 4),
        "next_step_cases": next_total,
        "plan_validity": round(plan_validity, 4),
        "plans_returned": plans,
        "isolation_violations": isolation_violations,
        "undeclared_tool_suggestions": undeclared,
        "thresholds": {
            "suggestion_hit_rate": SUGGESTION_HIT_RATE_MIN,
            "next_step_hit_rate": NEXT_STEP_HIT_RATE_MIN,
            "plan_validity": PLAN_VALIDITY_MIN,
            "isolation_violations": MAX_ISOLATION_VIOLATIONS,
            "undeclared_tool_suggestions": MAX_UNDECLARED_SUGGESTIONS,
        },
        "provenance": provenance(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "tool_gate.json").write_text(json.dumps(report, indent=2) + "\n")

    assert undeclared == MAX_UNDECLARED_SUGGESTIONS, report
    assert isolation_violations == MAX_ISOLATION_VIOLATIONS, report
    assert plan_validity >= PLAN_VALIDITY_MIN, report
    assert suggestion_rate >= SUGGESTION_HIT_RATE_MIN, report
    assert next_rate >= NEXT_STEP_HIT_RATE_MIN, report
