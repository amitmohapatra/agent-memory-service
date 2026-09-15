"""Tool memory gate (TOOL_MEMORY.md §30.7).

Replays ``tests/fixtures/tool_trajectories.json`` through the real service and measures what
the advice is actually worth. The held-out runs of each pattern are never replayed, so the
suggestion and next-step numbers are measured on trajectories the miner has not seen.

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
MAX_CACHE_VIOLATIONS = 0
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

    # --- suggestion hit rate: is the first tool of the real trajectory suggested first? ---
    suggestion_hits = 0
    undeclared = 0
    for run in evaluation:
        expected = run["invocations"][0]["tool"]
        declared = _declared({c["tool"] for c in run["invocations"]})
        async with uow_factory() as uow:
            suggestions = await service.suggest(
                uow, ctx, task=run["task"], available_tools=declared, scope_keys=keys
            )
            await uow.commit()
        names = {s.tool for s in suggestions}
        undeclared += len(names - {d["name"] for d in declared})
        if suggestions and suggestions[0].tool == expected:
            suggestion_hits += 1
    suggestion_rate = suggestion_hits / len(evaluation)

    # --- next-step hit rate over every prefix of every held-out trajectory ---
    next_total = next_hits = 0
    for run in evaluation:
        successful = [c for c in run["invocations"] if c.get("status") == "ok"]
        declared = _declared({c["tool"] for c in run["invocations"]})
        for cut in range(1, len(successful)):
            prefix = [
                {
                    "tool": c["tool"],
                    "status": "ok",
                    "output_fields": c.get("output") or {},
                }
                for c in successful[:cut]
            ]
            expected = successful[cut]["tool"]
            async with uow_factory() as uow:
                nxt = await service.next_step(
                    uow,
                    ctx,
                    task=run["task"],
                    trajectory_so_far=prefix,
                    available_tools=declared,
                    scope_keys=keys,
                )
                await uow.commit()
            next_total += 1
            if nxt.suggestions and nxt.suggestions[0].tool == expected:
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

    # --- cache violations: a non-replayable tool must never be served from cache, and a
    #     run-scoped entry must never leak into another run ---
    cache_violations = 0
    async with uow_factory() as uow:
        await service.register(
            uow,
            ctx,
            ToolDescriptor(
                tenant_id="acme",
                name="pricing.lookup_price",
                policy=ToolPolicy(
                    deterministic=True, cacheable=True, side_effects="read", cache_scope="run"
                ),
            ),
            widen_policy=True,
        )
        await uow.commit()
    probe_ctx = _ctx("run_cache_probe")
    async with uow_factory() as uow:
        await service.record(
            uow,
            probe_ctx,
            tool="pricing.lookup_price",
            args={"sku": "GATE-1"},
            output={"price": 1},
            step=0,
        )
        await uow.commit()
        if not (
            await service.lookup(
                uow, probe_ctx, tool="pricing.lookup_price", args={"sku": "GATE-1"}
            )
        )["cached"]:
            cache_violations += 1  # a replayable tool that does not hit is a miss, not a leak
        other = await service.lookup(
            uow, _ctx("run_cache_other"), tool="pricing.lookup_price", args={"sku": "GATE-1"}
        )
        if other["cached"]:
            cache_violations += 1  # cross-scope hit
        for name in ("crm.update_quote", "support.escalate"):
            leaked = await service.lookup(uow, probe_ctx, tool=name, args={"x": 1})
            if leaked["cached"]:
                cache_violations += 1  # non-replayable tool served from cache

    # --- isolation: another agent's unshared calls must not be visible or suggestible ---
    rival_tools = {
        c["tool"] for r in runs if r.get("agent_id") == "rival-agent" for c in r["invocations"]
    }
    isolation_violations = 0
    async with uow_factory() as uow:
        visible = await uow.tools.recent("acme", scope_keys=keys, limit=500)
        isolation_violations += sum(1 for i in visible if i.tool_name in rival_tools)
        suggestions = await service.suggest(
            uow,
            ctx,
            task="update quote Q-9990 with EMEA price for SKU-990",
            available_tools=_declared(all_tools),
            scope_keys=keys,
        )
        await uow.commit()
    isolation_violations += sum(1 for s in suggestions if s.tool in rival_tools)

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
        "cache_violations": cache_violations,
        "isolation_violations": isolation_violations,
        "undeclared_tool_suggestions": undeclared,
        "thresholds": {
            "suggestion_hit_rate": SUGGESTION_HIT_RATE_MIN,
            "next_step_hit_rate": NEXT_STEP_HIT_RATE_MIN,
            "plan_validity": PLAN_VALIDITY_MIN,
            "cache_violations": MAX_CACHE_VIOLATIONS,
            "isolation_violations": MAX_ISOLATION_VIOLATIONS,
            "undeclared_tool_suggestions": MAX_UNDECLARED_SUGGESTIONS,
        },
        "provenance": provenance(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "tool_gate.json").write_text(json.dumps(report, indent=2) + "\n")

    assert undeclared == MAX_UNDECLARED_SUGGESTIONS, report
    assert isolation_violations == MAX_ISOLATION_VIOLATIONS, report
    assert cache_violations == MAX_CACHE_VIOLATIONS, report
    assert plan_validity >= PLAN_VALIDITY_MIN, report
    assert suggestion_rate >= SUGGESTION_HIT_RATE_MIN, report
    assert next_rate >= NEXT_STEP_HIT_RATE_MIN, report
