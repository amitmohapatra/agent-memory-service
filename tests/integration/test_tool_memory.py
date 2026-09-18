"""Tool memory end to end (TOOL_MEMORY.md §30.0-§30.6): registry defaults, idempotent
recording, the replay cache and its refusals, chain mining, and the advice built from them."""

from __future__ import annotations

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Visibility
from memory_service.domain.tools import ToolDescriptor, ToolPolicy

pytestmark = pytest.mark.integration

PRICING = {"name": "pricing.lookup_price", "description": "list price for a SKU"}
CRM = {"name": "crm.update_quote", "description": "write a price onto a quote"}
DECLARED = [PRICING, CRM]
TASK = "update quote Q-1183 with EMEA price for SKU-22"


def _ctx(run: str = "run_1", **kw) -> MemoryExecutionContext:
    base = {
        "tenant_id": "acme",
        "user_id": "u1",
        "workspace_id": "ws1",
        "thread_id": "thr_1",
        "agent_id": "pricing-agent",
        "agent_run_id": run,
    }
    base.update(kw)
    return MemoryExecutionContext(**base)


async def _run_once(container, service, *, run: str, task: str, quote: str, sku: str) -> None:
    """One successful two-step run: look the price up, then write it onto the quote. The quote
    id flows from the first output into the second call's arguments, which is the data flow the
    miner must find."""
    ctx = _ctx(run)
    async with container.services["uow_factory"]() as uow:
        await service.record(
            uow,
            ctx,
            tool="pricing.lookup_price",
            args={"sku": sku, "region": "EMEA"},
            output={"price": 1200, "currency": "EUR", "quote_id": quote},
            task=task,
            step=0,
            latency_ms=40.0,
        )
        await service.record(
            uow,
            ctx,
            tool="crm.update_quote",
            args={"quote_id": quote, "amount": 1200},
            output={"ok": True},
            task=task,
            step=1,
            latency_ms=90.0,
        )
        await service.set_outcome(uow, ctx, run_id=run, success=True)
        await uow.commit()


async def test_unregistered_tool_gets_the_conservative_policy(container) -> None:
    service = container.services["tool_memory"]
    ctx = _ctx()
    async with container.services["uow_factory"]() as uow:
        declared = await service.declare(uow, ctx, DECLARED)
        await uow.commit()
    policy = declared["pricing.lookup_price"].policy
    assert policy.deterministic is False
    assert policy.cacheable is False
    assert policy.side_effects == "unknown"
    assert policy.replayable is False


async def test_declaration_never_widens_a_policy(container) -> None:
    """An agent claiming a tool is cacheable must not make it so; only an admin registration can."""
    service = container.services["tool_memory"]
    ctx = _ctx()
    async with container.services["uow_factory"]() as uow:
        await service.declare(uow, ctx, DECLARED)
        await service.register(
            uow,
            ctx,
            ToolDescriptor(
                tenant_id="acme",
                name="pricing.lookup_price",
                policy=ToolPolicy(deterministic=True, cacheable=True, side_effects="read"),
            ),
            widen_policy=False,
        )
        await uow.commit()
        stored = await uow.tools.by_name("acme", "pricing.lookup_price")
    assert stored is not None and stored.policy.replayable is False


async def test_recording_is_idempotent_across_retries(container) -> None:
    service = container.services["tool_memory"]
    ctx = _ctx("run_retry")
    async with container.services["uow_factory"]() as uow:
        first = await service.record(
            uow, ctx, tool="pricing.lookup_price", args={"sku": "A"}, output={"price": 1}, step=0
        )
        second = await service.record(
            uow, ctx, tool="pricing.lookup_price", args={"sku": "A"}, output={"price": 1}, step=0
        )
        await uow.commit()
        rows = await uow.tools.invocations_for_run("acme", "run_retry")
    assert first.invocation_id == second.invocation_id
    assert len(rows) == 1


async def test_a_procedure_is_mined_from_successful_runs(container) -> None:
    """Three successful runs of one task pattern produce a validated chain.

    The advice endpoints that used to read this — suggest and next — are gone: modern models
    plan tool use better than a support count can. What the model cannot know is what worked
    here before, and that is what the procedure carries.
    """
    service = container.services["tool_memory"]
    ctx = _ctx()
    keys = list((await container.services["authz"].visibility(ctx)).keys)
    for index, (quote, sku) in enumerate([("Q-1", "S-1"), ("Q-2", "S-2"), ("Q-3", "S-3")]):
        await _run_once(container, service, run=f"run_{index}", task=TASK, quote=quote, sku=sku)

    async with container.services["uow_factory"]() as uow:
        procedures = await service.procedures(uow, ctx, task=TASK, scope_keys=keys)
        plan = await service.plan(uow, ctx, task=TASK, available_tools=DECLARED, scope_keys=keys)
        await uow.commit()

    assert procedures, "three successful runs of one pattern must yield a procedure"
    procedure = procedures[0]
    assert procedure.tools == ["pricing.lookup_price", "crm.update_quote"]
    # the data flow was mined: the second step's quote_id comes from the first step's output
    second = procedure.steps[1]
    bound = [b for b in second.bindings if b.resolvable_from_trajectory]
    assert any(b.argument == "quote_id" and b.source_step == 0 for b in bound), second.bindings
    assert plan["valid"] is True and len(plan["steps"]) == 2
    assert "pricing" in plan["script"]
    assert plan["support"] == 3 and plan["success_rate"] == 1.0


async def test_a_plan_never_names_a_tool_the_caller_cannot_call(container) -> None:
    service = container.services["tool_memory"]
    ctx = _ctx()
    keys = list((await container.services["authz"].visibility(ctx)).keys)
    await _run_once(container, service, run="run_x", task=TASK, quote="Q-1", sku="S-1")
    async with container.services["uow_factory"]() as uow:
        plan = await service.plan(uow, ctx, task=TASK, available_tools=[PRICING], scope_keys=keys)
        await uow.commit()
    assert plan["valid"] is False and "did not declare" in plan["reason"]


async def test_another_agents_unshared_calls_are_invisible(container) -> None:
    """A run-scoped record belongs to its run: another agent, and the user, must not see it."""
    service = container.services["tool_memory"]
    owner = _ctx("run_private", agent_id="agent-a")
    async with container.services["uow_factory"]() as uow:
        await service.record(
            uow,
            owner,
            tool="pricing.lookup_price",
            args={"sku": "SECRET"},
            output={"price": 1},
            task=TASK,
            step=0,
            visibility=Visibility.RUN,
        )
        await uow.commit()

    other = _ctx("run_other", agent_id="agent-b")
    other_keys = list((await container.services["authz"].visibility(other)).keys)
    user_ctx = MemoryExecutionContext(
        tenant_id="acme", user_id="u1", workspace_id="ws1", thread_id="thr_1"
    )
    user_keys = list((await container.services["authz"].visibility(user_ctx)).keys)
    async with container.services["uow_factory"]() as uow:
        seen_by_agent = await uow.tools.recent("acme", scope_keys=other_keys)
        seen_by_user = await uow.tools.recent("acme", scope_keys=user_keys)
    assert all(i.args_redacted.get("sku") != "SECRET" for i in seen_by_agent)
    assert all(i.args_redacted.get("sku") != "SECRET" for i in seen_by_user)


async def test_redacted_arguments_never_reach_storage(container) -> None:
    service = container.services["tool_memory"]
    ctx = _ctx("run_secret")
    async with container.services["uow_factory"]() as uow:
        await service.register(
            uow,
            ctx,
            ToolDescriptor(
                tenant_id="acme",
                name="crm.update_quote",
                policy=ToolPolicy(redact=["auth.token"]),
            ),
            widen_policy=True,
        )
        stored = await service.record(
            uow,
            ctx,
            tool="crm.update_quote",
            args={"quote_id": "Q-1", "auth": {"token": "super-secret-value"}},
            output={"ok": True},
            step=0,
        )
        await uow.commit()
        rows = await uow.tools.invocations_for_run("acme", "run_secret")
    assert stored.args_redacted["auth"]["token"] == "[redacted]"
    assert "super-secret-value" not in str(rows[0].args_redacted)
    # the hash still covers the real value, so two different tokens are different calls
    assert rows[0].args_hash != ""
