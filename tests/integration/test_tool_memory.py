"""Tool memory against PostgreSQL: the catalog, idempotent recording and its statistics, the
learning job (stored procedures, distillation, graph edges) and the hints read back."""

from __future__ import annotations

import json

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Visibility
from memory_service.domain.feedback import Feedback, FeedbackTargetKind, FeedbackVerdict
from memory_service.domain.graph import IDENTIFIED_BY, USED_ENTITY
from memory_service.domain.tools import ToolDescriptor
from memory_service.modules.tools.learning import ToolLearning
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration

PRICING = "pricing-lookup_price"
CRM = "crm-update_quote"
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


async def _keys(container, ctx: MemoryExecutionContext) -> list[str]:
    return list((await container.services["authz"].visibility(ctx)).keys)


async def _run_once(container, *, run: str, quote: str, sku: str, success: bool = True) -> None:
    """One two-step run: look the price up, then write it onto the quote. The quote id flows
    from the first output into the second call's arguments."""
    service = container.services["tool_memory"]
    ctx = _ctx(run)
    async with container.services["uow_factory"]() as uow:
        await service.record(
            uow,
            ctx,
            tool=PRICING,
            args={"sku": sku, "region": "EMEA"},
            output={"price": 1200, "currency": "EUR", "quote_id": quote},
            task=TASK,
            step=0,
            latency_ms=40.0,
        )
        await service.record(
            uow,
            ctx,
            tool=CRM,
            args={"quote_id": quote, "amount": 1200},
            output={"ok": True},
            status="ok" if success else "error",
            error_class=None if success else "QuoteLocked",
            task=TASK,
            step=1,
            latency_ms=90.0,
        )
        await service.set_outcome(uow, ctx, run_id=run, success=success)
        await uow.commit()


async def _procedures(container, ctx: MemoryExecutionContext):
    return await container.services["tool_hints"].procedures(
        ctx, TASK, await _keys(container, ctx), k=3
    )


async def test_recording_is_idempotent_and_counted_once(container, uow_factory) -> None:
    service = container.services["tool_memory"]
    ctx = _ctx("run_retry")
    async with uow_factory() as uow:
        first, created = await service.record(
            uow, ctx, tool=PRICING, args={"sku": "A"}, output={"price": 1}, step=0, latency_ms=10
        )
        second, again = await service.record(
            uow, ctx, tool=PRICING, args={"sku": "A"}, output={"price": 1}, step=0, latency_ms=10
        )
        await uow.commit()
        rows = await uow.tools.invocations_for_run("acme", "run_retry")
        stats = await uow.tools.stats("acme", [PRICING])
    assert (created, again) == (True, False) and first.invocation_id == second.invocation_id
    assert len(rows) == 1
    assert stats[PRICING].calls == 1 and stats[PRICING].avg_latency_ms == 10.0


async def test_the_catalog_versions_a_changed_schema_and_a_workspace_shadows_the_tenant(
    container, uow_factory
) -> None:
    service = container.services["tool_memory"]
    tenant_wide = _ctx(workspace_id=None)
    entry = ToolDescriptor(
        tenant_id="",
        name=PRICING,
        input_schema={"type": "object", "properties": {"sku": {"type": "string"}}},
        side_effects="read",
    )
    async with uow_factory() as uow:
        [first] = await service.put_catalog(uow, tenant_wide, [entry])
        [same] = await service.put_catalog(uow, tenant_wide, [entry])
        changed = entry.model_copy(update={"input_schema": {"type": "object", "properties": {}}})
        [second] = await service.put_catalog(uow, tenant_wide, [changed])
        [own] = await service.put_catalog(
            uow, _ctx(), [entry.model_copy(update={"side_effects": "write"})]
        )
        await uow.commit()
        in_workspace = await uow.tools.by_name("acme", PRICING, workspace_id="ws1")
        elsewhere = await uow.tools.by_name("acme", PRICING, workspace_id="ws2")
    assert (first.version, same.version, second.version) == (1, 1, 2)
    assert own.workspace_id == "ws1"
    assert in_workspace is not None and in_workspace.side_effects == "write"
    assert elsewhere is not None and elsewhere.side_effects == "read"


async def test_the_learning_job_stores_a_procedure_only_once_enough_runs_support_it(
    container, uow_factory
) -> None:
    learning: ToolLearning = container.services["tool_learning"]
    ctx = _ctx()
    await _run_once(container, run="run_0", quote="Q-1", sku="S-1")
    await learning.learn()
    assert await _procedures(container, ctx) == [], "one run is not a procedure"
    async with uow_factory() as uow:
        audience = (await uow.tools.invocations_for_run("acme", "run_0"))[0].visibility_keys[0]
        candidate = await uow.procedures.by_pattern("acme", audience, _pattern())
    assert candidate is not None and candidate.status == "candidate"

    await _run_once(container, run="run_1", quote="Q-2", sku="S-2")
    assert await learning.learn() == 2
    [procedure] = await _procedures(container, ctx)
    assert procedure.tools == [PRICING, CRM] and procedure.support == 2
    assert procedure.status == "active" and procedure.procedure_id == candidate.procedure_id
    # without a model the title is the pattern and the strategy the miner's rendering
    assert procedure.title == _pattern() and PRICING in procedure.strategy
    binding = next(b for b in procedure.bindings if b["argument"] == "quote_id")
    assert binding["step"] == 1 and binding["source_step"] == 0
    assert await learning.learn() == 0, "everything was learned"


async def test_relabelling_a_run_relearns_its_pattern(container, uow_factory) -> None:
    learning = container.services["tool_learning"]
    for i in range(2):
        await _run_once(container, run=f"run_{i}", quote=f"Q-{i}", sku=f"S-{i}")
    await learning.learn()
    ctx = _ctx()
    assert await _procedures(container, ctx)
    for i in range(2):
        async with uow_factory() as uow:
            await container.services["tool_memory"].set_outcome(
                uow, _ctx(f"run_{i}"), run_id=f"run_{i}", success=False
            )
            await uow.commit()
    assert await learning.learn() == 4
    assert await _procedures(container, ctx) == [], "no successful run: retired"


async def test_a_procedure_is_distilled_from_its_successes_and_failures(
    container, uow_factory
) -> None:
    for i in range(3):
        await _run_once(container, run=f"run_{i}", quote=f"Q-{i}", sku=f"S-{i}")
    await _run_once(container, run="run_bad", quote="Q-9", sku="S-9", success=False)
    reply = {
        "title": "Reprice a quote",
        "strategy": "Look the price up, then write it onto the quote the lookup names.",
        "avoid": "Do not update a locked quote.",
    }
    with mocked_gateway([json.dumps(reply)]) as gateway:
        assist = gateway.assist(uses=["procedure_abstraction"])
        learning = ToolLearning(uow_factory, assist, container.graph_store)
        await learning.learn()
        prompt = gateway.prompts()[0]["messages"][1]["content"]
    [procedure] = await _procedures(container, _ctx())
    assert procedure.title == "Reprice a quote"
    assert procedure.strategy.endswith("Avoid: Do not update a locked quote.")
    assert "Runs that failed:" in prompt and "QuoteLocked" in prompt
    assert procedure.distilled == procedure.steps_hash


async def test_another_agent_s_private_calls_teach_nothing_it_can_read(
    container, uow_factory
) -> None:
    for i in range(2):
        await _run_once(container, run=f"run_{i}", quote=f"Q-{i}", sku=f"S-{i}")
    await container.services["tool_learning"].learn()
    assert await _procedures(container, _ctx())
    assert await _procedures(container, _ctx("run_x", agent_id="rival-agent")) == []
    user_only = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
    assert await _procedures(container, user_only) == []


async def test_a_rejected_procedure_is_not_offered_until_its_steps_change(
    container, uow_factory
) -> None:
    for i in range(2):
        await _run_once(container, run=f"run_{i}", quote=f"Q-{i}", sku=f"S-{i}")
    learning = container.services["tool_learning"]
    await learning.learn()
    [procedure] = await _procedures(container, _ctx())
    feedback = container.services["feedback"]
    async with uow_factory() as uow:
        await feedback.submit(
            uow,
            _ctx(),
            Feedback(
                tenant_id="acme",
                target_kind=FeedbackTargetKind.PROCEDURE,
                target_id=procedure.procedure_id,
                verdict=FeedbackVerdict.REJECT,
            ),
        )
        await uow.commit()
    await container.tasks.drain()
    assert await _procedures(container, _ctx()) == []
    await _run_once(container, run="run_2", quote="Q-2", sku="S-2")
    await learning.learn()
    assert await _procedures(container, _ctx()) == [], "same steps: still rejected"


async def test_redacted_arguments_never_reach_storage(container, uow_factory) -> None:
    service = container.services["tool_memory"]
    ctx = _ctx("run_secret")
    async with uow_factory() as uow:
        await service.put_catalog(
            uow, ctx, [ToolDescriptor(tenant_id="", name=CRM, redact=["auth.token"])]
        )
        stored, _ = await service.record(
            uow,
            ctx,
            tool=CRM,
            args={"quote_id": "Q-1", "auth": {"token": "super-secret-value"}},
            output={"ok": True},
            step=0,
        )
        await uow.commit()
        rows = await uow.tools.invocations_for_run("acme", "run_secret")
    assert stored.args_redacted["auth"]["token"] == "[redacted]"
    assert "super-secret-value" not in str(rows[0].args_redacted)
    assert rows[0].args_hash != ""


async def test_a_typed_argument_links_the_call_to_the_entity_and_its_id(
    container, uow_factory
) -> None:
    """``supplier: ORG`` makes the call ``used_entity`` Acme, and the id the lookup returned
    for it ``identified_by``; the hint for a later task naming Acme fills the id in."""
    service = container.services["tool_memory"]
    ctx = _ctx("run_supplier")
    lookup = ToolDescriptor(
        tenant_id="",
        name="erp-find_supplier",
        input_schema={"type": "object", "properties": {"supplier": {"type": "string"}}},
        argument_entity_types={"supplier": "ORG"},
        side_effects="read",
    )
    order = ToolDescriptor(
        tenant_id="",
        name="erp-create_po",
        input_schema={
            "type": "object",
            "properties": {"supplier_id": {"type": "string"}, "amount": {"type": "number"}},
        },
        required=["supplier_id", "amount"],
        argument_entity_types={"supplier_id": "ORG"},
    )
    async with uow_factory() as uow:
        await service.put_catalog(uow, ctx, [lookup, order])
        await service.record(
            uow,
            ctx,
            tool="erp-find_supplier",
            args={"supplier": "Acme Paper"},
            output={"supplier_id": "SUP-42", "name": "Acme Paper"},
            task="find Acme Paper",
            visibility=Visibility.PRIVATE,
        )
        await uow.commit()
    await container.tasks.drain()  # tools.index
    await container.services["tool_learning"].learn()
    keys = await _keys(container, ctx)
    graph = container.graph_store
    [acme] = [
        e
        for e in await graph.find_entities("acme", ["acme paper"], scope_keys=keys)
        if e.entity_type == "ORG"
    ]
    relations = await graph.entity_relations(
        "acme", acme.entity_id, scope_keys=keys, current=True, limit=10
    )
    assert {r.predicate for r in relations} == {USED_ENTITY, IDENTIFIED_BY}
    assert all(r.layer == "procedural" for r in relations)

    hints = await container.services["tool_hints"].hints(
        ctx, "order 500 sheets from Acme Paper", available=["erp-create_po"], k=3, scope_keys=keys
    )
    assert hints.next == "erp-create_po"
    assert hints.prefill["supplier_id"].value == "SUP-42"
    assert hints.prefill["supplier_id"].source == "graph"
    assert hints.prefill["amount"].value == "500" and hints.prefill["amount"].source == "task"
    assert hints.missing == []


def _pattern() -> str:
    from memory_service.modules.tools.patterns import task_pattern

    return task_pattern(TASK)
