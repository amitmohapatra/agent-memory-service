"""The README's examples, executed.

A README that drifts from the API is worse than no README, so every snippet a reader is
invited to copy runs here against the real service. If you change the SDK surface and this
fails, fix the README in the same commit.
"""

from __future__ import annotations

import pytest

from tests.e2e.conftest import sdk_client
from universal_memory import ToolCall

pytestmark = pytest.mark.e2e

NOTE = "Source A contradicts source B"
TASK = "update quote Q-1183 with EMEA price for SKU-22"
TOOLS = [{"name": "pricing.lookup_price"}, {"name": "crm.update_quote"}]


def _bind(memory, **over):
    scope = {
        "tenant_id": "acme",
        "workspace_id": "ws1",
        "user_id": "u1",
        "thread_id": "thr-readme",
        "session_id": "ses-readme",
        "turn_id": "trn-readme-1",
    }
    scope.update(over)
    return memory.bind(**scope)


async def test_readme_single_agent_walkthrough(app, client) -> None:
    """README: Use it in one agent — chat, facts, context bundle, evidence gating."""
    memory = sdk_client(app)
    ctx = _bind(memory)

    await ctx.chat.user("Revenue was EUR 412 million in FY26.")
    await ctx.chat.assistant("Noted — that's up 4% year on year.")
    await ctx.remember("Prefers metric units", memory_type="PREFERENCE")
    await ctx.observe("User cancelled the Pro plan", kind="EVENT")

    bundle = await ctx.context("how did revenue develop?")
    # the attributes the README tells readers to inspect
    assert isinstance(bundle.rendered, str) and bundle.rendered
    for attr in ("conversation", "memories", "knowledge", "graph_facts", "evidence"):
        assert hasattr(bundle, attr), attr
    assert bundle.evidence.status in {"COMPLETE", "PARTIAL", "INSUFFICIENT_EVIDENCE"}

    assert await ctx.recall("revenue") != []
    assert await ctx.memories() != []

    gated = await ctx.context("what were FY26 restructuring savings?", require_evidence=True)
    assert gated.evidence.status  # the README branches on this value


async def test_readme_multi_agent_visibility(app, client) -> None:
    """README: Use it with multiple agents — a RUN note reaches the owner and the run it
    spawns, and nobody else. This is the claim the section is built on."""
    memory = sdk_client(app)
    ctx = _bind(memory)
    researcher = ctx.agent("researcher", agent_group_id="analysis-crew")
    writer = ctx.agent("writer", agent_group_id="analysis-crew")
    await researcher.remember(NOTE, visibility="RUN")
    child = researcher.agent("fact-checker")

    async def sees(scope) -> bool:
        return any(NOTE in (i.text or "") for i in await scope.recall(NOTE, kinds=["memory"]))

    assert await sees(researcher), "the owning run must see its own note"
    assert await sees(child), "a spawned child must receive the hand-off"
    assert not await sees(writer), "a sibling agent must not see RUN-scoped notes"
    assert not await sees(ctx), "the user must not see agent notes"

    shared = "FY26 revenue is EUR 412m, confirmed in two sources"
    await researcher.remember(shared, memory_type="SHARED", visibility="AGENT_GROUP")

    # sharing is only meaningful if the crew can actually read it: the sibling that saw
    # nothing of the RUN note must see this one
    async def sees_shared(scope) -> bool:
        return any(shared in (i.text or "") for i in await scope.recall(shared, kinds=["memory"]))

    assert await sees_shared(writer), "an AGENT_GROUP memory must reach the crew"


async def test_readme_tool_memory_walkthrough(app, client) -> None:
    """README: Tool memory — record what ran, label the run, and the service mines the
    procedure. Three endpoints, because the advice endpoints it used to have (register,
    lookup, suggest, next) were removed: the model plans better than a support count, every
    framework already owns a tool catalogue, and an output cache replays stale results."""
    memory = sdk_client(app)
    ctx = _bind(memory)
    agent = ctx.agent("ops", agent_run_id="run_readme_1")

    for i in range(3):
        run = ctx.agent("ops", agent_run_id=f"run_readme_{i}")
        await run.tools.record(
            "pricing.lookup_price",
            args={"sku": f"SKU-{i}", "region": "EMEA"},
            output={"price": 1200, "currency": "EUR", "quote_id": f"Q-{i}"},
            task=TASK,
            step=0,
            latency_ms=42,
        )
        await run.tools.record(
            "crm.update_quote",
            args={"quote_id": f"Q-{i}", "amount": 1200},
            output={"ok": True},
            task=TASK,
            step=1,
        )
        # only a run labelled successful validates a procedure
        await run.runs.outcome(f"run_readme_{i}", success=True)

    plan = await agent.tools.plan(TASK, available_tools=TOOLS)
    assert plan.valid and plan.render() and plan.render_script()
    steps = [s["tool"] for s in plan.steps]
    assert steps == ["pricing.lookup_price", "crm.update_quote"], steps
    assert plan.support == 3 and plan.success_rate == 1.0
    # the headline claim: an argument is bound from an earlier step's output
    binding = next(b for b in plan.steps[1]["bindings"] if b["argument"] == "quote_id")
    assert binding["source_step"] == 0 and binding["source_field"]

    ran: list[str] = []

    async def runner(name: str, args: dict) -> dict:
        ran.append(name)
        return {"ok": True}

    result = await agent.tools.execute(
        ToolCall(tool="crm.update_quote", args={"quote_id": "Q-X"}, task=TASK), runner
    )
    assert ran == ["crm.update_quote"] and result.recorded
