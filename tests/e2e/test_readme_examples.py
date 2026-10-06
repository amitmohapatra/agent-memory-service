"""The walkthroughs the READMEs and the usage guide teach, executed.

A README that drifts from the API is worse than no README, so every walkthrough a reader is
invited to copy runs here against the real service: the Way 2 turn in README.md and
sdk/python/README.md, and the single-agent, multi-agent and tool-memory walkthroughs that
docs/USAGE.md and examples/03, 05 and 07 teach. If you change the SDK surface and this fails,
fix the docs and the examples in the same commit (``make docs-check`` and ``make examples``).
"""

from __future__ import annotations

import pytest

from tests.e2e.conftest import sdk_client

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
    """One agent (USAGE, examples/03): chat, facts, context bundle, evidence gating."""
    memory = sdk_client(app)
    ctx = _bind(memory)

    await ctx.history.add(
        [
            ("USER", "Revenue was EUR 412 million in FY26."),
            ("ASSISTANT", "Noted — that's up 4% year on year."),
        ]
    )
    await ctx.remember("Prefers metric units", memory_type="PREFERENCE")
    await ctx.history.add([("EVENT", "User cancelled the Pro plan")])

    prompt = await ctx.context("how did revenue develop?")
    assert prompt.rendered and prompt.bundle_id
    bundle = await ctx.context("how did revenue develop?", format="full")
    # the attributes the README tells readers to inspect
    for attr in (
        "conversation",
        "thread_summary",
        "memories",
        "knowledge",
        "graph_facts",
        "evidence_status",
        "missing_evidence",
    ):
        assert hasattr(bundle, attr), attr
    assert bundle.evidence_status in {"COMPLETE", "INCOMPLETE", "INSUFFICIENT"}

    assert await ctx.search("revenue") != []
    assert await ctx.advanced.memories.list() != []

    gated = await ctx.context("what were FY26 restructuring savings?", format="full")
    assert gated.evidence_status  # the README branches on this value


async def test_readme_multi_agent_visibility(app, client) -> None:
    """Several agents (USAGE, examples/05): a RUN note reaches the owner and the run it
    spawns, and nobody else. This is the claim the walkthrough is built on."""
    memory = sdk_client(app)
    ctx = _bind(memory)
    researcher = ctx.agent("researcher", agent_group_id="analysis-crew")
    writer = ctx.agent("writer", agent_group_id="analysis-crew")
    await researcher.remember(NOTE, visibility="RUN")
    child = researcher.agent("fact-checker")

    async def sees(scope) -> bool:
        return any(NOTE in (i.text or "") for i in await scope.search(NOTE, kinds=["memory"]))

    assert await sees(researcher), "the owning run must see its own note"
    assert await sees(child), "a spawned child must receive the hand-off"
    assert not await sees(writer), "a sibling agent must not see RUN-scoped notes"
    assert not await sees(ctx), "the user must not see agent notes"

    shared = "FY26 revenue is EUR 412m, confirmed in two sources"
    await researcher.remember(shared, memory_type="SHARED", visibility="AGENT_GROUP")

    # sharing is only meaningful if the crew can actually read it: the sibling that saw
    # nothing of the RUN note must see this one
    async def sees_shared(scope) -> bool:
        return any(shared in (i.text or "") for i in await scope.search(shared, kinds=["memory"]))

    assert await sees_shared(writer), "an AGENT_GROUP memory must reach the crew"


async def test_readme_tool_memory_walkthrough(app, client) -> None:
    """Tool memory (USAGE, examples/07): publish the catalog, record what ran, label the run, and the
    service learns the procedure; tool hints read it back as a plan with the next step and
    its arguments."""
    memory = sdk_client(app)
    ctx = _bind(memory)
    agent = ctx.agent("ops", agent_run_id="run_readme_next")
    await agent.advanced.tools.put_catalog(TOOLS)

    for i in range(3):
        run = ctx.agent("ops", agent_run_id=f"run_readme_{i}")
        await run.record_tool(
            "pricing.lookup_price",
            args={"sku": f"SKU-{i}", "region": "EMEA"},
            output={"price": 1200, "currency": "EUR", "quote_id": f"Q-{i}"},
            task=TASK,
            step=0,
            latency_ms=42,
        )
        await run.record_tool(
            "crm.update_quote",
            args={"quote_id": f"Q-{i}", "amount": 1200},
            output={"ok": True},
            task=TASK,
            step=1,
        )
        # only a run labelled successful validates a procedure
        await run.feedback("run", run.scope.agent_run_id, "confirm", source="system")

    hints = await agent.tool_hints(TASK, available=[t["name"] for t in TOOLS])
    assert hints.plan is not None and hints.next is not None
    assert hints.next.name == "pricing.lookup_price"
    assert hints.plan.steps == ["pricing.lookup_price", "crm.update_quote"], hints.plan.steps
    assert hints.plan.runs == 3 and hints.plan.success_rate == 1.0
    # the headline claim: an argument is bound from an earlier step's output
    run = ctx.agent("ops", agent_run_id="run_readme_next")
    await run.record_tool(
        "pricing.lookup_price",
        args={"sku": "SKU-22", "region": "EMEA"},
        output={"price": 1300, "currency": "EUR", "quote_id": "Q-77"},
        task=TASK,
        step=0,
    )
    after = await run.tool_hints(TASK, available=[t["name"] for t in TOOLS])
    assert after.next is not None and after.next.name == "crm.update_quote"
    assert after.next.args["quote_id"] == "Q-77"


async def test_sdk_readme_one_turn_on_its_own(app, client) -> None:
    """README: Where this fits, Way 2: context into the prompt, the turn recorded, feedback
    on the run, with no framework and no harness."""
    memory = sdk_client(app)
    run = memory.bind(tenant_id="acme", user_id="u1", thread_id="thr_1").agent("support")
    question = "What changed in EBITDA?"

    pushed = await run.context(question, window=False)
    assert pushed.rendered and pushed.bundle_id
    answer = f"Answered from {pushed.token_estimate} tokens of context"
    await run.history.add([("USER", question), ("ASSISTANT", answer)])
    assert [(m.role, m.content) for m in await run.history()][-2:] == [
        ("USER", question),
        ("ASSISTANT", answer),
    ]

    verdict = await run.feedback("run", run.scope.agent_run_id, "confirm")
    assert verdict.target_kind == "run" and verdict.target_id == run.scope.agent_run_id
