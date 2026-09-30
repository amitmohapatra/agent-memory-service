"""The pushed context as an agent gets it, section by section, from a realistic seeded scope:
the pinned profile, the thread's summary and its recent window, the memories, documents and
graph facts that answer the task, the procedure learned for it, and the tool hints (the next
tool, argument values, what is missing) - each present when it should be, and about the task.
"""

from __future__ import annotations

import pytest

from tests.agent.conftest import BOOTSTRAP, sdk

pytestmark = pytest.mark.e2e

TASK = "update quote Q-1183 with the EMEA price for SKU-22"
LOOKUP, UPDATE = "pricing-lookup_price", "crm-update_quote"
CATALOG = [
    {
        "name": LOOKUP,
        "description": "Current list price for a SKU in a region",
        "input_schema": {
            "type": "object",
            "properties": {"sku": {"type": "string"}, "region": {"type": "string"}},
            "required": ["sku", "region"],
        },
        "side_effects": "read",
    },
    {
        "name": UPDATE,
        "description": "Write a price onto a quote",
        "input_schema": {
            "type": "object",
            "properties": {
                "quote": {"type": "string"},
                "price": {"type": "number"},
                "approver": {"type": "string"},
            },
            "required": ["quote", "price", "approver"],
        },
        "side_effects": "write",
    },
]
MEMORY = "The EMEA list price for SKU-22 was raised to 1200 EUR in September."
GRAPH = "Priya Raman owns quote Q-1183 for Globex GmbH."
POLICY = (
    b"EMEA pricing policy.\n\nQuotes for SKU-22 in the EMEA region use the EMEA list price. "
    b"A quote update above 1000 EUR needs a named approver from the pricing desk.\n"
)


async def _tenant(app, tenant_id: str = "acme"):
    platform = sdk(app, BOOTSTRAP)
    tenant = await platform.admin.create_tenant(tenant_id.title(), tenant_id=tenant_id)
    admin = sdk(app, tenant.admin_key.token)
    service = await admin.tenant.keys.issue("service", f"{tenant_id}-harness")
    return admin, sdk(app, service.token)


async def _learned_run(agent, quote: str) -> None:
    await agent.record_tool(
        LOOKUP, {"sku": "SKU-22", "region": "EMEA"}, output={"quote": quote}, task=TASK, step=0
    )
    await agent.record_tool(UPDATE, {"quote": quote, "price": 1200}, task=TASK, step=1)
    await agent.feedback("run", str(agent.scope.agent_run_id), "confirm", source="system")


async def _seeded(app):
    _, harness = await _tenant(app)
    user = harness.bind(user_id="u1", thread_id="thr_push")
    await user.agent("quote-bot").advanced.tools.put_catalog(CATALOG)
    for quote in ("Q-1", "Q-2"):
        await _learned_run(user.agent("quote-bot"), quote)
    agent = user.agent("quote-bot")
    await agent.profile.edit("user", "name: Ann\nregion: EMEA")
    await agent.remember(MEMORY, visibility="USER")
    await agent.remember(GRAPH, visibility="USER", entities=["Priya Raman", "Q-1183"])
    doc = await agent.advanced.documents.add(
        ("pricing-policy.txt", POLICY, "text/plain"), visibility="USER"
    )
    assert (await agent.advanced.documents.wait_ready(doc.document_id)).status == "READY"
    await agent.history.add(
        [("USER" if i % 2 == 0 else "ASSISTANT", f"Turn {i} about quotes.") for i in range(20)]
    )
    await agent.history.add([("USER", "Please reprice Q-1183 for the EMEA customer.")])
    return agent


@pytest.mark.covers("retrieval.context")
async def test_every_section_arrives_when_it_should_and_is_about_the_task(app, running) -> None:
    agent = await _seeded(app)

    bundle = await agent.context(TASK, token_budget=6000, tools=[LOOKUP, UPDATE], format="full")
    # the pinned sections
    assert [b.block for b in bundle.profile] == ["user"]
    assert bundle.thread_summary is not None and bundle.thread_summary.covers_to_sequence >= 20
    # the window carries what the summary does not cover yet
    assert bundle.conversation.message_ids, "the latest message is in the window"
    assert "reprice Q-1183" in bundle.conversation.rendered
    # what answers the task
    assert any("1200 EUR" in m.text for m in bundle.memories), bundle.memories
    assert any("approver" in k.text for k in bundle.knowledge), bundle.knowledge
    assert any("Q-1183" in f.text for f in bundle.graph_facts), bundle.graph_facts
    # what the agent learned to do
    assert [s["tool"] for s in bundle.procedures[0].steps] == [LOOKUP, UPDATE]
    assert bundle.tools is not None and bundle.tools.next == LOOKUP
    assert bundle.tools.prefill[f"{LOOKUP}.region"].value == "EMEA"
    assert bundle.tools.prefill[f"{LOOKUP}.sku"].value == "SKU-22"
    # every item is citable by its handle, and every handle names an item of the bundle
    ids = {i.item_id for i in (*bundle.memories, *bundle.knowledge, *bundle.graph_facts)}
    assert bundle.handles and set(bundle.handles.values()) <= ids | {
        s.item_id for s in bundle.summaries
    }

    rendered = bundle.rendered
    order = [
        "## Profile",
        "## Conversation summary",
        "## Procedures that worked for this task",
        "## Tools",
        "## Recent conversation",
    ]
    assert [rendered.index(h) for h in order] == sorted(rendered.index(h) for h in order)
    assert "## Knowledge" in rendered and "## Facts" in rendered

    # The prompt form is the same context, rendered, with the tools that fit.
    prompt = await agent.context(TASK, token_budget=6000, tools=[LOOKUP, UPDATE])
    assert prompt.rendered == rendered and prompt.bundle_id
    assert prompt.tool_candidates and prompt.tool_candidates[0] == LOOKUP

    # A framework that keeps its own history asks without the window: the summary stays.
    own = await agent.context(TASK, token_budget=6000, window=False, format="full")
    assert own.conversation.message_ids == [] and "## Recent conversation" not in own.rendered
    assert own.thread_summary is not None and own.tools is None and own.procedures == []

    # An edit to the profile is in the next context: nothing stale is served from cache.
    await agent.profile.edit("user", "APAC", old="EMEA")
    moved = await agent.context(TASK, token_budget=6000, tools=[LOOKUP, UPDATE])
    assert "APAC" in moved.rendered and "region: EMEA" not in moved.rendered


@pytest.mark.covers("retrieval.context")
async def test_the_arguments_nothing_can_fill_are_asked_for(app, running) -> None:
    agent = await _seeded(app)
    await agent.record_tool(
        LOOKUP,
        {"sku": "SKU-22", "region": "EMEA"},
        output={"price": 1200, "quote": "Q-1183"},
        task=TASK,
        step=0,
    )
    bundle = await agent.context(TASK, token_budget=6000, tools=[LOOKUP, UPDATE], format="full")
    assert bundle.tools is not None and bundle.tools.next == UPDATE
    assert bundle.tools.prefill[f"{UPDATE}.quote"].value == "Q-1183"
    assert [(m.tool, m.arg) for m in bundle.tools.missing] == [(UPDATE, "approver")]
    assert f"missing {UPDATE}.approver" in bundle.rendered
