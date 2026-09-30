"""The pushed context as an agent gets it: the pinned profile, the thread's summary, the
procedure learned for the task, tool hints, the revision and a delta - through the SDK."""

from __future__ import annotations

import pytest

from tests.agent.conftest import BOOTSTRAP, sdk

pytestmark = pytest.mark.e2e

TASK = "update quote Q-1183 with the EMEA price for SKU-22"
LOOKUP, UPDATE = "pricing-lookup_price", "crm-update_quote"


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
    await agent.outcome(success=True)


@pytest.mark.covers("retrieval.context")
async def test_the_push_carries_profile_summary_procedure_tools_and_a_delta(app, running) -> None:
    _, harness = await _tenant(app)
    user = harness.bind(user_id="u1", thread_id="thr_push")
    for quote in ("Q-1", "Q-2"):
        await _learned_run(user.agent("quote-bot"), quote)
    agent = user.agent("quote-bot")
    await agent.profile.set("user", "name: Ann\nregion: EMEA")
    for i in range(20):
        await (agent.chat.user if i % 2 == 0 else agent.chat.assistant)(f"Turn {i} about quotes.")

    bundle = await agent.context(TASK, token_budget=4000, tools={"available": [LOOKUP, UPDATE]})
    assert [b.block for b in bundle.profile] == ["user"]
    assert bundle.thread_summary is not None and bundle.thread_summary.covers_to_sequence == 20
    assert bundle.conversation.message_ids == [], "every message is in the summary"
    assert [s["tool"] for s in bundle.procedures[0].steps] == [LOOKUP, UPDATE]
    assert bundle.tools is not None and bundle.tools.next == LOOKUP
    assert bundle.tools.prefill["region"].value == "EMEA"
    rendered = bundle.rendered
    assert rendered.index("## Profile") < rendered.index("## Conversation summary")
    assert "## Procedures that worked for this task" in rendered and "## Tools" in rendered
    assert bundle.revision > 0 and bundle.delta is False

    again = await agent.context(
        TASK,
        token_budget=4000,
        tools={"available": [LOOKUP, UPDATE]},
        since_revision=bundle.revision,
    )
    assert again.delta is True and again.revision == bundle.revision
    assert again.memories == [] and again.knowledge == []
    assert again.profile == bundle.profile, "the pinned sections always come whole"

    await agent.profile.edit("user", "EMEA", "APAC")
    moved = await agent.context(TASK, token_budget=4000, tools={"available": [LOOKUP, UPDATE]})
    assert moved.revision > bundle.revision and "APAC" in moved.rendered
