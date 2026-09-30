"""Tool memory end to end: an agent publishes its catalog, records what it ran, labels the run,
asks which tool to call next time and with which arguments, reads the approval rules its
reviewers' decisions support, and registers its own model key.
"""

from __future__ import annotations

import base64

import pytest

from memory_service.api.app import create_app
from tests.agent.conftest import BOOTSTRAP, sdk
from tests.conftest import PG_AVAILABLE, _test_overrides
from trellis.memory import MemoryError

pytestmark = pytest.mark.e2e

TASK = "update quote Q-1183 with the EMEA price for SKU-22"
LOOKUP = "pricing-lookup_price"
UPDATE = "crm-update_quote"
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
        "source": "mcp",
        "server": "pricing",
    },
    {
        "name": UPDATE,
        "description": "Write a price onto a quote",
        "input_schema": {
            "type": "object",
            "properties": {"quote": {"type": "string"}, "price": {"type": "number"}},
            "required": ["quote", "price"],
        },
        "side_effects": "write",
        "source": "mcp",
        "server": "crm",
    },
]
#: An envelope key an operator would hold; the agent key it wraps never leaves the store.
ENVELOPE = base64.urlsafe_b64encode(b"p9b-agent-credential-key-32bytes").decode()


@pytest.fixture
def app(make_settings):
    """The shared agent app plus the one operator setting the model-key routes need: an
    envelope key. Without it a tenant cannot register a key at all, so the routes could only
    ever be tested for their refusals."""
    if not PG_AVAILABLE:
        pytest.skip("PostgreSQL not reachable")
    settings = make_settings(
        authentication={"mode": "api_key", "bootstrap_admin_key": BOOTSTRAP},
        agent_credentials={"active_key_id": "p9b", "encryption_keys": {"p9b": ENVELOPE}},
    )
    return create_app(settings, overrides=_test_overrides(tasks="inline"))


async def _harness(app, tenant_id: str = "acme"):
    platform = sdk(app, BOOTSTRAP)
    tenant = await platform.admin.create_tenant(tenant_id.title(), tenant_id=tenant_id)
    admin = sdk(app, tenant.admin_key.token)
    service = await admin.tenant.keys.issue("service", f"{tenant_id}-harness")
    return admin, sdk(app, service.token)


async def _one_successful_run(agent, quote: str = "Q-1183") -> str:
    """Record the two calls of a run and label the run successful, as an adapter would. The
    quote id flows from the lookup's output into the update's arguments."""
    first = await agent.record_tool(
        LOOKUP,
        {"sku": "SKU-22", "region": "EMEA"},
        output={"price": 1200, "quote": quote},
        task=TASK,
        step=0,
    )
    second = await agent.record_tool(
        UPDATE,
        {"quote": quote, "price": 1200},
        output={"ok": True},
        task=TASK,
        step=1,
        latency_ms=42.0,
    )
    assert first.step == 0 and second.step == 1
    outcome = await agent.outcome(success=True, note="accepted")
    assert outcome.success is True
    return str(agent.scope.agent_run_id)


@pytest.mark.covers("tools.record_invocation", "tools.set_run_outcome")
async def test_an_agent_records_its_calls_and_labels_the_run(app, running) -> None:
    _, harness = await _harness(app)
    agent = harness.bind(user_id="u1").agent("quote-bot")

    recorded = await agent.record_tool(
        LOOKUP, {"sku": "SKU-22", "region": "EMEA"}, output={"price": 1200}, task=TASK, step=0
    )
    assert recorded.invocation_id and recorded.args_hash and recorded.recorded is True

    # Idempotent on run + step + tool + arguments: a retried record is the same invocation,
    # and says so, which is what lets an adapter retry a failed HTTP call blindly.
    again = await agent.record_tool(
        LOOKUP, {"sku": "SKU-22", "region": "EMEA"}, output={"price": 1200}, task=TASK, step=0
    )
    assert again.invocation_id == recorded.invocation_id and again.recorded is False

    outcome = await agent.outcome(success=True, note="accepted")
    assert (outcome.run_id, outcome.success, outcome.source) == (
        agent.scope.agent_run_id,
        True,
        "explicit",
    )

    # The label is the run's, and the last word wins: a run corrected to a failure stops
    # validating anything mined from it.
    corrected = await agent.outcome(success=False, note="rolled back")
    assert corrected.success is False


@pytest.mark.covers("tools.put_catalog", "tools.list_tools")
async def test_an_agent_publishes_its_catalog_and_reads_it_back_with_statistics(
    app, running
) -> None:
    _, harness = await _harness(app)
    agent = harness.bind(user_id="u1").agent("quote-bot")

    stored = await agent.advanced.tools.put_catalog(CATALOG)
    assert {t.name for t in stored} == {LOOKUP, UPDATE}
    lookup = next(t for t in stored if t.name == LOOKUP)
    assert lookup.side_effects == "read" and lookup.required == ["sku", "region"]
    again = await agent.advanced.tools.put_catalog(CATALOG)
    assert [t.version for t in again] == [t.version for t in stored], "unchanged: no new version"

    await _one_successful_run(agent)
    listed = await agent.advanced.tools.catalog(names=[LOOKUP, "nobody-knows_this"])
    assert [t.name for t in listed] == [LOOKUP]
    assert listed[0].stats.calls == 1 and listed[0].stats.success_rate == 1.0
    everything = await agent.advanced.tools.catalog()
    assert {t.name for t in everything} == {LOOKUP, UPDATE}


@pytest.mark.covers("tools.tool_hints")
async def test_labelled_runs_become_a_plan_with_the_next_step_and_its_arguments(
    app, running
) -> None:
    _, harness = await _harness(app)
    user = harness.bind(user_id="u1")
    await user.agent("quote-bot").advanced.tools.put_catalog(CATALOG)

    cold = await user.agent("quote-bot").tool_hints(TASK, available=[LOOKUP, UPDATE])
    assert cold.plan is None, "nothing learned yet: no plan is invented"
    assert {c.name for c in cold.candidates} <= {LOOKUP, UPDATE}

    for quote in ("Q-1183", "Q-2001"):
        await _one_successful_run(user.agent("quote-bot"), quote)

    fresh = user.agent("quote-bot")
    hints = await fresh.tool_hints(TASK, available=[LOOKUP, UPDATE])
    assert hints.plan is not None and [s["tool"] for s in hints.plan.steps] == [LOOKUP, UPDATE]
    assert hints.plan.support == 2 and hints.plan.success_rate == 1.0
    assert hints.next == LOOKUP and hints.candidates[0].name == LOOKUP
    # every labelled run looked the price up in EMEA: the procedure binds the literal
    assert hints.prefill["region"].value == "EMEA"
    assert hints.prefill["region"].source == "procedure"

    # after the lookup, the plan moves on and the quote comes from the lookup's output
    await fresh.record_tool(
        LOOKUP,
        {"sku": "SKU-22", "region": "EMEA"},
        output={"price": 1200, "quote": "Q-3003"},
        task=TASK,
        step=0,
    )
    after = await fresh.tool_hints(TASK, available=[LOOKUP, UPDATE])
    assert after.next == UPDATE
    assert after.prefill["quote"].value == "Q-3003" and after.prefill["quote"].source == "procedure"

    # a caller that cannot call the second tool is never handed a plan that needs it
    partial = await fresh.tool_hints(TASK, available=[LOOKUP])
    assert partial.plan is None and {c.name for c in partial.candidates} == {LOOKUP}


@pytest.mark.covers("feedback.submit_feedback", "tools.approval_suggestions")
async def test_approvals_become_a_suggested_rule_that_is_never_applied(app, running) -> None:
    _, harness = await _harness(app)
    agent = harness.bind(user_id="u1").agent("quote-bot")
    for i in range(5):
        await agent.feedback(
            "tool_call",
            f"call_{i}",
            "approve",
            metadata={"tool": UPDATE, "args": {"quote": f"Q-{i}", "price": 1200 + i}},
        )
    suggestions = await agent.advanced.tools.approval_suggestions()
    assert [(s.tool, s.suggestion, s.support) for s in suggestions] == [(UPDATE, "auto_approve", 5)]
    assert suggestions[0].arg_shape == "price:num:1e3,quote:str"
    assert await agent.advanced.tools.approval_suggestions(tool=LOOKUP) == []
    stats = await agent.advanced.tools.catalog(names=[UPDATE])
    assert stats == [] or stats[0].stats.approvals == 5


@pytest.mark.covers("agents.set_key", "agents.key_status", "agents.revoke_key")
async def test_an_agent_registers_rotates_and_revokes_its_own_model_key(app, running) -> None:
    _, harness = await _harness(app)
    agent = harness.bind(user_id="u1").agent("keyed-bot")

    fresh = await agent.advanced.model_keys.status()
    assert (fresh.registered, fresh.revoked, fresh.revision) == (False, False, 0)

    registered = await agent.advanced.model_keys.set("vk-agent-first")
    assert registered.registered is True and registered.revoked is False
    assert registered.revision == 1 and registered.updated_at is not None

    rotated = await agent.advanced.model_keys.set("vk-agent-second")
    assert rotated.revision == 2 and rotated.registered is True

    status = await agent.advanced.model_keys.status()
    assert status.revision == 2 and status.registered is True
    assert "vk-agent" not in str(status.model_dump()), "a status never carries the secret"

    revoked = await agent.advanced.model_keys.revoke()
    assert revoked.revoked is True and revoked.revision == 3
    # Revocation is a tombstone, not a delete: the revision keeps climbing so a cached key
    # can be told it is stale.
    assert (await agent.advanced.model_keys.status()).revoked is True


@pytest.mark.covers_error(
    "tools.record_invocation",
    "tools.set_run_outcome",
    "tools.tool_hints",
    "tools.put_catalog",
    "tools.list_tools",
    "tools.approval_suggestions",
    "agents.set_key",
    "agents.key_status",
    "agents.revoke_key",
)
async def test_another_tenant_reaches_no_tool_memory_and_no_agent_key(app, running) -> None:
    _, acme = await _harness(app, "acme")
    _, globex = await _harness(app, "globex")
    mine = acme.bind(user_id="u1").agent("quote-bot")
    await mine.advanced.tools.put_catalog(CATALOG)
    await _one_successful_run(mine)
    await mine.advanced.model_keys.set("vk-acme-only")

    claiming = globex.bind(tenant_id="acme", user_id="u1").agent(
        "quote-bot", agent_run_id=mine.scope.agent_run_id
    )
    for call in (
        claiming.record_tool(LOOKUP, {"sku": "X"}, task=TASK, step=0),
        claiming.tool_hints(TASK),
        claiming.advanced.tools.put_catalog(CATALOG),
        claiming.advanced.tools.catalog(),
        claiming.advanced.tools.approval_suggestions(),
        claiming.outcome(success=False),
        claiming.advanced.model_keys.set("vk-stolen"),
        claiming.advanced.model_keys.status(),
        claiming.advanced.model_keys.revoke(),
    ):
        with pytest.raises(MemoryError) as refused:
            await call
        assert refused.value.status == 403, refused.value

    # And what was learned stays the owner's: the same task under its own tenant has no plan
    # and no catalog.
    theirs = globex.bind(user_id="u1").agent("quote-bot")
    assert (await theirs.tool_hints(TASK)).plan is None
    assert await theirs.advanced.tools.catalog() == []
