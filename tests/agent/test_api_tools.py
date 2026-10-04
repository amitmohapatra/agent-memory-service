"""Tool memory end to end: an agent publishes its catalog, records what it ran, reports how
the run ended (RUN feedback), asks which tool to call next time and with which arguments,
accepts the approval rule its reviewers' decisions support, and registers its own model key.
"""

from __future__ import annotations

import base64

import pytest

from memory_service.api.app import create_app
from tests.agent.conftest import BOOTSTRAP, sdk
from tests.conftest import PG_AVAILABLE, _test_overrides
from trellis.memory import MemoryError
from trellis.memory.approval import evaluate

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
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
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
        authentication={"bootstrap_admin_key": BOOTSTRAP},
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
    run_id = str(agent.scope.agent_run_id)
    outcome = await agent.feedback("run", run_id, "confirm", source="system", comment="accepted")
    assert outcome.target_kind == "run" and outcome.target_id == run_id
    return run_id


@pytest.mark.covers("tools.record_invocation", "feedback.submit_feedback")
async def test_an_agent_records_its_calls_and_reports_how_the_run_ended(app, running) -> None:
    admin, harness = await _harness(app)
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

    run_id = str(agent.scope.agent_run_id)
    ended = await agent.feedback("run", run_id, "confirm", source="system")
    assert ended.source == "system" and ended.target_id == run_id
    assert (await agent.feedback.get(ended.feedback_id)).projection.action == "run_labelled"  # type: ignore[union-attr]

    # A person's verdict outranks the run's own status: the run corrected to a failure stops
    # validating anything mined from it.
    corrected = await agent.feedback("run", run_id, "reject", comment="rolled back")
    assert corrected.review is not None and corrected.review.state == "pending"
    await admin.bind(tenant_id="acme").feedback.approve(corrected.feedback_id)
    assert (await agent.feedback.get(corrected.feedback_id)).projection.action == "run_labelled"  # type: ignore[union-attr]


@pytest.mark.covers("tools.put_catalog", "tools.list_tools")
async def test_an_agent_publishes_its_catalog_and_reads_it_back_with_statistics(
    app, running
) -> None:
    _, harness = await _harness(app)
    agent = harness.bind(user_id="u1").agent("quote-bot")

    stored = await agent.advanced.tools.put_catalog(CATALOG)
    assert {t.name for t in stored} == {LOOKUP, UPDATE}
    lookup = next(t for t in stored if t.name == LOOKUP)
    assert lookup.required == ["sku", "region"]
    # no side_effects given: the risk tier follows the MCP annotations, as they were sent
    assert lookup.side_effects is None and lookup.risk == "read"
    assert lookup.annotations == {"readOnlyHint": True, "openWorldHint": False}
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
    assert {c.name for c in cold.tools} <= {LOOKUP, UPDATE}

    for quote in ("Q-1183", "Q-2001"):
        await _one_successful_run(user.agent("quote-bot"), quote)

    fresh = user.agent("quote-bot")
    hints = await fresh.tool_hints(TASK, available=[LOOKUP, UPDATE])
    assert hints.plan is not None and hints.plan.steps == [LOOKUP, UPDATE]
    assert hints.plan.runs == 2 and hints.plan.success_rate == 1.0
    assert hints.next is not None and hints.next.name == LOOKUP == hints.tools[0].name
    assert 0.0 <= hints.next.confidence <= 1.0
    # every labelled run looked the price up in EMEA (the procedure binds the literal), and
    # the task names the SKU itself - never the quote, which only looks like one
    assert hints.next.args == {"region": "EMEA", "sku": "SKU-22"}

    # after the lookup, the plan moves on and the quote comes from the lookup's output
    await fresh.record_tool(
        LOOKUP,
        {"sku": "SKU-22", "region": "EMEA"},
        output={"price": 1200, "quote": "Q-3003"},
        task=TASK,
        step=0,
    )
    after = await fresh.tool_hints(TASK, available=[LOOKUP, UPDATE])
    assert after.next is not None and after.next.name == UPDATE
    assert after.next.args["quote"] == "Q-3003", "the quote comes from the lookup's output"

    # a caller that cannot call the second tool is never handed a plan that needs it
    partial = await fresh.tool_hints(TASK, available=[LOOKUP])
    assert partial.plan is None and {c.name for c in partial.tools} == {LOOKUP}


@pytest.mark.covers(
    "feedback.submit_feedback", "tools.approval_suggestions", "tools.accept_approval_suggestion"
)
async def test_approvals_become_a_rule_that_applies_once_accepted(app, running) -> None:
    _, harness = await _harness(app)
    agent = harness.bind(user_id="u1").agent("quote-bot")
    await agent.advanced.tools.put_catalog(CATALOG)
    for i in range(5):
        await agent.feedback(
            "tool_call",
            f"call_{i}",
            "approve",
            metadata={"tool": UPDATE, "args": {"quote": f"Q-{i}", "price": 1200 + i}},
        )
    suggestions = await agent.advanced.tools.approval_suggestions()
    assert [(s.tool, s.suggestion, s.support) for s in suggestions] == [(UPDATE, "auto_approve", 5)]
    suggestion = suggestions[0]
    assert suggestion.arg_shape == "price:num:1e3,quote:str" and not suggestion.accepted
    assert await agent.advanced.tools.approval_suggestions(tool=LOOKUP) == []
    before = (await agent.advanced.tools.catalog(names=[UPDATE]))[0]
    assert before.approve_when is None, "a suggestion is never applied on its own"

    entry = await agent.advanced.tools.accept_suggestion(suggestion.id)
    assert entry.name == UPDATE and entry.approve_when
    # calls of the learned shape stop asking; any other shape still follows the tool's tier
    assert evaluate(entry.approve_when, {"quote": "Q-9", "price": 1500}) is False
    assert (await agent.advanced.tools.approval_suggestions())[0].accepted is True
    # accepting again is the same rule, not a second clause
    assert (await agent.advanced.tools.accept_suggestion(suggestion.id)).approve_when == (
        entry.approve_when
    )
    with pytest.raises(MemoryError) as other_agent:
        await (
            harness.bind(user_id="u1")
            .agent("other-bot")
            .advanced.tools.accept_suggestion(suggestion.id)
        )
    assert other_agent.value.status == 404


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
    "tools.tool_hints",
    "tools.put_catalog",
    "tools.list_tools",
    "tools.approval_suggestions",
    "tools.accept_approval_suggestion",
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
        claiming.advanced.tools.accept_suggestion("sug_x"),
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
