"""Tool memory end to end: an agent records what it ran, labels the run, asks what to call next
time, and registers its own model key.

``POST /v1/tools/plan`` and ``POST /v1/runs/{run_id}/outcome`` were exercised by nothing in the
repository before this file, which is also why they are asserted hardest here: a plan is only
worth anything if it names the caller's own tools and comes from a run somebody labelled.
"""

from __future__ import annotations

import base64

import pytest

from memory_service.api.app import create_app
from tests.agent import coverage
from tests.agent.conftest import BOOTSTRAP, sdk
from tests.conftest import PG_AVAILABLE, _test_overrides
from trellis.memory import MemoryError

pytestmark = pytest.mark.e2e

TASK = "update quote Q-1183 with the EMEA price for SKU-22"
LOOKUP = "pricing.lookup_price"
UPDATE = "crm.update_quote"
DECLARED = [
    {"name": LOOKUP, "description": "Current list price for a SKU in a region"},
    {"name": UPDATE, "description": "Write a price onto a quote"},
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


async def _one_successful_run(agent) -> str:
    """Record the two calls of a run and label the run successful, as an adapter would."""
    first = await agent.tools.record(
        LOOKUP, {"sku": "SKU-22", "region": "EMEA"}, output={"price": 1200}, task=TASK, step=0
    )
    second = await agent.tools.record(
        UPDATE,
        {"quote": "Q-1183", "price": 1200},
        output={"ok": True},
        task=TASK,
        step=1,
        latency_ms=42.0,
    )
    assert first.step == 0 and second.step == 1
    outcome = await agent.runs.outcome(agent.scope.agent_run_id, success=True, note="accepted")
    assert outcome["success"] is True
    return str(agent.scope.agent_run_id)


@pytest.mark.covers("tools.record_invocation", "tools.set_run_outcome")
async def test_an_agent_records_its_calls_and_labels_the_run(app, running) -> None:
    _, harness = await _harness(app)
    agent = harness.bind(user_id="u1").agent("quote-bot")

    recorded = await agent.tools.record(
        LOOKUP, {"sku": "SKU-22", "region": "EMEA"}, output={"price": 1200}, task=TASK, step=0
    )
    assert recorded.invocation_id and recorded.args_hash and recorded.recorded is True

    # Idempotent on run + step + tool + arguments: a retried record is the same invocation,
    # and says so, which is what lets an adapter retry a failed HTTP call blindly.
    again = await agent.tools.record(
        LOOKUP, {"sku": "SKU-22", "region": "EMEA"}, output={"price": 1200}, task=TASK, step=0
    )
    assert again.invocation_id == recorded.invocation_id and again.recorded is False

    outcome = await agent.runs.outcome(agent.scope.agent_run_id, success=True, note="accepted")
    assert outcome == {
        "run_id": agent.scope.agent_run_id,
        "success": True,
        "source": "explicit",
    }

    # The label is the run's, and the last word wins: a run corrected to a failure stops
    # validating anything mined from it.
    corrected = await agent.runs.outcome(
        agent.scope.agent_run_id, success=False, note="rolled back"
    )
    assert corrected["success"] is False


@pytest.mark.covers("tools.plan_tools", "tools.list_procedures", "tools.record_invocation")
async def test_a_labelled_run_becomes_a_plan_that_only_names_declared_tools(app, running) -> None:
    _, harness = await _harness(app)
    user = harness.bind(user_id="u1")

    # Nothing recorded yet: the plan says so rather than inventing a chain.
    cold = await user.agent("quote-bot").tools.plan(TASK, available_tools=DECLARED)
    assert cold.valid is False and cold.reason == "no validated procedure yet"
    assert cold.steps == [] and cold.task_pattern

    agent = user.agent("quote-bot")
    run_id = await _one_successful_run(agent)

    procedures = await agent.tools.procedures(TASK)
    assert procedures, "a successful run with steps is a procedure"
    assert [step["tool"] for step in procedures[0]["steps"]] == [LOOKUP, UPDATE]

    plan = await agent.tools.plan(TASK, available_tools=DECLARED)
    assert plan.valid is True and plan.problems == []
    assert [step["tool"] for step in plan.steps] == [LOOKUP, UPDATE]
    assert plan.support >= 1 and plan.success_rate == 1.0
    assert run_id in plan.run_ids
    assert plan.script and plan.rendered

    # A caller that does not hold the second tool is told so, and is never handed a step it
    # cannot execute.
    partial = await agent.tools.plan(TASK, available_tools=DECLARED[:1])
    assert partial.valid is False and partial.steps == []
    assert UPDATE in (partial.reason or "")


@pytest.mark.covers("tools.record_tool")
async def test_the_deprecated_record_alias_still_answers_and_says_it_is_deprecated(
    app, running
) -> None:
    _, harness = await _harness(app)
    agent = harness.bind(user_id="u1").agent("legacy-bot")

    body = await harness.transport.request(
        "POST",
        "/v1/tools/record",
        scope=agent.scope,
        json={
            "scope": agent.scope.model_dump(mode="json", exclude_none=True, exclude={"trace_id"}),
            "tool": LOOKUP,
            "args": {"sku": "SKU-99", "region": "APAC"},
            "output": {"price": 900},
            "task": TASK,
            "step": 0,
        },
    )

    assert body["invocation_id"] and body["recorded"] is True and body["step"] == 0
    headers = coverage.headers_of("tools.record_tool")
    assert headers["deprecation"] == "@1790553600"
    assert headers["link"] == '</v1/tools/invocations>; rel="successor-version"'


@pytest.mark.covers("agents.set_key", "agents.key_status", "agents.revoke_key")
async def test_an_agent_registers_rotates_and_revokes_its_own_model_key(app, running) -> None:
    _, harness = await _harness(app)
    agent = harness.bind(user_id="u1").agent("keyed-bot")

    fresh = await agent.model_key_status()
    assert (fresh.registered, fresh.revoked, fresh.revision) == (False, False, 0)

    registered = await agent.set_model_key("vk-agent-first")
    assert registered.registered is True and registered.revoked is False
    assert registered.revision == 1 and registered.updated_at is not None

    rotated = await agent.set_model_key("vk-agent-second")
    assert rotated.revision == 2 and rotated.registered is True

    status = await agent.model_key_status()
    assert status.revision == 2 and status.registered is True
    assert "vk-agent" not in str(status.model_dump()), "a status never carries the secret"

    revoked = await agent.revoke_model_key()
    assert revoked.revoked is True and revoked.revision == 3
    # Revocation is a tombstone, not a delete: the revision keeps climbing so a cached key
    # can be told it is stale.
    assert (await agent.model_key_status()).revoked is True


@pytest.mark.covers_error(
    "tools.record_invocation",
    "tools.record_tool",
    "tools.plan_tools",
    "tools.list_procedures",
    "tools.set_run_outcome",
    "agents.set_key",
    "agents.key_status",
    "agents.revoke_key",
)
async def test_another_tenant_reaches_no_tool_memory_and_no_agent_key(app, running) -> None:
    _, acme = await _harness(app, "acme")
    _, globex = await _harness(app, "globex")
    mine = acme.bind(user_id="u1").agent("quote-bot")
    await _one_successful_run(mine)
    await mine.set_model_key("vk-acme-only")

    claiming = globex.bind(tenant_id="acme", user_id="u1").agent(
        "quote-bot", agent_run_id=mine.scope.agent_run_id
    )
    for call in (
        claiming.tools.record(LOOKUP, {"sku": "X"}, task=TASK, step=0),
        claiming.tools.plan(TASK, available_tools=DECLARED),
        claiming.tools.procedures(TASK),
        claiming.runs.outcome(str(mine.scope.agent_run_id), success=False),
        claiming.set_model_key("vk-stolen"),
        claiming.model_key_status(),
        claiming.revoke_model_key(),
    ):
        with pytest.raises(MemoryError) as refused:
            await call
        assert refused.value.status == 403, refused.value

    with pytest.raises(MemoryError) as alias:
        await globex.transport.request(
            "POST",
            "/v1/tools/record",
            scope=claiming.scope,
            json={
                "scope": claiming.scope.model_dump(
                    mode="json", exclude_none=True, exclude={"trace_id"}
                ),
                "tool": LOOKUP,
                "args": {"sku": "X"},
                "task": TASK,
                "step": 0,
            },
        )
    assert alias.value.status == 403

    # And the procedure stays the owner's: the same task, mined under its own tenant, is empty.
    assert await globex.bind(user_id="u1").agent("quote-bot").tools.procedures(TASK) == []
