"""The user's never-rule guards the tool hints on the paths that build no context.

"Never create a stock rule at store S40", said in one conversation, is a stored rule of the
user. ``POST /v1/tools/hints`` and the ``tool_search`` agent tool read the stored profile,
which never lists the ``standing_rules`` section the context computes per request: they
handed over ``create_stock_rule`` at S40 as the next step, with its arguments filled. They
now read the rules that apply to the task themselves.
"""

from __future__ import annotations

import httpx
import pytest

from memory_service.api.app import create_app
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.tools import ToolDescriptor
from tests.integration.test_statements_follow_the_user import _chat, _say

pytestmark = pytest.mark.integration

RULE = "Never create a stock rule at store S40."
AT_S40 = "Create a stock rule for ST-101 at S40, min 1 max 3."
AT_S12 = "Create a stock rule for ST-101 at S12, min 1 max 3."
HEADERS = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "u1"}


def _owner() -> MemoryExecutionContext:
    """The user's own agent, in a new conversation of theirs."""
    return MemoryExecutionContext(
        tenant_id="acme", user_id="u1", workspace_id="ws1", agent_id="buyer", agent_run_id="r1"
    )


def _stock_rule() -> ToolDescriptor:
    return ToolDescriptor(
        tenant_id="",
        name="create_stock_rule",
        description="Create a min/max stock rule for a style at a store.",
        side_effects="write",
        input_schema={
            "type": "object",
            "properties": {
                "store": {"type": "string", "description": "The store's id, such as S12."},
                "style": {"type": "string", "description": "The style's id, such as ST-101."},
                "min_units": {"type": "integer", "description": "The fewest units."},
                "max_units": {"type": "integer", "description": "The most units."},
            },
        },
        required=["store", "style", "min_units", "max_units"],
    )


async def _publish(container, uow_factory) -> None:
    async with uow_factory() as uow:
        await container.services["tool_memory"].put_catalog(
            uow, _owner(), [(_stock_rule(), frozenset(ToolDescriptor.model_fields))]
        )
        await uow.commit()
    await container.tasks.drain()  # tools.index


async def _the_user_said_the_rule(container, uow_factory) -> None:
    await _publish(container, uow_factory)
    await _say(container, uow_factory, _chat(), RULE)


def _choice(body: dict) -> dict:
    return next(t for t in body["tools"] if t["name"] == "create_stock_rule")


async def test_the_tool_search_agent_tool_does_not_hand_over_a_forbidden_call(
    container, uow_factory
) -> None:
    await _the_user_said_the_rule(container, uow_factory)
    tools = container.services["agent_tools"]

    forbidden = _choice(await tools.call(_owner(), "tool_search", {"task": AT_S40}))
    assert forbidden["forbidden"].rstrip(".") == RULE.rstrip(".")
    assert "next" not in forbidden and "args" not in forbidden and "missing" not in forbidden

    # the rule is about S40: the call at S12 is handed over as before
    allowed = _choice(await tools.call(_owner(), "tool_search", {"task": AT_S12}))
    assert "forbidden" not in allowed and allowed["next"] is True
    assert allowed["args"]["store"] == "S12"


async def test_the_hints_endpoint_does_not_hand_over_a_forbidden_call(
    container, uow_factory
) -> None:
    await _the_user_said_the_rule(container, uow_factory)
    app = create_app(container.settings, container=container)
    app.state.container = container  # the fixture's own: no second lifespan to close it
    scope = {"workspace_id": "ws1"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://memory.test"
    ) as client:
        at_s40 = await client.post(
            "/v1/tools/hints", headers=HEADERS, json={"scope": scope, "task": AT_S40}
        )
        at_s12 = await client.post(
            "/v1/tools/hints", headers=HEADERS, json={"scope": scope, "task": AT_S12}
        )
    assert at_s40.status_code == 200, at_s40.text
    forbidden = _choice(at_s40.json())
    assert forbidden["forbidden"].rstrip(".") == RULE.rstrip(".")
    assert "next" not in forbidden and "args" not in forbidden
    allowed = _choice(at_s12.json())
    assert "forbidden" not in allowed and allowed["next"] is True


async def test_a_later_rule_lifts_the_guard(container, uow_factory) -> None:
    await _the_user_said_the_rule(container, uow_factory)
    await _say(
        container,
        uow_factory,
        _chat(),
        "From now on, create stock rules at store S40 only with my approval.",
    )
    hints = await container.services["agent_tools"].call(_owner(), "tool_search", {"task": AT_S40})
    assert "forbidden" not in _choice(hints)


async def test_a_deployment_with_no_memory_yet_hints_as_before(container, uow_factory) -> None:
    """No memory indexed yet: the rules' read finds none (no collection is no error)."""
    await _publish(container, uow_factory)
    hints = await container.services["agent_tools"].call(_owner(), "tool_search", {"task": AT_S40})
    choice = _choice(hints)
    assert "forbidden" not in choice and choice["next"] is True
