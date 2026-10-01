"""The memory tools an agent calls itself, through the SDK: listed once, called in scope."""

from __future__ import annotations

import pytest

from tests.agent.conftest import BOOTSTRAP, sdk
from trellis.memory import NotFoundError, ValidationError

pytestmark = pytest.mark.e2e


async def _tenant(app, tenant_id: str = "acme"):
    platform = sdk(app, BOOTSTRAP)
    tenant = await platform.admin.create_tenant(tenant_id.title(), tenant_id=tenant_id)
    admin = sdk(app, tenant.admin_key.token)
    service = await admin.tenant.keys.issue("service", f"{tenant_id}-harness")
    return admin, sdk(app, service.token)


@pytest.mark.covers("agent_tools.list_agent_tools", "agent_tools.call_agent_tool")
async def test_an_agent_lists_its_memory_tools_and_uses_them(app, running) -> None:
    _, harness = await _tenant(app)
    agent = harness.bind(user_id="u1", thread_id="thr_pull").agent("buyer")
    tools = await agent.agent_tools()
    assert len(tools) == 6 and all(t.input_schema["type"] == "object" for t in tools)

    stored = await agent.call_agent_tool(
        "memory_remember", {"content": "Deliveries go to dock 4", "kind": "SEMANTIC"}
    )
    found = await agent.call_agent_tool("memory_search", {"query": "where do deliveries go"})
    assert found[0]["id"] == stored["id"] and "dock 4" in found[0]["text"]
    # a memory the agent stated for the user is the user's: another agent of theirs finds it
    other = harness.bind(user_id="u1").agent("planner")
    assert (await other.call_agent_tool("memory_search", {"query": "deliveries dock"}))[0][
        "id"
    ] == stored["id"]


@pytest.mark.covers_error("agent_tools.call_agent_tool")
async def test_an_unknown_tool_or_bad_arguments_are_refused(app, running) -> None:
    _, harness = await _tenant(app)
    agent = harness.bind(user_id="u1").agent("buyer")
    with pytest.raises(NotFoundError):
        await agent.call_agent_tool("shell_exec", {})
    with pytest.raises(ValidationError):
        await agent.call_agent_tool("memory_search", {"query": "x", "k": 500})
