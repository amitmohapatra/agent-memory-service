"""Memory inference cannot inherit a Bifrost key's MCP tool permissions."""

import json

import httpx
import pytest
import respx

from memory_service.adapters.models.llm import EndpointLLM, LLMCallFailed
from memory_service.ports.models import LLMMessage
from tests.support_llm import BASE, CATALOG, chat_response, llm_settings

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("structured", [False, True])
async def test_all_memory_calls_explicitly_deny_mcp_and_use_service_prompts(structured):
    """On a Bifrost route: the gateway's deny-all MCP scope, no content logging."""
    adapter = EndpointLLM(llm_settings(gateway="bifrost"))
    messages = [LLMMessage(role="system", content="Service-owned prompt.")]
    with respx.mock as mock:
        mock.get(f"{BASE}/models").respond(200, json=CATALOG)
        route = mock.post(f"{BASE}/chat/completions").respond(
            200, json=chat_response('{"ok":true}')
        )
        if structured:
            await adapter.structured(messages, schema={"type": "object"})
        else:
            await adapter.complete(messages)
        request = route.calls[0].request
        assert request.headers["x-bf-mcp-include-clients"] == ""
        assert request.headers["x-bf-mcp-include-tools"] == ""
        assert request.headers["x-bf-disable-content-logging"] == "true"
        assert "x-bf-prompt-id" not in request.headers
        assert "x-bf-mcp-session-id" not in request.headers
        body = json.loads(request.content)
        assert "tool_choice" not in body and "tools" not in body
        assert body["messages"][0]["content"] == "Service-owned prompt."
    await adapter.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("tool_calls", [{"id": "x", "function": {"name": "delete", "arguments": "{}"}}]),
        ("function_call", {"name": "delete", "arguments": "{}"}),
    ],
)
async def test_tool_response_is_rejected_even_when_text_is_present(field, value):
    payload = chat_response("Here is an answer.")
    payload["choices"][0]["message"][field] = value
    adapter = EndpointLLM(llm_settings())
    with respx.mock as mock:
        mock.get(f"{BASE}/models").respond(200, json=CATALOG)
        route = mock.post(f"{BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=payload)
        )
        with pytest.raises(LLMCallFailed, match="unexpectedly requested a tool"):
            await adapter.complete([LLMMessage(role="user", content="remember this")])
        assert route.call_count == 1
    await adapter.close()


async def test_another_endpoint_is_sent_no_bifrost_header_at_all() -> None:
    adapter = EndpointLLM(llm_settings())
    with respx.mock as mock:
        mock.get(f"{BASE}/models").respond(200, json=CATALOG)
        route = mock.post(f"{BASE}/chat/completions").respond(200, json=chat_response("ok"))
        await adapter.complete([LLMMessage(role="user", content="x")])
        request = route.calls[0].request
        assert not [h for h in request.headers if h.lower().startswith("x-bf-")]
        assert "tool_choice" not in json.loads(request.content)
    await adapter.close()
