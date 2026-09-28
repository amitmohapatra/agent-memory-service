"""The SDK's ids survive the real service: what trellis.memory sends is what comes back."""

from __future__ import annotations

import re
from collections.abc import Iterator

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memory_service.api.app import create_app
from trellis.memory import MemoryClient
from trellis.memory.models import Scope

TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"


@pytest.fixture
def served(settings, overrides) -> Iterator[FastAPI]:
    """The app with its lifespan running, so an ASGI transport can call it."""
    app = create_app(settings, overrides=overrides)
    with TestClient(app):
        yield app


async def test_the_sdk_ids_survive_the_real_service(served: FastAPI) -> None:
    """X-Request-ID is echoed rather than replaced (it satisfies the service's id pattern), a
    W3C scope trace id is the trace the request ran under, and an opaque one is echoed as
    the correlation id."""
    seen: list[httpx.Response] = []

    async def capture(response: httpx.Response) -> None:
        seen.append(response)

    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=served),
        base_url="http://memory.test",
        event_hooks={"response": [capture]},
    )
    client = MemoryClient("http://memory.test", api_key="test-key", http_client=http)
    try:
        await client.transport.request("GET", "/version", scope=Scope(trace_id=TRACE))
        traced = seen[-1]
        assert traced.headers["X-Request-ID"] == traced.request.headers["X-Request-ID"]
        assert re.fullmatch(r"[0-9a-f]{32}", traced.request.headers["X-Request-ID"])
        assert traced.headers["X-Trace-ID"] == TRACE
        assert traced.headers["traceparent"].startswith(f"00-{TRACE}-")

        await client.transport.request("GET", "/version", scope=Scope(trace_id="turn-42"))
        opaque = seen[-1]
        assert "X-Trace-ID" not in opaque.request.headers
        assert opaque.headers["X-Correlation-ID"] == "turn-42"
        assert opaque.headers["X-Trace-ID"] != TRACE
    finally:
        await client.aclose()
