"""Overload protection (ADR 0031): bounded model queues, per-request deadlines, the judge's
own deadline, and the uvicorn limits the entrypoint passes."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import httpx
import orjson
import pytest
from starlette.types import Receive, Scope, Send

from memory_service.adapters.models._runner import SerialRunner, configure_runners
from memory_service.api.deadline import DeadlineMiddleware, route_class
from memory_service.config.constants import OverloadTuning
from memory_service.domain.errors import DependencyUnavailable

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("GET", "/v1/memories", "read"),
        ("POST", "/v1/context", "read"),
        ("POST", "/v1/recall", "read"),
        ("POST", "/v1/tools/hints", "read"),
        ("POST", "/v1/verify", "verify"),
        ("POST", "/v1/messages", "write"),
        ("DELETE", "/v1/memories/mem_1", "write"),
        ("POST", "/v1/documents", None),
        ("GET", "/v1/documents", "read"),
        ("GET", "/health/ready", None),
        ("GET", "/metrics", None),
    ],
)
def test_route_classes(method: str, path: str, expected: str | None) -> None:
    assert route_class(method, path) == expected


def _app(delay: float, *, start_first: bool = False, own_timeout: bool = False):
    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if own_timeout:
            raise TimeoutError("the application's own")
        if start_first:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"partial", "more_body": True})
        await asyncio.sleep(delay)
        if not start_first:
            await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"done"})

    return app


async def _call(app: Any, method: str = "POST", path: str = "/v1/context") -> httpx.Response:
    limits = OverloadTuning(read_deadline_seconds=0.05, write_deadline_seconds=0.5)
    transport = httpx.ASGITransport(app=DeadlineMiddleware(app, limits=limits))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        return await client.request(method, path)


async def test_a_read_past_its_deadline_is_a_retryable_timeout_problem() -> None:
    response = await _call(_app(1.0))
    assert response.status_code == 504
    assert response.headers["content-type"].startswith("application/problem+json")
    body = orjson.loads(response.content)
    assert body["code"] == "TIMEOUT" and body["retryable"] is True
    assert body["instance"] == "/v1/context"


async def test_a_write_has_the_longer_budget_and_a_fast_request_is_untouched() -> None:
    assert (await _call(_app(0.1), path="/v1/messages")).status_code == 200
    assert (await _call(_app(0.0))).content == b"done"


async def test_uploads_and_probes_carry_no_deadline() -> None:
    assert (await _call(_app(0.1), path="/v1/documents")).status_code == 200
    assert (await _call(_app(0.1), method="GET", path="/health/ready")).status_code == 200


async def test_a_started_response_is_not_turned_into_a_504() -> None:
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Any) -> None:
        sent.append(message)

    middleware = DeadlineMiddleware(
        _app(1.0, start_first=True), limits=OverloadTuning(read_deadline_seconds=0.05)
    )
    scope = {"type": "http", "method": "POST", "path": "/v1/context", "headers": []}
    await middleware(scope, receive, send)  # type: ignore[arg-type]
    # the status already sent stays what it was; the body is just cut short
    assert [m.get("status") for m in sent if m["type"] == "http.response.start"] == [200]


async def test_the_applications_own_timeout_is_not_reported_as_the_deadline() -> None:
    with pytest.raises(TimeoutError, match="own"):
        await _call(_app(0.0, own_timeout=True))


async def test_a_full_model_queue_refuses_at_once_and_drains_back() -> None:
    configure_runners(max_waiters=1)
    runner = SerialRunner("test")
    release = threading.Event()
    try:
        inside = asyncio.create_task(runner.run(release.wait, 5))
        await asyncio.sleep(0.05)
        queued = asyncio.create_task(runner.run(lambda: "queued"))
        await asyncio.sleep(0.05)
        assert runner.waiting == 1
        with pytest.raises(DependencyUnavailable, match="saturated"):
            await runner.run(lambda: "refused")
        release.set()
        assert await inside is True
        assert await queued == "queued"
        assert runner.waiting == 0
        assert await runner.run(lambda: "after") == "after"
    finally:
        release.set()
        configure_runners(max_waiters=None)
        runner.close()


async def test_an_unbounded_queue_never_refuses() -> None:
    configure_runners(max_waiters=None)
    runner = SerialRunner("test")
    try:
        results = await asyncio.gather(*(runner.run(lambda i=i: i) for i in range(50)))
        assert results == list(range(50))
    finally:
        runner.close()


async def test_the_judge_has_one_deadline_for_the_answer_and_the_claim_stays_borderline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The judge was bounded only by the call's timeout times its retries, per claim."""
    import dataclasses
    from types import SimpleNamespace

    from memory_service.adapters.models.nli import LexicalNLI
    from memory_service.config.constants import LLM, NLISettings
    from memory_service.modules.grounding import cascade as cascade_module
    from tests.unit.test_grounding import E1

    calls: list[str] = []

    async def slow(use: str, **_: Any) -> dict[str, Any]:
        calls.append(use)
        await asyncio.sleep(5)
        return {"supported": True, "reason": "too late"}

    assist = SimpleNamespace(wants=lambda use: True, structured=slow, tokens_used=lambda: 0)
    monkeypatch.setattr(
        cascade_module, "LLM", dataclasses.replace(LLM, judge_deadline_seconds=0.05)
    )
    claim = "Restructuring savings explain the EBITDA growth reported for the year."
    cascade = cascade_module.GroundingCascade(
        LexicalNLI(),
        settings=NLISettings(),
        assist=assist,  # type: ignore[arg-type]
    )
    started = asyncio.get_running_loop().time()
    report = await asyncio.wait_for(cascade.verify(claim, [E1]), timeout=2)
    assert asyncio.get_running_loop().time() - started < 1
    assert calls == ["grounding_judge"]
    assert report.claims[0].verdict == "borderline" and report.claims[0].method == "nli"


async def test_readiness_is_postgres_and_the_process_and_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only PostgreSQL and the process are mandatory; a down search store degrades the
    answer without failing it, and probes within the cache window ping nothing."""
    from memory_service.application.container import Container, Dependency
    from memory_service.config.settings import Settings

    pings: dict[str, int] = {"postgres": 0, "qdrant": 0, "openfga": 0}

    def pinger(name: str, ok: bool, delay: float = 0.0):
        async def ping() -> bool:
            pings[name] += 1
            await asyncio.sleep(delay)
            return ok

        return ping

    container = Container(settings=Settings(_env_file=None), version="t")
    container.add_dependency(Dependency("postgres", True, pinger("postgres", True)))
    container.add_dependency(Dependency("qdrant", False, pinger("qdrant", False)))
    # a hung store is a timed-out ping, not a hung probe
    container.add_dependency(Dependency("openfga", False, pinger("openfga", True, delay=10)))
    from memory_service.application import container as container_module

    monkeypatch.setattr(
        container_module,
        "OVERLOAD",
        container_module.OVERLOAD.__class__(
            readiness_cache_seconds=60, readiness_ping_timeout_seconds=0.05
        ),
    )
    first = await asyncio.wait_for(container.readiness(), timeout=2)
    assert first["postgres"] == {"ok": True, "mandatory": True}
    assert first["qdrant"] == {"ok": False, "mandatory": False}
    assert first["openfga"]["ok"] is False and first["openfga"]["error"] == "TimeoutError"
    assert first["process"] == {"ok": True, "mandatory": True}
    await container.readiness()
    assert pings == {"postgres": 1, "qdrant": 1, "openfga": 1}, "the second probe pinged"
    await container.close()
    assert (await container.readiness())["process"]["ok"] is False, "shutdown drains first"


def test_only_postgres_is_wired_as_mandatory() -> None:
    import re
    from pathlib import Path

    wiring = (
        Path(__file__).resolve().parents[2] / "src/memory_service/adapters/wiring.py"
    ).read_text()
    assert re.findall(r'name="(\w+)", mandatory=True', wiring) == ["postgres"]
