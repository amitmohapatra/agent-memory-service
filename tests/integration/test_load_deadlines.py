"""N concurrent turns (a message, the context, a verify) against one process with a small
connection pool and a slow NLI model: every request is answered by its deadline - a 504 at
the deadline, not when the model is done - and the pool is never exhausted."""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from collections.abc import Sequence

import httpx
import pytest

from memory_service.__about__ import __version__
from memory_service.adapters.models._runner import SerialRunner
from memory_service.adapters.models.nli import BatchedNLI
from memory_service.api.app import create_app
from memory_service.application.container import build_container
from memory_service.config import constants
from memory_service.config.constants import FROZEN_MODELS, OverloadTuning
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType
from memory_service.ports.models import NLIScore, ProviderInfo
from tests.integration.conftest import TABLES, integration_overrides, integration_settings

pytestmark = pytest.mark.integration

TURNS = 24
VERIFY_DEADLINE = 3.0
#: one model batch; an answer of four claims is about 1.2 s of model
BATCH_SECONDS = 0.4
H = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-Agent": "load"}
ANSWER = (
    "Supplier S01 delivers pallets to warehouse 1 every Tuesday. "
    "Supplier S05 delivers pallets to warehouse 0 every Friday. "
    "Supplier S09 delivers pallets to warehouse 4 every Monday. "
    "Supplier S13 delivers pallets to warehouse 3 every Tuesday."
)


class SlowNLI(BatchedNLI):
    """The trained head's shape, slow: ``BATCH_SECONDS`` a model batch, in its thread."""

    info = ProviderInfo(name="slow-nli", license="Apache-2.0", origin="test", locality="local")
    spec = FROZEN_MODELS.nli

    def __init__(self) -> None:
        self._runner = SerialRunner("nli-load")

    def _score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScore]:
        batches = -(-len(pairs) // self.spec.batch_size)
        time.sleep(BATCH_SECONDS * batches)
        return [NLIScore(entailment=0.9, neutral=0.05, contradiction=0.05) for _ in pairs]

    def fingerprint(self) -> str:
        return "slow-nli"


@pytest.fixture
async def app(make_settings, tmp_path, monkeypatch):
    from sqlalchemy import text

    from tests.conftest import PG_AVAILABLE
    from tests.support_real import reset_real_backends

    if not PG_AVAILABLE:
        pytest.skip("PostgreSQL not reachable")
    # one API process with the scenario stack's request pool: five plus five overflow
    settings = integration_settings(
        make_settings,
        blob={"filesystem_root": str(tmp_path / "blob")},
        database={"connection_budget": 16},
        service={"workers": 1, "environment": "test", "log_json": False, "log_level": "WARNING"},
    )
    monkeypatch.setattr(
        constants, "OVERLOAD", OverloadTuning(verify_deadline_seconds=VERIFY_DEADLINE)
    )
    container = await build_container(
        settings, __version__, overrides=integration_overrides(blob=None)
    )
    assert container.database.plan.main_size + container.database.plan.main_overflow == 10
    async with container.database.engine.begin() as conn:
        await conn.execute(text("TRUNCATE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE"))
    await reset_real_backends(container)
    nli = SlowNLI()
    container.services["grounding"].nli = nli
    for n in range(4):
        ctx = MemoryExecutionContext(tenant_id="acme", user_id=f"u{n}")
        async with container.services["uow_factory"]() as uow:
            for i in range(20):
                await container.services["memory"].remember(
                    uow,
                    ctx,
                    memory_type=MemoryType.SEMANTIC,
                    lifetime=Lifetime.LONG_TERM,
                    content=f"Supplier S{i:02d} delivers pallets to warehouse {i % 5} "
                    f"every {('Monday', 'Tuesday', 'Friday')[i % 3]}.",
                )
            await uow.commit()
    await container.tasks.drain()
    application = create_app(settings, container=container)
    application.state.container = container
    try:
        yield application
    finally:
        nli.close()
        await container.close()


async def test_concurrent_turns_meet_their_deadlines_on_a_small_pool(app) -> None:
    container = app.state.container
    pool = container.database.engine.pool
    peak = 0
    results: list[tuple[str, int, float]] = []

    async def call(client: httpx.AsyncClient, name: str, path: str, user: str, body: dict):
        nonlocal peak
        started = time.monotonic()
        response = await client.post(path, headers={**H, "X-Trellis-User": user}, json=body)
        results.append((name, response.status_code, time.monotonic() - started))
        peak = max(peak, pool.checkedout())
        return response

    async def turn(client: httpx.AsyncClient, n: int) -> None:
        user = f"u{n % 4}"
        scope = {"agent_id": "load", "agent_run_id": f"run-{n}", "thread_id": f"thr-{user}-{n % 2}"}
        await call(
            client,
            "messages",
            "/v1/messages",
            user,
            {"scope": scope, "messages": [{"role": "USER", "content": f"turn {n}: pallets?"}]},
        )
        context = await call(
            client,
            "context",
            "/v1/context",
            user,
            {"scope": scope, "query": "Who delivers pallets to warehouse 1?", "token_budget": 4000},
        )
        if context.status_code == 200:
            bundle = context.json()["bundle_id"]
            await call(
                client,
                "verify",
                "/v1/verify",
                user,
                {"scope": scope, "bundle_id": bundle, "answer": ANSWER},
            )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=60) as client:
        await asyncio.gather(*(turn(client, n) for n in range(TURNS)))

    codes = Counter((name, code) for name, code, _ in results)
    assert not [r for r in results if r[1] == 503], f"the pool ran out: {codes}"
    assert {code for name, code, _ in results if name != "verify"} <= {200, 202}, codes
    verifies = [(code, took) for name, code, took in results if name == "verify"]
    assert len(verifies) == TURNS, codes
    late = [took for code, took in verifies if took > VERIFY_DEADLINE + 0.5]
    assert not late, f"answered past the {VERIFY_DEADLINE} s deadline: {sorted(late)} {codes}"
    assert any(code == 200 for code, _ in verifies), codes
    assert peak <= 10
