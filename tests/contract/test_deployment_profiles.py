"""The deployment profiles that actually exist, each built and exercised end to end.

The declared provider space is ~1.3 million combinations, which is neither testable nor
meaningful: most of it is nonsense (a GCS blob store with an in-memory authorization store),
and much of the rest is test stand-ins rather than deployment choices. What is worth pinning
is the handful of shapes a real deployment takes, plus the degraded ones we promise to
survive — cache gone, graph enrichment off, no LLM.

Each profile is wired with ``build_container`` and then made to do real work: submit an
observation, run the pipeline, and read the memory back. A profile that wires but cannot
answer is not a working profile.
"""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import text

from memory_service import __version__
from memory_service.application.container import build_container
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import ObservationKind
from memory_service.modules.jobs.registry import register_handlers
from tests.conftest import DB_URL
from tests.integration.conftest import PG_AVAILABLE, TABLES

pytestmark = [pytest.mark.contract, pytest.mark.usefixtures()]

#: name -> settings overrides. Everything not named here stays at the suite's hermetic default.
PROFILES: dict[str, dict] = {
    # what CI and a laptop run: nothing outside the process
    "all-in-process": {
        "cache": {"provider": "memory"},
        "tasks": {"provider": "memory"},
        "search": {"provider": "memory"},
        "blob": {"provider": "memory"},
    },
    # the dev stack shape: real cache and real vector store
    "server-backed": {
        "cache": {"provider": "dragonfly"},
        "tasks": {"provider": "memory"},
        "search": {"provider": "qdrant"},
        "blob": {"provider": "memory"},
    },
    # a cache outage must degrade, not fail: the canonical store still answers
    "no-cache": {
        "cache": {"provider": "disabled"},
        "tasks": {"provider": "memory"},
        "search": {"provider": "memory"},
        "blob": {"provider": "memory"},
    },
    # graph enrichment is optional; memories must still land without it
    "no-graph": {
        "cache": {"provider": "memory"},
        "tasks": {"provider": "memory"},
        "search": {"provider": "memory"},
        "blob": {"provider": "memory"},
        "graph_enrichment": {"provider": "disabled"},
    },
    # the model tier deployed separately: nothing loads weights in this process
    "remote-models": {
        "cache": {"provider": "memory"},
        "tasks": {"provider": "memory"},
        "search": {"provider": "memory"},
        "blob": {"provider": "memory"},
        "models": {
            "embedding": {"dimension": 3},
            "reranker": {},
            "nli": {},
        },
    },
    # no generative model at all — the rule-based path has to carry the service
    "no-llm": {
        "cache": {"provider": "memory"},
        "tasks": {"provider": "memory"},
        "search": {"provider": "memory"},
        "blob": {"provider": "memory"},
        "models": {"llm": {"enabled": False}},
    },
}


@pytest.fixture(params=sorted(PROFILES), ids=lambda n: n)
def profile(request: pytest.FixtureRequest) -> tuple[str, dict]:
    return request.param, PROFILES[request.param]


async def test_the_profile_wires_and_can_answer(profile, make_settings, tmp_path) -> None:
    if not PG_AVAILABLE:
        pytest.skip("PostgreSQL not reachable")
    name, overrides = profile
    server = None
    if name == "remote-models":
        # a real inference server on a real socket: the point of this profile is that the
        # process loads no weights at all, so a stand-in adapter would prove nothing
        from tests.contract.test_remote_models import FakeTEI

        server = FakeTEI()
        server.__enter__()
        overrides = json.loads(json.dumps(overrides))
        for section in overrides["models"].values():
            section["url"] = server.url
    sections: dict = {
        "database": {"url": DB_URL},
        "blob": {"provider": "filesystem", "filesystem_root": str(tmp_path / "blob")},
    }
    for key, value in overrides.items():  # the profile wins over the defaults above
        sections[key] = {**sections.get(key, {}), **value} if isinstance(value, dict) else value
    if sections["blob"].get("provider") == "memory":
        sections["blob"] = {"provider": "memory"}
    settings = make_settings(**sections)

    container = await build_container(settings, __version__)
    try:
        async with container.database.engine.begin() as conn:
            await conn.execute(text("TRUNCATE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE"))

        # every mandatory dependency must actually answer, not merely be constructed
        for dependency in container.dependencies.values():
            if dependency.mandatory and dependency.ping is not None:
                assert await dependency.ping(), (
                    f"{name}: mandatory dependency {dependency.name} is down"
                )

        register_handlers(container)
        ctx = MemoryExecutionContext(
            tenant_id="acme", user_id=f"u-{uuid.uuid4().hex[:6]}", workspace_id="ws1"
        )
        fact = "My timezone is Europe/Berlin."
        uow_factory = container.services["uow_factory"]
        async with uow_factory() as uow:
            await container.services["memory"].submit_observation(
                uow, ctx, kind=ObservationKind.MESSAGE, content=fact
            )
            await uow.commit()
        await container.tasks.drain()
        await container.tasks.drain()

        async with uow_factory() as uow:
            memories = await container.services["memory"].list_memories(uow, ctx)
        assert memories, f"{name}: an observation produced no memory"
        assert any("berlin" in (m.object or "").lower() for m in memories), (
            f"{name}: the fact was stored but not extracted"
        )
    finally:
        await container.close()
        if server is not None:
            server.__exit__()


async def test_a_profile_that_cannot_work_is_refused_at_startup(make_settings, tmp_path) -> None:
    """Misconfiguration must fail while someone is watching.

    A provider that needs a generative model, in a deployment that has none, is not a runtime
    surprise to discover on the first request — it is a startup error. This is the property
    that makes the combination space safe to leave large: the impossible corners refuse
    themselves.
    """
    if not PG_AVAILABLE:
        pytest.skip("PostgreSQL not reachable")
    settings = make_settings(
        database={"url": DB_URL},
        cache={"provider": "memory"},
        tasks={"provider": "memory"},
        search={"provider": "memory"},
        blob={"provider": "filesystem", "filesystem_root": str(tmp_path / "blob")},
        graph_enrichment={"provider": "graphiti"},
        models={"llm": {"enabled": False}},
    )
    with pytest.raises(NotImplementedError, match="requires an LLM"):
        await build_container(settings, __version__)
