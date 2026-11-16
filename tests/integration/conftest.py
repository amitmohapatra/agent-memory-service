"""Integration fixtures: real PostgreSQL (migrated) when reachable, else skip."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest
from sqlalchemy import event, text

from memory_service.__about__ import __version__
from memory_service.application.container import Container, Overrides, build_container
from memory_service.config.settings import Settings
from tests.conftest import (  # noqa: E402  (one definition, see tests/conftest.py)
    DB_URL,
    _test_overrides,
)

TABLES = [
    "feedback",
    "agent_credentials",
    "llm_policies",
    "llm_usage_daily",
    "tool_permissions",
    "tool_invocations",
    "tool_stats",
    "approval_patterns",
    "procedures",
    "profile_blocks",
    "thread_summaries",
    "run_outcomes",
    "tools",
    "graph_relations",
    "graph_entities",
    "belief_sources",
    "beliefs",
    "belief_progress",
    "memories",
    "context_edges",
    "chunks",
    "document_nodes",
    "document_versions",
    "file_staging",
    "documents",
    "archive_segments",
    "job_outbox",
    "idempotency_keys",
    "revisions",
    "observations",
    "agent_runs",
    "message_attachments",
    "messages",
    "turns",
    "sessions",
    "threads",
]


from tests.conftest import PG_AVAILABLE  # noqa: E402
from tests.support_real import reset_real_backends  # noqa: E402

requires_pg = pytest.mark.skipif(
    not PG_AVAILABLE, reason="PostgreSQL not reachable at MEMORY_TEST_DATABASE_URL"
)


def integration_settings(make_settings, **overrides):
    base = {"database": {"url": DB_URL}}
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return make_settings(**base)


def integration_overrides(**changes) -> Overrides:
    """The hermetic stand-ins with the *recording* queue: integration tests drain jobs
    themselves, and the cache stays in-process even under ``MEMORY_TEST_PROVIDERS=env``."""
    return _test_overrides(**{"tasks": "memory", "cache": "memory", **changes})


@pytest.fixture
async def container(make_settings, tmp_path, request) -> AsyncIterator[Container]:
    if not PG_AVAILABLE:
        pytest.skip("PostgreSQL not reachable")
    settings: Settings = integration_settings(
        make_settings, blob={"filesystem_root": str(tmp_path / "blob")}
    )
    # blob=None: the filesystem store the settings point at tmp_path, which the archive
    # tests read from disk and corrupt on purpose; the memory stand-in has no files
    # A module may define a ``container_overrides`` fixture: stand-in choices (``Overrides``
    # fields) for its own container. Looked up rather than required, because modules outside
    # this directory import this fixture by name.
    try:
        changes = request.getfixturevalue("container_overrides")
    except pytest.FixtureLookupError:
        changes = {}
    c = await build_container(
        settings, __version__, overrides=integration_overrides(blob=None, **changes)
    )
    async with c.database.engine.begin() as conn:
        # a reset, not a hot path: on a loaded host it may outlast the statement timeout
        await conn.execute(text("SET LOCAL statement_timeout = 0"))
        await conn.execute(text("TRUNCATE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE"))
        await conn.execute(
            text("TRUNCATE procrastinate_jobs, procrastinate_events RESTART IDENTITY CASCADE")
        )
    await reset_real_backends(c)
    try:
        yield c
    finally:
        await c.close()


@pytest.fixture
def uow_factory(container: Container):
    return container.services["uow_factory"]


@pytest.fixture
def statements(container: Container) -> Iterator[list[str]]:
    """Every SQL statement the container's engine runs while the test holds this list: for
    asserting that a batch is one round trip, not one per item."""
    seen: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany) -> None:  # type: ignore[no-untyped-def]
        seen.append(" ".join(statement.split()))

    engine = container.database.engine.sync_engine
    event.listen(engine, "before_cursor_execute", record)
    try:
        yield seen
    finally:
        event.remove(engine, "before_cursor_execute", record)
