"""How many processes the service runs and how many threads each of them may use.

None of this is visible from a request, and all of it decides whether 20 rps is served or
queued: one process is one GIL, and N processes each fanning their BLAS calls over every
core is thrash rather than throughput. The numbers live in three places that have to agree -
the setting the entrypoint reads, the image that runs it, and the compose file that caps the
ingestion container - so they are asserted together. There is one name for the worker count,
WEB_CONCURRENCY, and the test below sets it rather than comparing two literals: a number
asserted equal in two files still drifts from the number the process actually runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from memory_service.config.constants import DATABASE, OVERLOAD
from memory_service.config.settings import Settings, default_parallelism

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _no_inherited_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    """WEB_CONCURRENCY is read from the real environment, so the default is only the default
    when whatever ran this suite did not already set it."""
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)


def test_the_api_runs_the_configured_number_of_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import uvicorn

    from memory_service import __main__
    from memory_service.config import settings as settings_module

    captured: dict[str, Any] = {}

    def _fake_run(app: str, **kwargs: Any) -> None:
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", _fake_run)
    monkeypatch.setattr(__main__, "get_settings", lambda: settings_module.Settings(_env_file=None))
    # run_api points the workers at a fresh metrics directory; keep it out of this process
    monkeypatch.setenv(__main__.MULTIPROC_ENV, str(tmp_path / "metrics"))
    monkeypatch.setattr(settings_module, "available_cpus", lambda: 3)
    __main__.run_api()
    assert captured["workers"] == 3, "unset, one worker per available CPU"
    assert captured["factory"] is True, "each worker builds its own app, and its own container"
    # the outermost overload bound (ADR 0031): uvicorn refuses past it before the app runs
    assert captured["limit_concurrency"] == OVERLOAD.limit_concurrency
    assert captured["backlog"] == OVERLOAD.backlog


@pytest.mark.parametrize(("cpus", "expected"), [(1, 1), (2, 2), (6, 6), (64, 8)])
def test_unset_parallelism_follows_the_machine(
    monkeypatch: pytest.MonkeyPatch, cpus: int, expected: int
) -> None:
    """A constant three oversubscribed a two-CPU pod and idled most of a sixteen-CPU one."""
    from memory_service.config import settings as settings_module

    monkeypatch.setattr(settings_module, "available_cpus", lambda: cpus)
    settings = Settings(_env_file=None)
    assert settings.service.workers == expected
    assert settings.tasks.worker_concurrency == expected


def test_available_cpus_is_never_more_than_the_affinity() -> None:
    import os

    from memory_service.config.settings import available_cpus

    assert 1 <= available_cpus() <= (os.cpu_count() or 1)


def test_the_worker_count_is_bounded() -> None:
    assert 1 <= Settings(_env_file=None).service.workers <= 8
    with pytest.raises(ValueError, match="workers"):
        Settings(_env_file=None, service={"workers": 0})
    with pytest.raises(ValueError, match="workers"):
        Settings(_env_file=None, service={"workers": 9})


def test_web_concurrency_is_the_worker_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """The name the ecosystem sets has to be the name that decides, or an operator who caps
    a container at one worker silently keeps getting three. Asserting that two literals in
    two files both read 3 cannot catch that; running the settings with the variable set can.
    """
    monkeypatch.setenv("WEB_CONCURRENCY", "1")
    assert Settings(_env_file=None).service.workers == 1
    monkeypatch.setenv("WEB_CONCURRENCY", "")
    assert Settings(_env_file=None).service.workers == default_parallelism(), (
        "unset by a platform, not a failure"
    )
    monkeypatch.setenv("WEB_CONCURRENCY", "16")
    with pytest.raises(ValueError, match="workers"):
        Settings(_env_file=None)
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    assert Settings(_env_file=None, service={"workers": 4}).service.workers == 4, (
        "MEMORY__SERVICE__WORKERS still overrides it"
    )


def test_the_image_pins_workers_and_math_threads() -> None:
    dockerfile = (ROOT / "deploy" / "Dockerfile").read_text()
    for declaration in ("WEB_CONCURRENCY=3", "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2"):
        assert declaration in dockerfile, declaration
    declared = "\n".join(
        line for line in dockerfile.splitlines() if not line.lstrip().startswith("#")
    )
    assert "MEMORY__SERVICE__WORKERS" not in declared, (
        "one name for the worker count in the image; the second one is what drifts"
    )


def _compose_service(name: str) -> str:
    """One service block out of docker-compose.yml, read as text.

    Parsed rather than loaded because PyYAML is not a dependency of this project, and a test
    that pins the deployment must not be the reason one is added.
    """
    lines = (ROOT / "docker-compose.yml").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"  {name}:"))
    end = next(
        (
            i
            for i, line in enumerate(lines[start + 1 :], start + 1)
            if line.startswith("  ") and not line.startswith("   ") and line.rstrip().endswith(":")
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def test_ingestion_cannot_take_the_cores_the_query_path_is_measured_on() -> None:
    worker = _compose_service("memory-worker")
    assert "cpus: '2'" in worker and 'OMP_NUM_THREADS: "1"' in worker
    assert "deploy:" not in _compose_service("memory-api"), (
        "the API is the service being measured; it is not capped"
    )


def test_the_pools_are_per_process() -> None:
    """Three API workers plus the ingestion worker share one PostgreSQL: the per-process
    pool is what has to fit, not a single service-wide number. Unset, a process keeps the
    sizes the pools had before the budget existed."""
    database = Settings(_env_file=None).database
    plan = database.pool_plan(3)
    assert (plan.main_size, plan.main_overflow) == (8, 8)
    assert (plan.graph_size, plan.graph_overflow, plan.queue_max) == (4, 4, 4)
    assert plan.total * 4 < 200, "max_connections in compose"


@pytest.mark.parametrize("budget", [6, 28, 60, 84, 200])
@pytest.mark.parametrize("processes", [1, 3, 8])
def test_a_pod_budget_is_split_and_never_exceeded(budget: int, processes: int) -> None:
    """One number per pod (ADR 0031): each process takes ``budget // processes`` and splits
    it 4:2:1 between requests, the graph traversal and the task queue."""
    database = Settings(_env_file=None, database={"connection_budget": budget}).database
    plan = database.pool_plan(processes)
    share = max(6, budget // processes)
    assert plan.total <= share
    assert plan.main_size >= 1 and plan.graph_size >= 1 and plan.queue_max >= 2
    assert plan.main_size + plan.main_overflow >= plan.graph_size + plan.graph_overflow


def test_the_compose_app_goes_through_pgbouncer_and_the_queue_does_not() -> None:
    """Transaction pooling breaks LISTEN/NOTIFY and session parameters, so the request path
    goes through PgBouncer and the session work (queue, graph traversal, migrations) direct."""
    compose = (ROOT / "docker-compose.yml").read_text()
    assert "pgbouncer:6432" in compose
    assert "MEMORY__DATABASE__TRANSACTION_POOLER" in compose
    assert "MEMORY__DATABASE__DIRECT_URL: postgresql+psycopg://memory:memory@postgres:5432" in (
        compose
    )
    ini = (ROOT / "deploy" / "pgbouncer" / "pgbouncer.ini").read_text()
    assert "pool_mode = transaction" in ini
    assert f"statement_timeout={DATABASE.statement_timeout_ms}" in ini


def test_behind_a_transaction_pooler_nothing_outlives_a_transaction() -> None:
    from memory_service.adapters.db.engine import connect_args

    direct = Settings(_env_file=None).database
    pooled = Settings(_env_file=None, database={"transaction_pooler": True}).database
    assert connect_args(direct, statement_timeout_ms=15_000)["options"] == (
        "-c statement_timeout=15000"
    )
    args = connect_args(pooled, statement_timeout_ms=15_000)
    assert args["prepare_threshold"] is None, "no server-side prepared statements"
    assert "options" not in args, "PgBouncer refuses session parameters at startup"


def test_session_work_goes_direct_and_requests_through_the_pooler() -> None:
    from sqlalchemy.engine import make_url

    from tests.conftest import DB_URL

    pooled = make_url(DB_URL).set(host="pgbouncer", port=6432)
    database = Settings(
        _env_file=None,
        database={
            "url": pooled.render_as_string(hide_password=False),
            "direct_url": DB_URL,
            "transaction_pooler": True,
        },
    ).database
    assert "pgbouncer:6432" in database.dsn
    assert database.procrastinate_dsn == DB_URL.replace("+psycopg", "")
    assert database.sync_url == DB_URL, "migrations take locks: direct"
    plain = Settings(_env_file=None, database={"url": DB_URL}).database
    assert plain.direct_dsn == plain.dsn, "no pooler: one URL for everything"
