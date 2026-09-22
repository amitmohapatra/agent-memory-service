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

from memory_service.config.settings import Settings

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _no_inherited_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    """WEB_CONCURRENCY is read from the real environment, so the default is only the default
    when whatever ran this suite did not already set it."""
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)


def test_the_api_runs_the_configured_number_of_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    from memory_service import __main__
    from memory_service.config import settings as settings_module

    captured: dict[str, Any] = {}

    def _fake_run(app: str, **kwargs: Any) -> None:
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", _fake_run)
    monkeypatch.setattr(__main__, "get_settings", lambda: settings_module.Settings(_env_file=None))
    __main__.run_api()
    assert captured["workers"] == 3, "the default is three workers, not one"
    assert captured["factory"] is True, "each worker builds its own app, and its own container"


def test_the_worker_count_is_bounded() -> None:
    assert Settings(_env_file=None).service.workers == 3
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
    assert Settings(_env_file=None).service.workers == 3, "unset by a platform, not a failure"
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
    pool is what has to fit, not a single service-wide number."""
    database = Settings(_env_file=None).database
    assert (database.pool_size, database.max_overflow) == (8, 8)
    assert (database.pool_size + database.max_overflow) * 4 < 200, "max_connections in compose"
