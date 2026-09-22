"""How many processes the service runs and how many threads each of them may use.

None of this is visible from a request, and all of it decides whether 20 rps is served or
queued: one process is one GIL, and N processes each fanning their BLAS calls over every
core is thrash rather than throughput. The numbers live in three places that have to agree -
the setting the entrypoint reads, the image that runs it, and the compose file that caps the
ingestion container - so they are asserted together.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from memory_service.config.settings import Settings

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


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


def test_the_image_pins_workers_and_math_threads() -> None:
    dockerfile = (ROOT / "deploy" / "Dockerfile").read_text()
    for declaration in (
        "WEB_CONCURRENCY=3",
        "MEMORY__SERVICE__WORKERS=3",
        "OMP_NUM_THREADS=2",
        "MKL_NUM_THREADS=2",
    ):
        assert declaration in dockerfile, declaration
    # the two names for the same number must not be able to drift apart
    assert Settings(_env_file=None).service.workers == 3


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
