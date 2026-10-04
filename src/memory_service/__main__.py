"""Entrypoints: ``memory-api`` and ``memory-worker``."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import uvicorn

from memory_service.config.constants import HOST, OVERLOAD
from memory_service.config.settings import get_settings

#: where the API's worker processes write their metric values for one /metrics to read
MULTIPROC_ENV = "PROMETHEUS_MULTIPROC_DIR"


def prepare_multiprocess_metrics(workers: int) -> str | None:
    """Give the API's worker processes one metrics directory, emptied at start.

    prometheus_client keeps values in process memory unless ``PROMETHEUS_MULTIPROC_DIR`` is
    set when it is first imported; then each process writes its values to files there and
    ``MultiProcessCollector`` sums them on a scrape (``observability.metrics``). It must be
    set here, before uvicorn spawns the workers, and emptied, or a restarted container adds
    the previous run's counters to its own. One worker needs none of it.
    """
    if workers <= 1:
        return None
    directory = Path(
        os.environ.get(MULTIPROC_ENV) or Path(tempfile.gettempdir()) / "memory-metrics"
    )
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[MULTIPROC_ENV] = str(directory)
    return str(directory)


def run_api() -> None:
    """Run the API on ``service.workers`` processes.

    One process is one GIL for tokenising, pydantic, JSON and the search client's parsing,
    and that alone exceeds a core at the target rate. Each worker imports the factory in its
    own interpreter, so each builds its own container: its own model set, its own database
    pool, its own readiness. Nothing is shared but the sockets and the metrics directory.

    ``limit_concurrency`` and ``backlog`` are the outermost overload bound (ADR 0031): past
    the first, uvicorn answers 503 before any application code runs; the second is how many
    connections the kernel holds for accept.
    """
    settings = get_settings()
    prepare_multiprocess_metrics(settings.service.workers)
    uvicorn.run(
        "memory_service.api.app:create_app",
        factory=True,
        host=HOST,
        port=settings.service.port,
        workers=settings.service.workers,
        log_level=settings.service.log_level.lower(),
        access_log=False,
        limit_concurrency=OVERLOAD.limit_concurrency,
        backlog=OVERLOAD.backlog,
        # above every client's keep-alive (the SDK's is 30 s) and LB idle timeouts
        timeout_keep_alive=OVERLOAD.keep_alive_seconds,
        # in-flight requests get their longest deadline to finish on SIGTERM
        timeout_graceful_shutdown=int(OVERLOAD.write_deadline_seconds) + 5,
    )


def run_worker() -> None:
    from memory_service.worker import main

    asyncio.run(main())


if __name__ == "__main__":  # pragma: no cover
    run_api()
