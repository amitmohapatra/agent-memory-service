"""Entrypoints: ``memory-api`` and ``memory-worker``."""

from __future__ import annotations

import asyncio

import uvicorn

from memory_service.config.constants import HOST
from memory_service.config.settings import get_settings


def run_api() -> None:
    """Run the API on ``service.workers`` processes.

    One process is one GIL for tokenising, pydantic, JSON and the search client's parsing,
    and that alone exceeds a core at the target rate. Each worker imports the factory in its
    own interpreter, so each builds its own container: its own model set, its own database
    pool, its own readiness. Nothing is shared but the sockets.
    """
    settings = get_settings()
    uvicorn.run(
        "memory_service.api.app:create_app",
        factory=True,
        host=HOST,
        port=settings.service.port,
        workers=settings.service.workers,
        log_level=settings.service.log_level.lower(),
        access_log=False,
    )


def run_worker() -> None:
    from memory_service.worker import main

    asyncio.run(main())


if __name__ == "__main__":  # pragma: no cover
    run_api()
