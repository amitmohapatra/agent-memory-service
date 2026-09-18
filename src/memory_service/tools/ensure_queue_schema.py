"""Apply the Procrastinate schema.

Alembic owns the application tables; Procrastinate owns its own and installs them through its
own API. Both have to exist before the API or the worker starts, and neither creates them on
boot — so a fresh deployment ran until the first job was enqueued and then failed on a
missing table. This is the second half of the migration step.

Idempotent: Procrastinate skips a schema that is already applied.
"""

from __future__ import annotations

import asyncio

from memory_service.config.settings import get_settings
from memory_service.observability.logging import get_logger

log = get_logger(__name__)


async def _apply() -> None:
    settings = get_settings()
    if settings.tasks.provider != "procrastinate":
        log.info("queue_schema.skipped", provider=settings.tasks.provider)
        return
    from memory_service.adapters.tasks.procrastinate_queue import ProcrastinateTaskQueue

    dsn = str(settings.database.url).replace("postgresql+psycopg://", "postgresql://")
    queue = ProcrastinateTaskQueue(dsn)
    try:
        await queue.ensure_schema()
        log.info("queue_schema.ready")
    finally:
        await queue.close()


def main() -> None:
    asyncio.run(_apply())


if __name__ == "__main__":
    main()
