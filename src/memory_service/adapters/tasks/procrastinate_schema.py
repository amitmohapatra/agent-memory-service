"""Apply the Procrastinate schema (``make migrate`` runs this after Alembic)."""

from __future__ import annotations

import asyncio

from memory_service.adapters.tasks.procrastinate_queue import ProcrastinateTaskQueue
from memory_service.config.settings import get_settings


async def main() -> None:
    settings = get_settings()
    queue = ProcrastinateTaskQueue(settings.database.procrastinate_dsn)
    try:
        await queue.ensure_schema()
    finally:
        await queue.close()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
