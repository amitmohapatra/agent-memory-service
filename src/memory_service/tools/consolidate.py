"""Consolidate a tenant's memories into beliefs now, through the worker's own job.

What ``memory.process_observation`` does for each new fact, done for every fact a tenant
already holds: every (scope, person) unit its memories touch (``units_touched``, as the write
path computes it) is queued with ``schedule``, and the queue is drained through the
registered handlers - ``memory.consolidate`` (one model call per run of at most
``FACTS_PER_CALL`` facts, then the citation check), its continuations, and ``belief.index``.
Nothing here is a second implementation: a corpus ingested before beliefs existed ends where
a live deployment would.

The model is the one the service resolves for the tenant (``modules.llm.endpoints``) and the
tenant's policy must allow the ``reflection`` use. ``--rebuild`` makes every unit's next run a
full rebuild (its beliefs replaced from all of its facts). Runs in-process (the recording
queue), one unit at a time, and prints a JSON report.

    uv run python -m memory_service.tools.consolidate --tenant bench_conv [--rebuild]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from memory_service.__about__ import __version__
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.llm.cost import llm_accounting
from memory_service.modules.memory.consolidation import (
    Unit,
    schedule,
    units_touched,
)

#: memory ids read per page while finding the units
_PAGE = 500


async def tenant_units(uow_factory: Any, tenant_id: str) -> list[Unit]:
    """Every unit the tenant's live memories touch, as the write path finds them."""
    async with uow_factory() as uow:
        known = set(
            await uow.session.scalars(
                text(
                    "SELECT DISTINCT subject FROM memories WHERE tenant_id = :t "
                    "AND subject LIKE 'user:%' AND deleted_at IS NULL"
                ),
                {"t": tenant_id},
            )
        )
    found: set[Unit] = set()
    after = ""
    while True:
        async with uow_factory() as uow:
            ids = list(
                await uow.session.scalars(
                    text(
                        "SELECT memory_id FROM memories WHERE tenant_id = :t AND memory_id > :a "
                        "AND deleted_at IS NULL ORDER BY memory_id LIMIT :n"
                    ),
                    {"t": tenant_id, "a": after, "n": _PAGE},
                )
            )
            if not ids:
                return sorted(found)
            found |= units_touched(await uow.memories.get_many(tenant_id, ids), known)
        after = ids[-1]


async def consolidate_tenant(
    container: Any, tenant_id: str, *, rebuild: bool = False
) -> dict[str, Any]:
    """Queue every unit of the tenant and drain the in-process queue (``container.tasks``
    must be the recording queue: a benchmark's, or this tool's)."""
    register_handlers(container)
    uow_factory = container.services["uow_factory"]
    units = await tenant_units(uow_factory, tenant_id)
    now = datetime.now(UTC)
    async with uow_factory() as uow:
        for scope_key, person_key in units:
            progress = await uow.beliefs.progress(tenant_id, scope_key, person_key, lock=True)
            if rebuild and progress is not None:
                await uow.beliefs.save_progress(
                    progress.model_copy(
                        update={"last_consolidated_at": None, "failures": 0, "paused_at": None}
                    )
                )
            await schedule(uow, tenant_id, scope_key, person_key, now=now)
        await uow.commit()
    with llm_accounting() as tokens:
        jobs = await container.tasks.drain(max_rounds=100_000)
    async with uow_factory() as uow:
        beliefs = await uow.session.scalar(
            text(
                "SELECT count(*) FROM beliefs WHERE tenant_id = :t AND deleted_at IS NULL "
                "AND superseded_by IS NULL AND valid_to IS NULL"
            ),
            {"t": tenant_id},
        )
    return {
        "tenant_id": tenant_id,
        "units": len(units),
        "jobs": jobs,
        "current_beliefs": int(beliefs or 0),
        "input_tokens": tokens.input,
        "output_tokens": tokens.output,
    }


async def _main(args: argparse.Namespace) -> None:
    from memory_service.application.container import Overrides, build_container
    from memory_service.config.settings import get_settings

    container = await build_container(
        get_settings(), __version__, overrides=Overrides(tasks="memory")
    )
    try:
        report = await consolidate_tenant(container, args.tenant, rebuild=args.rebuild)
    finally:
        await container.close()
    sys.stdout.write(json.dumps(report) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--rebuild", action="store_true", help="replace every unit's beliefs")
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
