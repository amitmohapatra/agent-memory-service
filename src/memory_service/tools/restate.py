"""Restate the conversation turns stored before the model could (ADR 0027).

``memory_restatement`` runs at ingest, so turns a tenant stored before it registered a key or
named the use in its policy keep the key they were indexed with. This walks those turns -
CURRENT verbatim turns with no restatement yet - restates each through the gateway exactly
as ingest would (bound to the turn's owner: their key pays, their tenant's policy decides),
stores the restatement and its relations, then re-indexes the turns and re-links them in the
graph::

    uv run python -m memory_service.tools.restate --tenant acme [--limit 500] [--force]

``--force`` restates turns that already have a restatement (after a model change). A turn
the model was not allowed to restate, or whose output failed the checks, is left as it was
and counted as skipped; nothing about it changes.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any

from sqlalchemy import select

from memory_service.__about__ import __version__
from memory_service.adapters.db.orm import MemoryRow
from memory_service.application.container import Container, build_container
from memory_service.config.settings import Settings
from memory_service.modules.memory.restatement import restate
from memory_service.observability.logging import get_logger
from memory_service.ports.credentials import ModelIdentity

log = get_logger(__name__)

#: turns restated, then indexed and linked, per batch
BATCH = 50


@dataclass
class RestateReport:
    restated: int = 0
    skipped: int = 0
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"restated": self.restated, "skipped": self.skipped, "failures": self.failures}


async def restate_turns(
    container: Container, *, tenant_id: str, limit: int | None = None, force: bool = False
) -> RestateReport:
    """Restate ``tenant_id``'s stored turns (those without a restatement, or all with
    ``force``), then re-index and re-link the ones that changed."""
    report = RestateReport()
    ids = await _turns(container, tenant_id, limit=limit, force=force)
    log.info("restate.tenant", tenant_id=tenant_id, turns=len(ids))
    for start in range(0, len(ids), BATCH):
        changed: list[str] = []
        for memory_id in ids[start : start + BATCH]:
            try:
                if await _restate_one(container, tenant_id, memory_id):
                    changed.append(memory_id)
                    report.restated += 1
                else:
                    report.skipped += 1
            except Exception as exc:
                report.failures.append(f"memory {tenant_id}/{memory_id}: {exc}")
        if changed:
            await container.services["indexer"].index_memories(tenant_id, changed)
            await container.services["graph"].enrich_memories(tenant_id, changed)
    log.info("restate.done", tenant_id=tenant_id, **report.as_dict())
    return report


async def _turns(
    container: Container, tenant_id: str, *, limit: int | None, force: bool
) -> list[str]:
    query = (
        select(MemoryRow.memory_id)
        .where(
            MemoryRow.tenant_id == tenant_id,
            MemoryRow.temporal_status == "CURRENT",
            MemoryRow.deleted_at.is_(None),
            MemoryRow.system_metadata["category"].astext == "verbatim_turn",
        )
        .order_by(MemoryRow.observed_at, MemoryRow.memory_id)
    )
    if not force:
        query = query.where(MemoryRow.system_metadata["restatement"].astext.is_(None))
    if limit:
        query = query.limit(limit)
    async with container.database.session_factory() as session:
        return list((await session.execute(query)).scalars())


async def _restate_one(container: Container, tenant_id: str, memory_id: str) -> bool:
    """Whether the model restated the turn (it is then stored, unindexed)."""
    assist = container.services["llm_assist"]
    async with container.database.session_factory() as session:
        row = await session.get(MemoryRow, memory_id)
        if row is None or row.tenant_id != tenant_id:
            return False
        meta = dict(row.system_metadata or {})
        before = meta.get("preceding_turn")
        said_at = row.observed_at if row.observed_at.tzinfo else row.observed_at.replace(tzinfo=UTC)
        async with assist.bound(ModelIdentity(tenant_id, row.owner_principal)):
            said = await restate(
                assist,
                text=row.content,
                speaker=row.user_id or row.agent_id or "",
                said_at=said_at,
                before=before if isinstance(before, dict) else None,
            )
        if said is None:
            return False
        meta.pop("restatement", None)
        meta.pop("restatement_relations", None)
        if said.text:
            meta["restatement"] = said.text
        if said.relations:
            meta["restatement_relations"] = [list(r) for r in said.relations]
        # the content is unchanged, so nothing derived from the turn is invalidated: only
        # its index key and its graph links are rebuilt
        row.system_metadata = meta
        row.indexed_at = None
        await session.commit()
    return True


async def _main(args: argparse.Namespace) -> int:
    container = await build_container(Settings(), __version__)
    try:
        report = await restate_turns(
            container, tenant_id=args.tenant, limit=args.limit, force=args.force
        )
    finally:
        await container.close()
    sys.stdout.write(f"{report.as_dict()}\n")
    return 0 if not report.failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Restate stored conversation turns")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--limit", type=int, default=None, help="at most this many turns")
    parser.add_argument(
        "--force", action="store_true", help="also restate turns that already have one"
    )
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
