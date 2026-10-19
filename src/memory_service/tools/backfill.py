"""Bring memory an earlier build wrote up to this one, in place and while serving (ADR 0041).

Production memory is never cleared, and an earlier build left two things this one reads:

* **values replaced across conversations that were never marked.** A store written before
  ADR 0038 (or before 0040's out-of-order rule) holds a user's old and new statements of one
  thing both current. Every user's own statements are read again, message by message in the
  order they were said (event time, never the order of storing), and each message goes
  through the same replacement the write path runs (``CrossConversationReplacement.run``): an
  older value becomes an earlier value (history) or is superseded, exactly as if it had been
  said after this build shipped. Nothing is deleted, and a replacement stays reversible.
* **index points without the payload this build filters on** (``statement_kind``, the
  rule channel of ``context.standing``). Every memory of the users it read is indexed again
  (``Indexer.index_memories``: an upsert of the same ids); ``--reindex-all`` re-indexes the
  whole store with ``tools.reindex`` instead.

Safe on live data: one message per transaction, under the same per-user serialization the
write path takes, each one bumping the revisions its changes touch (so no cached bundle
serves a value it marked). Idempotent: a statement already marked is not marked again, and
a second run changes nothing. Resumable: ``--state FILE`` records every user finished and,
for the user in progress, the instant of the last message done; a run stopped anywhere
continues from there. Batched: ``--batch`` messages between checkpoints.

    uv run python -m memory_service.tools.backfill --state /var/lib/memory/backfill.json \\
        [--tenant acme] [--user u1] [--batch 50] [--reindex-all]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from memory_service.__about__ import __version__
from memory_service.adapters.db.orm import MemoryRow
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import TemporalStatus, Visibility
from memory_service.domain.memory import CanonicalMemory
from memory_service.observability.logging import get_logger

log = get_logger(__name__)

#: messages between two checkpoints of ``--state``
BATCH = 50
#: a user's statements read per query (oldest first)
PAGE = 500


@dataclass
class BackfillReport:
    users: int = 0
    messages: int = 0
    replaced: int = 0
    indexed: int = 0
    skipped_users: int = 0
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "users": self.users,
            "messages": self.messages,
            "replaced": self.replaced,
            "indexed": self.indexed,
            "skipped_users": self.skipped_users,
            "failures": self.failures,
        }


class Checkpoint:
    """``--state``: the users finished, and the user in progress with the instant of the last
    message done. Written atomically (a file beside it, then renamed)."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.data: dict[str, Any] = {"done": [], "current": None}
        if path is not None and path.exists():
            self.data = json.loads(path.read_text())

    def finished(self, tenant: str, principal: str) -> bool:
        return [tenant, principal] in self.data["done"]

    def resume_after(self, tenant: str, principal: str) -> str | None:
        current = self.data.get("current")
        if current and current["user"] == [tenant, principal]:
            return str(current["after"])
        return None

    def progress(self, tenant: str, principal: str, after: str) -> None:
        self.data["current"] = {"user": [tenant, principal], "after": after}
        self._write()

    def finish(self, tenant: str, principal: str) -> None:
        self.data["done"].append([tenant, principal])
        self.data["current"] = None
        self._write()

    def _write(self) -> None:
        if self.path is None:
            return
        part = self.path.with_suffix(self.path.suffix + ".part")
        part.write_text(json.dumps(self.data))
        part.replace(self.path)


def messages(statements: Sequence[CanonicalMemory]) -> list[list[CanonicalMemory]]:
    """A user's statements grouped by the message they were read from, in the order the
    messages were said (their instant, then the message id: two said in one instant keep one
    order on every run)."""
    by_message: dict[tuple[str, ...], list[CanonicalMemory]] = {}
    for m in statements:
        sources = tuple(sorted(e.source_id for e in m.evidence if e.source_type != "memory"))
        if sources:
            by_message.setdefault(sources, []).append(m)
    return sorted(
        by_message.values(),
        key=lambda group: (
            min(m.temporal.observed_at for m in group),
            group[0].evidence[0].source_id,
        ),
    )


async def users(
    container: Any, tenant: str | None, user: str | None
) -> list[tuple[str, str, str | None]]:
    """``(tenant, owner principal, workspace)`` of every user with current statements of
    their own, in a fixed order."""
    async with container.database.session_factory() as session:
        q = (
            select(MemoryRow.tenant_id, MemoryRow.owner_principal, MemoryRow.workspace_id)
            .where(
                MemoryRow.deleted_at.is_(None),
                MemoryRow.temporal_status == TemporalStatus.CURRENT.value,
                MemoryRow.visibility == Visibility.USER.value,
                MemoryRow.owner_principal.startswith("user:"),
            )
            .distinct()
        )
        if tenant:
            q = q.where(MemoryRow.tenant_id == tenant)
        if user:
            q = q.where(MemoryRow.owner_principal == f"user:{user}")
        rows = (await session.execute(q)).all()
    seen: dict[tuple[str, str], str | None] = {}
    for t, principal, workspace in rows:
        seen.setdefault((t, principal), workspace)
    return sorted((t, p, w) for (t, p), w in seen.items())


async def _statements(
    container: Any, tenant: str, principal: str, after: str | None
) -> list[CanonicalMemory]:
    """The user's current statements said after ``after`` (an ISO instant), oldest first."""
    out: list[CanonicalMemory] = []
    async with container.services["uow_factory"]() as uow:
        rows = await uow.memories.statements_of(
            tenant,
            principal,
            before=datetime.max.replace(tzinfo=UTC),
            limit=1_000_000,
            after=datetime.fromisoformat(after) if after else None,
        )
    out = sorted(rows, key=lambda m: (m.temporal.observed_at, m.memory_id))
    return out


async def backfill_user(
    container: Any,
    tenant: str,
    principal: str,
    workspace: str | None,
    *,
    checkpoint: Checkpoint,
    report: BackfillReport,
    batch: int = BATCH,
) -> None:
    """One user: every message of theirs through the replacement, oldest first, then their
    memories indexed again."""
    from memory_service.modules.memory.pipeline import bump_memory_revisions

    pipeline = container.services["observation_pipeline"]
    replacement = pipeline.replacement
    user_id = principal.removeprefix("user:")
    ctx = MemoryExecutionContext(tenant_id=tenant, user_id=user_id, workspace_id=workspace)
    after = checkpoint.resume_after(tenant, principal)
    statements = await _statements(container, tenant, principal, after)
    touched: set[str] = {m.memory_id for m in statements}
    for n, said in enumerate(messages(statements), start=1):
        if replacement is not None and replacement.enabled:
            async with container.services["uow_factory"]() as uow:
                await uow.serialize(f"memory-user:{tenant}:{principal}")
                current = [
                    m
                    for m in await uow.memories.get_many(tenant, [m.memory_id for m in said])
                    if m.temporal.status is TemporalStatus.CURRENT and m.deleted_at is None
                ]
                if current:
                    outcome = await replacement.run(uow, ctx, current, now=datetime.now(UTC))
                    if outcome.changed:
                        report.replaced += len(outcome.changed)
                        touched |= {m.memory_id for m in outcome.changed}
                        await bump_memory_revisions(uow, outcome.changed)
                await uow.commit()
        report.messages += 1
        if n % batch == 0:
            checkpoint.progress(tenant, principal, said[0].temporal.observed_at.isoformat())
    indexer = container.services["indexer"]
    ids = sorted(touched)
    for start in range(0, len(ids), 200):
        report.indexed += await indexer.index_memories(tenant, ids[start : start + 200])
    checkpoint.finish(tenant, principal)
    report.users += 1


async def backfill(
    container: Any,
    *,
    state: Path | None = None,
    tenant: str | None = None,
    user: str | None = None,
    batch: int = BATCH,
) -> BackfillReport:
    report = BackfillReport()
    checkpoint = Checkpoint(state)
    for t, principal, workspace in await users(container, tenant, user):
        if checkpoint.finished(t, principal):
            report.skipped_users += 1
            continue
        try:
            await backfill_user(
                container,
                t,
                principal,
                workspace,
                checkpoint=checkpoint,
                report=report,
                batch=batch,
            )
        except Exception as exc:  # one user's failure is reported; the others still run
            report.failures.append(f"{t}/{principal}: {type(exc).__name__}: {exc}")
            log.warning("backfill.user_failed", tenant_id=t, principal=principal, error=str(exc))
    log.info("backfill.done", **report.as_dict())
    return report


async def _main(args: argparse.Namespace) -> int:
    from memory_service.application.container import build_container
    from memory_service.config.settings import Settings
    from memory_service.modules.jobs.registry import register_handlers

    container = await build_container(Settings(), __version__)
    register_handlers(container)
    try:
        report = await backfill(
            container,
            state=Path(args.state) if args.state else None,
            tenant=args.tenant,
            user=args.user,
            batch=args.batch,
        )
        if args.reindex_all:
            from memory_service.tools.reindex import rebuild_search_index

            rebuilt = await rebuild_search_index(container, tenant_id=args.tenant)
            report.failures += rebuilt.failures
        sys.stdout.write(json.dumps(report.as_dict(), indent=2) + "\n")
        return 0 if not report.failures else 1
    finally:
        await container.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", default=None, help="checkpoint file: resume where a run stopped")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--user", default=None, help="one user id (with --tenant)")
    parser.add_argument("--batch", type=int, default=BATCH, help="messages between checkpoints")
    parser.add_argument(
        "--reindex-all", action="store_true", help="re-index the whole store after (tools.reindex)"
    )
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
