"""Bring memory an earlier build wrote up to this one, in place and while serving (ADR 0041).

Production memory is never cleared, and an earlier build left three things this one reads:

* **labels it reads otherwise** (``read_again``). Each statement is read again first, as the
  write path reads it: a sentence that gives an instruction for every time is marked a
  standing instruction (an earlier build marked none, and the read path then took a rule
  restated with a new value for two current rules), and, in a deployment without a model, a
  kind this build's labeller cannot give it is read again by the labeller.

* **values replaced across conversations that were never marked.** A store written before
  ADR 0038 (or before 0040's out-of-order rule) holds a user's old and new statements of one
  thing both current. Every user's own statements are read again, message by message in the
  order they were said (event time, never the order of storing), and each message goes
  through the same replacement the write path runs (``CrossConversationReplacement.run``): an
  older value becomes an earlier value (history) or is superseded, exactly as if it had been
  said after this build shipped. Nothing is deleted, and a replacement stays reversible.
* **index points without the payload this build filters on** (``statement_kind`` and
  ``standing_instruction``, the rule channel of ``context.standing``). Every memory of the users it read is indexed again
  (``Indexer.index_memories``: an upsert of the same ids); ``--reindex-all`` re-indexes the
  whole store with ``tools.reindex`` instead.

Safe on live data: one message per transaction, under the same per-user serialization the
write path takes, each one bumping the revisions its changes touch (so no cached bundle
serves a value it marked). Idempotent: a statement already marked is not marked again, and
a second run changes nothing. Resumable: ``--state FILE`` records every user finished and,
for the user in progress, the instant of the last message done; a run stopped anywhere
continues from there. Batched: ``--batch`` messages between checkpoints. Versioned: the state
records ``BACKFILL_REVISION``, and a state an earlier revision left starts over (a release that
changes what the backfill applies must reach the users an earlier one finished). Cheap when
there is nothing to do: with no user left to bring up (a fresh install, or every user finished
at this revision) it exits before loading a model, which is what lets compose run it on every
``docker compose up`` (the ``memory-backfill`` service).

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

#: The rules a backfill applies, as one number. Bump it whenever they change (what the
#: replacement decides, a label read again, what is indexed or enriched): a ``--state`` file
#: records it, and one from another revision starts over instead of skipping every user an
#: earlier revision finished.
BACKFILL_REVISION = 1
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
    relabelled: int = 0
    standing: int = 0
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "users": self.users,
            "messages": self.messages,
            "replaced": self.replaced,
            "indexed": self.indexed,
            "skipped_users": self.skipped_users,
            "relabelled": self.relabelled,
            "standing": self.standing,
            "failures": self.failures,
        }


class Checkpoint:
    """``--state``: the revision of the rules applied, the users finished, and the user in
    progress with the instant of the last message done. Written atomically (a file beside it,
    then renamed). A state from another revision (or none: written before revisions were
    recorded) is a run of other rules: this one starts over."""

    def __init__(self, path: Path | None, revision: int = BACKFILL_REVISION) -> None:
        self.path = path
        self.revision = revision
        self.data: dict[str, Any] = self._empty()
        if path is not None and path.exists():
            stored = json.loads(path.read_text())
            if stored.get("revision") == revision:
                self.data = stored
            else:
                log.info(
                    "backfill.state_restarted",
                    state=str(path),
                    stored_revision=stored.get("revision"),
                    revision=revision,
                )

    def _empty(self) -> dict[str, Any]:
        return {"revision": self.revision, "done": [], "current": None}

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
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
    database: Any, tenant: str | None, user: str | None
) -> list[tuple[str, str, str | None]]:
    """``(tenant, owner principal, workspace)`` of every user with current statements of
    their own, in a fixed order. ``database``: anything with a ``session_factory`` (the
    container's, or a bare ``Database`` before any model is loaded)."""
    async with database.session_factory() as session:
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


def without_a_model(container: Any) -> bool:
    """The deployment has no model (``LLMSettings.base_url`` unset): every statement kind it
    stored came from the lexicon and the NLI head, so this build's may read them again."""
    settings = getattr(container, "settings", None)
    llm = getattr(settings, "llm", None)
    return llm is not None and not getattr(llm, "base_url", None)


async def read_again(
    memory: CanonicalMemory, labeller: Any, *, relabel: bool
) -> CanonicalMemory | None:
    """``memory`` with what this build's write path reads in its words where an earlier
    build stored otherwise; None when nothing changes. Only a statement the labeller labelled
    (``statement_kind``: a turn, or what was read from one sentence of it) is read again.

    * **A standing instruction** (every deployment: the words alone decide it, a model never
      does): a sentence that gives an instruction it says is for every time ("Setz die
      Auftragsnummer immer in den E-Mail-Betreff ...", "أضف دائمًا ... في نسخة") is marked
      ``standing_instruction`` as this build marks it on the write path, so the read path
      treats it as a rule (the rules section, a rule restated with a new value) whatever kind
      the labeller left it. An earlier build never marked one.
    * **A kind this build's labeller cannot give it** (``relabel``: a deployment without a
      model, where every kind came from the lexicon and the NLI head): read again by the
      labeller, with the NLI head for what its lexicon leaves open, as the write path reads
      it (Hindi "आज से X संभालेंगे" was a rule by an earlier lexicon and is a dated fact or a
      correction by this one, ADR 0040). A kind it can still give (the lexicon's, or the
      kind it leaves to the head) is the same head's verdict and is kept. A rule reading this
      build reads no rule in becomes a fact.
    """
    from memory_service.modules.memory.native import split_sentences, standing_instruction

    meta = memory.system_metadata
    if not meta.get("statement_kind"):
        return None
    sentences = split_sentences(memory.content) or [memory.content]
    lexical = [labeller.lexicon.label(s) for s in sentences]
    fixed = dict(meta)
    if relabel:
        fixed |= await _kind_again(meta, sentences, lexical, labeller)
        if fixed.get("category") != "rule" and meta.get("category") == "rule":
            fixed.pop("standing_instruction", None)
    if any(standing_instruction(x) for x in lexical):
        fixed["standing_instruction"] = "yes"
    if fixed == meta:
        return None
    return memory.model_copy(update={"system_metadata": fixed})


async def _kind_again(
    meta: dict[str, Any], sentences: list[str], lexical: list[Any], labeller: Any
) -> dict[str, Any]:
    """The labels to change when this build's labeller cannot give the stored kind (none of
    the lexicon's kinds, nor a kind it leaves to the head): the labeller's own reading, and a
    rule reading it reads no rule in is a fact."""
    from memory_service.modules.memory.statements import RULE_KINDS, most_salient

    possible = {x.kind for x in lexical} | {x.maybe for x in lexical if not x.decided}
    if meta.get("statement_kind") in {k.value for k in possible if k is not None}:
        return {}
    kind = most_salient(label.kind for label in await labeller.label(sentences))
    if kind is None or kind.value == meta.get("statement_kind"):
        return {}
    out: dict[str, Any] = {"statement_kind": kind.value}
    if meta.get("category") == "rule" and kind not in RULE_KINDS:
        out["category"] = "fact"
    return out


async def _relabel(
    uow: Any,
    memories: list[CanonicalMemory],
    report: BackfillReport,
    labeller: Any,
    *,
    relabel: bool,
) -> list[CanonicalMemory]:
    """``memories`` read again (``read_again``), the changed ones stored."""
    out = []
    for memory in memories:
        fixed = await read_again(memory, labeller, relabel=relabel)
        if fixed is not None:
            await uow.memories.update(fixed)
            before, after = memory.system_metadata, fixed.system_metadata
            if before.get("statement_kind") != after.get("statement_kind"):
                report.relabelled += 1
            if before.get("standing_instruction") != after.get("standing_instruction"):
                report.standing += 1
        out.append(fixed or memory)
    return out


async def _index_again(
    container: Any, tenant: str, ids: Sequence[str], report: BackfillReport
) -> None:
    """``ids`` indexed again and their graph enrichment run again, as the write path's index
    job does: a memory no longer current takes its graph facts with it ("mentions" of an old
    value would come back as current)."""
    indexer = container.services["indexer"]
    graph = container.services.get("graph")
    for start in range(0, len(ids), 200):
        chunk = list(ids[start : start + 200])
        report.indexed += await indexer.index_memories(tenant, chunk)
        if graph is not None:
            await graph.enrich_memories(tenant, chunk)


async def _replace(
    uow: Any, replacement: Any, ctx: MemoryExecutionContext, said: list[Any], report: Any
) -> list[CanonicalMemory]:
    """One message's statements through the write path's replacement: what it changed."""
    if not said:
        return []
    outcome = await replacement.run(uow, ctx, said, now=datetime.now(UTC))
    report.replaced += len(outcome.changed)
    return list(outcome.changed)


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
    memories indexed and their graph enrichment run again (which retires the graph facts of
    the memories now replaced)."""
    from memory_service.modules.memory.pipeline import bump_memory_revisions
    from memory_service.modules.memory.statements import StatementLabeller

    pipeline = container.services["observation_pipeline"]
    replacement = pipeline.replacement
    replacing = bool(replacement is not None and replacement.enabled)
    # the kinds only with the head that decides them, as on the write path
    relabel = replacing and without_a_model(container)
    # the write path's labeller (the lexicon, and the NLI head for what it leaves open)
    labeller = getattr(pipeline.provider, "labeller", None) or StatementLabeller()
    user_id = principal.removeprefix("user:")
    ctx = MemoryExecutionContext(tenant_id=tenant, user_id=user_id, workspace_id=workspace)
    after = checkpoint.resume_after(tenant, principal)
    statements = await _statements(container, tenant, principal, after)
    touched: set[str] = {m.memory_id for m in statements}
    for n, said in enumerate(messages(statements), start=1):
        async with container.services["uow_factory"]() as uow:
            await uow.serialize(f"memory-user:{tenant}:{principal}")
            current = [
                m
                for m in await uow.memories.get_many(tenant, [m.memory_id for m in said])
                if m.temporal.status is TemporalStatus.CURRENT and m.deleted_at is None
            ]
            read = await _relabel(uow, current, report, labeller, relabel=relabel)
            changed = [m for m, before in zip(read, current, strict=True) if m is not before]
            if replacing:
                changed += await _replace(uow, replacement, ctx, read, report)
            touched |= {m.memory_id for m in changed}
            if changed:
                await bump_memory_revisions(uow, changed)
            await uow.commit()
        report.messages += 1
        if n % batch == 0:
            checkpoint.progress(tenant, principal, said[0].temporal.observed_at.isoformat())
    await _index_again(container, tenant, sorted(touched), report)
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
    for t, principal, workspace in await users(container.database, tenant, user):
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


async def pending_users(
    settings: Any, *, state: Path | None, tenant: str | None, user: str | None
) -> tuple[int, int]:
    """``(users, of them not finished at this revision)``, read with a bare database
    connection: the models the backfill needs are loaded only when one is left."""
    from memory_service.adapters.db.engine import Database

    database = Database(settings.database)
    try:
        found = await users(database, tenant, user)
    finally:
        await database.close()
    checkpoint = Checkpoint(state)
    return len(found), sum(1 for t, p, _ in found if not checkpoint.finished(t, p))


async def _main(args: argparse.Namespace) -> int:
    from memory_service.application.container import build_container
    from memory_service.config.settings import Settings
    from memory_service.modules.jobs.registry import register_handlers

    settings = Settings()
    state = Path(args.state) if args.state else None
    if not args.reindex_all:
        found, left = await pending_users(settings, state=state, tenant=args.tenant, user=args.user)
        if not left:
            # a fresh install, or every user finished at this revision: nothing to load
            report = BackfillReport(skipped_users=found)
            sys.stdout.write(json.dumps(report.as_dict(), indent=2) + "\n")
            return 0
    container = await build_container(settings, __version__)
    register_handlers(container)
    try:
        report = await backfill(
            container,
            state=state,
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
    parser = argparse.ArgumentParser(description=(__doc__ or "backfill").strip().splitlines()[0])
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
