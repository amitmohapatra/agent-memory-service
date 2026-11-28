"""Forget what a model says was never the person's to keep (ADR 0050).

Before a model read a person's messages (ADR 0047), the word patterns made a lasting rule of
any sentence with "always" or "never" and kept every turn as said, whatever it was: a skill's
text pasted into a chat became the standing rules of the person who pasted it, a request
("show the top 20 as a chart") stayed a memory of theirs. Reading new messages with the model
changes nothing about those: they are stored, and they are still served.

This reads the stored messages again with the model, exactly as ingest now would (bound to
the message's author: their key pays, their tenant's policy decides), and where it says a
message is a request or shared material - not a statement - forgets what was made from it:
the turn as said, and the rules, instructions, preferences and attributes the patterns read
from it. Forgotten the way a person's own "forget this" is (``MemoryService.forget``): the
index entry goes, the profile is rebuilt, and what the forgotten statement had replaced holds
again - a rule of the person's own that a pasted "never ..." superseded is back.

What the model itself reads from the same words stays. "Show the top 20. From now on always
answer as a table." is a request, and its second sentence is the person's rule: the model's
reading says so, quoting it, and the rule the patterns made of that sentence is kept (a rule
or a preference by a reading that is one; a fact about the person by any reading of its
words). The message is read as it was the first time: what was made from it is not among
the memories the model is shown, or it would add nothing - a memory holds it - and the rule
would go with the request.

    uv run python -m memory_service.tools.clean_up --tenant acme [--user u1] \\
        [--before 2026-11-27] [--limit 500] [--apply]

Without ``--apply`` nothing changes: the report says what would be forgotten. It only
removes: nothing is added or reworded, and a message the model says is a statement keeps
everything made from it. A message the model may not read, or does not answer for, is left
as it is and counted (``unread``); run it again for those. One model call per message.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final

from sqlalchemy import select

from memory_service.__about__ import __version__
from memory_service.adapters.db.orm import MemoryRow, ObservationRow
from memory_service.application.container import Container, build_container
from memory_service.config.settings import Settings
from memory_service.domain.enums import ObservationKind
from memory_service.domain.memory import VERBATIM_TURN
from memory_service.modules.memory.pipeline import Gaps, context_from_observation
from memory_service.observability.logging import get_logger

log = get_logger(__name__)

#: What the patterns made of a message that is the person's own only when the message is
#: their statement: the turn as said, and what was read of them from it.
MADE: Final = (VERBATIM_TURN, "rule", "instruction", "preference", "attribute")
#: A standing statement of the person (how they want things done) stays only when the
#: model's reading of the same words is one too: a fact read from a skill's "never ..."
#: does not make that sentence the person's rule.
STANDING: Final = ("rule", "instruction", "preference")
OWN: Final = ("rule", "preference")
#: Two texts shorter than this are the same words only when equal: a word or two is in many
#: sentences without being read from any of them (as ``context.check.PART_CHARS``).
PART_CHARS: Final = 16
#: the model's readings are its own: never re-read, never forgotten here
MODEL: Final = "llm"
#: how many examples of each outcome a report shows
EXAMPLES: Final = 20


@dataclass
class CleanUpReport:
    """What a run read and what it forgot (or, without ``apply``, would forget)."""

    applied: bool = False
    #: messages the model read again
    read: int = 0
    #: of those, statements: everything made from them stays
    kept: int = 0
    #: messages the model may not read or did not answer for: left as they are
    unread: int = 0
    #: memories forgotten (or that would be), and the messages they were made from
    forgotten: int = 0
    messages: int = 0
    #: readings of those messages that stay: the model reads the same from the same words
    confirmed: int = 0
    examples: list[dict[str, str]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "read": self.read,
            "kept": self.kept,
            "unread": self.unread,
            "messages": self.messages,
            "forgotten": self.forgotten,
            "confirmed": self.confirmed,
            "examples": self.examples,
            "failures": self.failures,
        }


async def clean_up(
    container: Container,
    *,
    tenant_id: str,
    user_id: str | None = None,
    before: datetime | None = None,
    limit: int | None = None,
    apply: bool = False,
) -> CleanUpReport:
    """Read ``tenant_id``'s stored messages again (one user's with ``user_id``, those said
    before ``before``, at most ``limit``) and forget what was made from the ones that are no
    statement; with ``apply`` false, only report."""
    report = CleanUpReport(applied=apply)
    made = await _made(container, tenant_id, user_id)
    observations = await _observations(container, tenant_id, sorted(made), before, limit)
    log.info("clean_up.tenant", tenant_id=tenant_id, messages=len(observations), apply=apply)
    for observation_id, message_id in observations:
        try:
            await _one(container, tenant_id, observation_id, made[message_id], report)
        except Exception as exc:
            report.failures.append(f"message {tenant_id}/{message_id}: {exc}")
    log.info("clean_up.done", tenant_id=tenant_id, **{**report.as_dict(), "examples": None})
    return report


async def _made(
    container: Container, tenant_id: str, user_id: str | None
) -> dict[str, list[tuple[str, str, str]]]:
    """By message: the current memories the patterns made from it (id, category, text)."""
    query = select(
        MemoryRow.memory_id, MemoryRow.system_metadata, MemoryRow.evidence, MemoryRow.content
    ).where(
        MemoryRow.tenant_id == tenant_id,
        MemoryRow.temporal_status == "CURRENT",
        MemoryRow.deleted_at.is_(None),
        MemoryRow.system_metadata["category"].astext.in_(MADE),
    )
    if user_id is not None:
        query = query.where(MemoryRow.owner_principal == f"user:{user_id}")
    out: dict[str, list[tuple[str, str, str]]] = {}
    async with container.database.session_factory() as session:
        for memory_id, metadata, evidence, content in await session.execute(query):
            if (metadata or {}).get("provider") == MODEL:
                continue
            for ref in evidence or []:
                if isinstance(ref, dict) and ref.get("message_id"):
                    out.setdefault(str(ref["message_id"]), []).append(
                        (memory_id, str(metadata.get("category")), content or "")
                    )
    return out


async def _observations(
    container: Container,
    tenant_id: str,
    message_ids: list[str],
    before: datetime | None,
    limit: int | None,
) -> list[tuple[str, str]]:
    """The observations of ``message_ids`` (id, message), in the order they were said."""
    if not message_ids:
        return []
    query = (
        select(ObservationRow.observation_id, ObservationRow.message_id)
        .where(
            ObservationRow.tenant_id == tenant_id,
            ObservationRow.kind == ObservationKind.MESSAGE.value,
            ObservationRow.message_id.in_(message_ids),
        )
        .order_by(ObservationRow.created_at, ObservationRow.observation_id)
    )
    if before is not None:
        query = query.where(ObservationRow.occurred_at < before)
    if limit:
        query = query.limit(limit)
    async with container.database.session_factory() as session:
        return [(o, m) for o, m in await session.execute(query)]


async def _one(
    container: Container,
    tenant_id: str,
    observation_id: str,
    made: list[tuple[str, str, str]],
    report: CleanUpReport,
) -> None:
    """Read one message again and forget what was made from it when it is no statement."""
    async with container.services.uow_factory() as uow:
        observation = await uow.observations.get(tenant_id, observation_id)
    if observation is None:
        return
    gaps = await container.services.observation_pipeline.reading(observation)
    if gaps is None or gaps.unread:
        report.unread += 1
        return
    report.read += 1
    if gaps.said:  # a statement, or one by its shape not asked about: all of it stays
        report.kept += 1
        return
    report.messages += 1
    # an operator's maintenance, as the tenant: what a person said is otherwise theirs alone
    ctx = context_from_observation(observation).model_copy(update={"administers_tenant": True})
    for memory_id, category, content in made:
        if _confirmed(category, content, gaps):
            report.confirmed += 1
            continue
        if len(report.examples) < EXAMPLES:
            report.examples.append({"category": category, "text": " ".join(content.split())[:160]})
        if report.applied:
            async with container.services.uow_factory() as uow:
                await container.services.memory.forget(uow, ctx, memory_id)
                await uow.commit()
        report.forgotten += 1


def _confirmed(category: str, content: str, gaps: Gaps) -> bool:
    """Whether the model's reading of the message says what this memory of it says: a
    reading that quotes the memory's words (``STANDING``: one that is itself the person's
    rule or preference). The turn as said is never confirmed: the message is no statement."""
    if category == VERBATIM_TURN:
        return False
    for (read, _), quote in zip(gaps.fills, gaps.quotes, strict=True):
        if category in STANDING and read.category not in OWN:
            continue
        if _same_words(quote, content):
            return True
    return False


def _same_words(quote: str, content: str) -> bool:
    """Whether one text is, or holds, the other (case, spacing and end marks aside)."""
    a, b = _plain(quote), _plain(content)
    if not a or not b:
        return False
    return a == b or (min(len(a), len(b)) >= PART_CHARS and (a in b or b in a))


def _plain(text: str) -> str:
    return " ".join(text.split()).casefold().strip(" .!?;:,\"'")


async def _main(args: argparse.Namespace) -> int:
    container = await build_container(Settings(), __version__)
    try:
        report = await clean_up(
            container,
            tenant_id=args.tenant,
            user_id=args.user,
            before=args.before,
            limit=args.limit,
            apply=args.apply,
        )
    finally:
        await container.close()
    sys.stdout.write(f"{report.as_dict()}\n")
    return 0 if not report.failures else 1


def _day(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Forget what was made from stored messages a model says are no statement"
    )
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--user", default=None, help="only this user's messages")
    parser.add_argument(
        "--before", type=_day, default=None, help="only messages said before this day (YYYY-MM-DD)"
    )
    parser.add_argument("--limit", type=int, default=None, help="at most this many messages")
    parser.add_argument("--apply", action="store_true", help="forget; without it, only report")
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
