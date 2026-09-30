"""Durable thread summaries: the ``summary.refresh`` job.

A thread's summary rolls forward: every ``SUMMARY_EVERY`` messages a job folds the messages
after the last summary into it (new = f(previous summary, new messages)) and stores a new
version covering them. With the tenant's model (use ``summaries``, paid by the principal whose
message triggered it) the fold is abstractive; without one the stored summary is the
extractive digest (a line per message, role and first sentence), so every thread has one.

The pushed context carries the latest summary and only the messages after it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Final

from memory_service.domain.conversation import Message
from memory_service.domain.enums import MessageKind
from memory_service.domain.profile import ThreadSummary
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.llm.assist import LLMAssist
from memory_service.observability.logging import get_logger
from memory_service.ports.credentials import ModelIdentity
from memory_service.ports.tasks import JobSpec, Queue
from memory_service.ports.uow import UnitOfWork, UnitOfWorkFactory

log = get_logger(__name__)

TASK_SUMMARY_REFRESH: Final = "summary.refresh"
#: A thread's summary is refreshed every this many messages.
SUMMARY_EVERY: Final = 20
#: The most messages one refresh folds in (a thread that ran far ahead of its summary
#: catches up over several refreshes).
SUMMARY_SOURCE_MESSAGES: Final = 200
#: The summary's length bound, and one line of the extractive digest.
SUMMARY_MAX_CHARS: Final = 2000
LINE_CHARS: Final = 160
EXTRACTIVE: Final = "extractive"

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}
_SYSTEM = (
    "You keep the running summary of a conversation. Fold the new messages into the previous "
    "summary: what was asked, the facts and decisions stated, and what is still open. Use "
    "only what is given. Write in the language of the conversation. At most {max_chars} "
    'characters. Return JSON only: {{"summary": "..."}}.'
)


def digest(messages: Sequence[Message]) -> list[str]:
    """A line per visible message: role and first sentence."""
    lines = []
    for m in messages:
        if m.kind is not MessageKind.VISIBLE or not m.content.strip():
            continue
        first = re.split(r"(?<=[.!?])\s+", m.content.strip(), maxsplit=1)[0][:LINE_CHARS]
        lines.append(f"{m.role.value.lower()}: {first}")
    return lines


def rolled(previous: str, messages: Sequence[Message]) -> str:
    """The extractive fold: the previous summary and the new lines, the oldest lines dropped
    to stay within the bound."""
    lines = [*previous.splitlines(), *digest(messages)]
    while lines and sum(len(line) + 1 for line in lines) > SUMMARY_MAX_CHARS:
        lines.pop(0)
    return "\n".join(lines)


async def enqueue_refresh(uow: UnitOfWork, tenant_id: str, thread_id: str, **owner: Any) -> None:
    await uow.enqueue(
        JobSpec(
            task_name=TASK_SUMMARY_REFRESH,
            queue=Queue.SUMMARY,
            payload={"tenant_id": tenant_id, "thread_id": thread_id, **owner},
            tenant_id=tenant_id,
        )
    )


class ThreadSummaries:
    def __init__(self, uow_factory: UnitOfWorkFactory, assist: LLMAssist) -> None:
        self.uow_factory = uow_factory
        self.assist = assist

    async def refresh(
        self,
        tenant_id: str,
        thread_id: str,
        *,
        principal_id: str | None = None,
        workspace_id: str | None = None,
    ) -> ThreadSummary | None:
        async with self.uow_factory() as uow:
            previous = await uow.summaries.latest(tenant_id, thread_id)
            covered = previous.covers_to_sequence if previous else 0
            new = await uow.messages.list_after(
                tenant_id, thread_id, after_sequence=covered, limit=SUMMARY_SOURCE_MESSAGES
            )
        if not new:
            return previous
        before = previous.text if previous else ""
        text, model = await self._fold(before, new, tenant_id, principal_id, workspace_id)
        summary = ThreadSummary(
            tenant_id=tenant_id,
            thread_id=thread_id,
            version=(previous.version if previous else 0) + 1,
            text=text,
            covers_to_sequence=new[-1].sequence,
            model=model,
        )
        async with self.uow_factory() as uow:
            if await uow.summaries.add(summary):
                await uow.revisions.bump(tenant_id, RevisionKind.THREAD, thread_id)
                await uow.commit()
        log.info("summary.refreshed", tenant_id=tenant_id, version=summary.version, model=model)
        return summary

    async def _fold(
        self,
        previous: str,
        messages: Sequence[Message],
        tenant_id: str,
        principal_id: str | None,
        workspace_id: str | None,
    ) -> tuple[str, str]:
        if principal_id:
            async with self.assist.bound(ModelIdentity(tenant_id, principal_id, workspace_id)):
                if self.assist.wants("summaries"):
                    written = await self._abstractive(previous, messages)
                    if written is not None:
                        return written, self.assist.settings.model or "llm"
        return rolled(previous, messages), EXTRACTIVE

    async def _abstractive(self, previous: str, messages: Sequence[Message]) -> str | None:
        lines = "\n".join(
            f"{m.role.value.lower()}: {' '.join(m.content.split())}"
            for m in messages
            if m.kind is MessageKind.VISIBLE and m.content.strip()
        )
        result = await self.assist.structured(
            "summaries",
            system=_SYSTEM.format(max_chars=SUMMARY_MAX_CHARS),
            user=f"Previous summary:\n{previous or '(none)'}\n\nNew messages:\n{lines[-8000:]}",
            schema=SUMMARY_SCHEMA,
            max_tokens=900,
        )
        text = str((result or {}).get("summary", "")).strip()
        return text if text and len(text) <= int(SUMMARY_MAX_CHARS * 1.5) else None
