"""A conversation's messages form memories one at a time, in the order they were stored.

"I work at ACME Corp." and "I work at Globex now." sent in one ``history.add`` batch both
stayed current, although where someone works holds one value: the two messages' jobs ran side by
side, and each compared its statement with the store before the other had committed. Sent in two
calls, the second saw the first and replaced it. The jobs of one conversation now share the
queue's ``lock`` (jobs sharing one run one after another, in the order deferred); other
conversations are not held up by it.
"""

from __future__ import annotations

from itertools import count
from types import SimpleNamespace
from typing import Any

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import MessageRole
from memory_service.modules.conversation.service import (
    TASK_PROCESS_OBSERVATION,
    ConversationService,
)
from memory_service.ports.tasks import JobSpec

pytestmark = pytest.mark.unit

BATCH = MemoryExecutionContext(tenant_id="acme", user_id="u1", thread_id="thr_batch")
OTHER = BATCH.model_copy(update={"thread_id": "thr_other"})


class _Table:
    """A repository whose every call answers ``None`` unless named in ``answers``."""

    def __init__(self, **answers: Any) -> None:
        self.answers = answers

    def __getattr__(self, name: str) -> Any:
        async def call(*_: Any, **__: Any) -> Any:
            answer = self.answers.get(name)
            return answer() if callable(answer) else answer

        return call


class _Uow:
    """The unit of work a message is appended in: an existing thread, and the jobs queued."""

    def __init__(self) -> None:
        sequence = count(1)
        self.threads = _Table(get=SimpleNamespace, touch=1)
        self.turns = _Table(next_sequence=lambda: next(sequence))
        self.messages = _Table(next_sequence=lambda: next(sequence))
        self.sessions = self.observations = self.revisions = _Table()
        self.jobs: list[JobSpec] = []

    async def serialize(self, *_: str) -> None:
        return None

    async def enqueue(self, spec: JobSpec) -> int:
        self.jobs.append(spec)
        return len(self.jobs)


async def test_a_batchs_messages_are_read_in_order_and_other_conversations_beside_them() -> None:
    conversation = ConversationService(None, None, archive_enabled=False)  # type: ignore[arg-type]
    uow = _Uow()
    said = [
        (BATCH, "I work at ACME Corp."),
        (BATCH, "I work at Globex now."),
        (OTHER, "I live in Leeds."),
    ]
    acks = [
        await conversation.append_message(
            uow,  # type: ignore[arg-type]
            ctx,
            role=MessageRole.USER,
            content=text,
            may_write=True,
        )
        for ctx, text in said
    ]
    jobs = [j for j in uow.jobs if j.task_name == TASK_PROCESS_OBSERVATION]
    assert [j.payload["observation_id"] for j in jobs] == [a.ack.observation_id for a in acks]
    acme, globex, leeds = jobs
    assert acme.lock and acme.lock == globex.lock, "one conversation: one after another"
    assert leeds.lock and leeds.lock != acme.lock, "another conversation is not held up"
    # each message is still its own job, deduplicated on its own
    assert len({j.idempotency_key for j in jobs}) == 3
