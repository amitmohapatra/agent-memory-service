"""Who may say what about a memory: reading grants affirmation, rewriting needs the owner."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.errors import ScopeDenied, ValidationFailed
from memory_service.domain.feedback import (
    Feedback,
    FeedbackSource,
    FeedbackTargetKind,
    FeedbackVerdict,
)
from memory_service.modules.feedback.service import FeedbackService

OWNER = MemoryExecutionContext(tenant_id="acme", user_id="alice", workspace_id="fin")
READER = MemoryExecutionContext(tenant_id="acme", user_id="carol", workspace_id="fin")
AGENT_OF_OWNER = MemoryExecutionContext(
    tenant_id="acme", user_id="alice", workspace_id="fin", agent_id="ref"
)


class _Memory:
    owner_principal = "user:alice"


class _MemoryService:
    async def get_memory(self, uow, ctx, memory_id):
        return _Memory()  # readable by everyone in this test


class _Authz:
    def __init__(self, admins: set[str]) -> None:
        self.admins = admins

    async def is_tenant_admin(self, ctx) -> bool:
        return ctx.user_id in self.admins


class _FeedbackRepo:
    def __init__(self) -> None:
        self.rows: dict[str, Feedback] = {}

    async def get(self, tenant_id, feedback_id):
        return self.rows.get(feedback_id)

    async def add(self, feedback) -> bool:
        self.rows[feedback.feedback_id] = feedback
        return True


class _Uow:
    def __init__(self) -> None:
        self.feedback = _FeedbackRepo()
        self.jobs: list = []

    async def enqueue(self, spec):
        self.jobs.append(spec)


def _service(admins: set[str] = frozenset()) -> FeedbackService:
    return FeedbackService(
        lambda: None,  # the projector is not exercised here
        _Authz(set(admins)),
        _MemoryService(),
        clock=lambda: datetime(2026, 9, 28, 7, tzinfo=UTC),
    )


def _record(verdict: FeedbackVerdict, **fields) -> Feedback:
    return Feedback(
        tenant_id="acme",
        target_kind=FeedbackTargetKind.MEMORY,
        target_id="mem_1",
        verdict=verdict,
        correction="x" if verdict in (FeedbackVerdict.CORRECT, FeedbackVerdict.EDIT) else None,
        **fields,
    )


@pytest.mark.parametrize("verdict", [FeedbackVerdict.CONFIRM, FeedbackVerdict.APPROVE])
async def test_a_reader_may_affirm(verdict: FeedbackVerdict) -> None:
    stored, created = await _service().submit(_Uow(), READER, _record(verdict))
    assert created and stored.user_id == "carol"


@pytest.mark.parametrize(
    "verdict", [FeedbackVerdict.REJECT, FeedbackVerdict.CORRECT, FeedbackVerdict.EDIT]
)
async def test_only_the_owner_or_an_admin_may_retract_or_rewrite(verdict: FeedbackVerdict) -> None:
    with pytest.raises(ScopeDenied, match="owner"):
        await _service().submit(_Uow(), READER, _record(verdict))
    assert (await _service().submit(_Uow(), OWNER, _record(verdict)))[1]
    assert (await _service().submit(_Uow(), AGENT_OF_OWNER, _record(verdict)))[1]
    assert (await _service({"carol"}).submit(_Uow(), READER, _record(verdict)))[1]


async def test_provenance_and_identity_come_from_the_request() -> None:
    uow = _Uow()
    ctx = OWNER.model_copy(update={"trace_id": "4bf92f3577b34da6a3ce929d0e0e4736"})
    body = _record(
        FeedbackVerdict.CONFIRM,
        trace_id="deadbeef" * 4,
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
    )
    stored, _ = await _service().submit(uow, ctx, body)
    assert stored.trace_id == ctx.trace_id and stored.created_at.year == 2026
    assert stored.workspace_id == "fin" and stored.user_id == "alice"
    # a reader's vote waits for review: stored, nothing scheduled (ADR 0028)
    assert stored.pending and uow.jobs == []
    with pytest.raises(ValidationFailed, match="user_id"):
        await _service().submit(_Uow(), OWNER, _record(FeedbackVerdict.CONFIRM, user_id="mallory"))


async def test_what_needs_review_and_what_is_applied_as_it_arrives() -> None:
    """A vote waits; an owner's edit, a run's own status, a tool-call decision, the judge and
    an admin in person do not. An agent acting for an admin is still an agent."""
    run_ctx = OWNER.model_copy(update={"agent_id": "ref", "agent_run_id": "run_1"})
    admin_agent = READER.model_copy(update={"agent_id": "ref"})
    cases = [
        (READER, _record(FeedbackVerdict.CONFIRM), {}, True),
        (OWNER, _record(FeedbackVerdict.REJECT), {}, False),
        (
            run_ctx,
            Feedback(
                tenant_id="acme",
                target_kind=FeedbackTargetKind.RUN,
                target_id="run_1",
                verdict=FeedbackVerdict.CONFIRM,
                source=FeedbackSource.SYSTEM,
            ),
            {},
            False,
        ),
        (
            run_ctx,
            Feedback(
                tenant_id="acme",
                target_kind=FeedbackTargetKind.RUN,
                target_id="run_2",
                verdict=FeedbackVerdict.REJECT,
                source=FeedbackSource.SYSTEM,
            ),
            {},
            True,
        ),
        (
            READER,
            Feedback(
                tenant_id="acme",
                target_kind=FeedbackTargetKind.TOOL_CALL,
                target_id="call_1",
                verdict=FeedbackVerdict.APPROVE,
            ),
            {},
            False,
        ),
        (
            run_ctx,
            Feedback(
                tenant_id="acme",
                target_kind=FeedbackTargetKind.RUN,
                target_id="run_1",
                verdict=FeedbackVerdict.REJECT,
                source=FeedbackSource.SYSTEM,
                evidence_refs=[{"source_type": "memory", "source_id": "mem_1"}],
            ),
            {},
            True,
        ),
        (READER, _record(FeedbackVerdict.CONFIRM), {"trusted": True}, False),
        (admin_agent, _record(FeedbackVerdict.CONFIRM), {}, True),
    ]
    for ctx, record, kwargs, waits in cases:
        uow = _Uow()
        stored, _ = await _service({"carol"} if ctx is admin_agent else frozenset()).submit(
            uow, ctx, record, **kwargs
        )
        assert stored.pending is waits, (ctx.principal_id, record.target_kind, record.source)
        assert bool(uow.jobs) is not waits
    stored, _ = await _service({"carol"}).submit(_Uow(), READER, _record(FeedbackVerdict.CONFIRM))
    assert not stored.pending, "a tenant admin in person"
