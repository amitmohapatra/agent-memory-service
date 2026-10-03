"""Feedback on a memory changes the memory through the revision machinery, exactly once; a
vote changes nothing until a tenant admin approves it (ADR 0028)."""

from __future__ import annotations

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import TemporalStatus
from memory_service.domain.errors import Conflict, ScopeDenied, ValidationFailed
from memory_service.domain.feedback import (
    Feedback,
    FeedbackSource,
    FeedbackTargetKind,
    FeedbackVerdict,
    ProjectionAction,
    ReviewState,
)
from memory_service.domain.revisions import RevisionKind
from tests.integration.test_memory import _memories, _observe

pytestmark = pytest.mark.integration
ALICE = MemoryExecutionContext(tenant_id="acme", user_id="alice", workspace_id="fin")
BOB = MemoryExecutionContext(tenant_id="acme", user_id="bob", workspace_id="ops")
#: a tenant admin, whose word needs no review
ROOT = MemoryExecutionContext(tenant_id="acme", user_id="root")
#: who reviews through the API: the tenant's administrator credential
REVIEWER = "key:acme-admin"


async def _admin(container, uow_factory) -> None:
    async with uow_factory() as uow:
        await container.services["authz"].grant_membership(
            "acme", "root", admin=True, revisions=uow.revisions
        )
        await uow.commit()


async def _first_memory(container, uow_factory):
    await _observe(
        container,
        uow_factory,
        ALICE,
        "My name is Amit and my timezone is Europe/Berlin. I prefer concise answers with code.",
    )
    memories = await _memories(uow_factory, ALICE, container)
    assert memories, "the observation should have produced memories"
    return memories[0]


async def _submit(container, uow_factory, ctx, *, approve: bool = True, **fields) -> Feedback:
    """Submit; a vote that waits for review is approved by a tenant admin first (``approve``),
    so it lands as it did before review existed."""
    service = container.services["feedback"]
    record = Feedback(tenant_id=ctx.tenant_id, **fields)
    async with uow_factory() as uow:
        stored, created = await service.submit(uow, ctx, record)
        await uow.commit()
    assert created
    if stored.pending and approve:
        async with uow_factory() as uow:
            await service.review(
                uow, "acme", stored.feedback_id, approve=True, reviewed_by=REVIEWER
            )
            await uow.commit()
    await container.tasks.drain()  # feedback.project (+ memory.index it enqueues)
    await container.tasks.drain()
    async with uow_factory() as uow:
        return await service.get(uow, ctx, stored.feedback_id)


async def _revisions(uow_factory, tenant_id: str) -> dict[str, int]:
    async with uow_factory() as uow:
        return await uow.revisions.get_many(
            tenant_id, [(RevisionKind.USER, "alice"), (RevisionKind.TENANT, "")]
        )


@pytest.mark.parametrize("affirming", [FeedbackVerdict.CONFIRM, FeedbackVerdict.APPROVE])
async def test_confirm_reinforces_reject_retracts_and_the_cache_is_invalidated(
    container, uow_factory, affirming: FeedbackVerdict
) -> None:
    memory = await _first_memory(container, uow_factory)
    before = await _revisions(uow_factory, "acme")
    confirmed = await _submit(
        container,
        uow_factory,
        ALICE,
        target_kind=FeedbackTargetKind.MEMORY,
        target_id=memory.memory_id,
        verdict=affirming,
        score=0.9,
    )
    assert confirmed.projection is not None
    assert confirmed.projection.action is ProjectionAction.MEMORY_REINFORCED
    assert confirmed.workspace_id == "fin" and confirmed.user_id == "alice"
    async with uow_factory() as uow:
        reinforced = await uow.memories.get("acme", memory.memory_id)
    assert reinforced is not None
    assert reinforced.reinforcement_count == memory.reinforcement_count + 1
    assert reinforced.confidence == pytest.approx(min(1.0, memory.confidence + 0.1))
    assert await _revisions(uow_factory, "acme") != before

    # the projector is idempotent: running the job again changes nothing
    await container.services["feedback"].project("acme", confirmed.feedback_id)
    async with uow_factory() as uow:
        again = await uow.memories.get("acme", memory.memory_id)
    assert again is not None and again.reinforcement_count == reinforced.reinforcement_count

    rejected = await _submit(
        container,
        uow_factory,
        ALICE,
        target_kind=FeedbackTargetKind.MEMORY,
        target_id=memory.memory_id,
        verdict=FeedbackVerdict.REJECT,
    )
    assert rejected.projection is not None
    assert rejected.projection.action is ProjectionAction.MEMORY_RETRACTED
    async with uow_factory() as uow:
        retracted = await uow.memories.get("acme", memory.memory_id)
    assert retracted is not None and retracted.temporal.status is TemporalStatus.RETRACTED
    assert memory.memory_id not in {
        m.memory_id for m in await _memories(uow_factory, ALICE, container)
    }

    # a verdict on a memory that is no longer current is recorded, not projected
    late = await _submit(
        container,
        uow_factory,
        ALICE,
        target_kind=FeedbackTargetKind.MEMORY,
        target_id=memory.memory_id,
        verdict=FeedbackVerdict.CONFIRM,
    )
    assert late.projection is not None and late.projection.action is ProjectionAction.NONE
    assert "retracted" in (late.projection.reason or "")


@pytest.mark.parametrize("verdict", [FeedbackVerdict.CORRECT, FeedbackVerdict.EDIT])
async def test_a_correction_supersedes_the_memory_with_the_corrected_text(
    container, uow_factory, verdict: FeedbackVerdict
) -> None:
    memory = await _first_memory(container, uow_factory)
    corrected = await _submit(
        container,
        uow_factory,
        ALICE,
        target_kind=FeedbackTargetKind.MEMORY,
        target_id=memory.memory_id,
        verdict=verdict,
        correction={"content": "Amit's timezone is Europe/Lisbon, not Europe/Berlin."},
        reviewer="alice",
    )
    projection = corrected.projection
    assert projection is not None and projection.action is ProjectionAction.MEMORY_SUPERSEDED
    assert projection.superseded_by and projection.superseded_by != memory.memory_id
    async with uow_factory() as uow:
        old = await uow.memories.get("acme", memory.memory_id)
        new = await uow.memories.get("acme", projection.superseded_by)
    assert old is not None and old.temporal.status is TemporalStatus.SUPERSEDED
    assert old.temporal.superseded_by == new.memory_id if new else False
    assert new is not None and new.temporal.supersedes == memory.memory_id
    assert new.content == "Amit's timezone is Europe/Lisbon, not Europe/Berlin."
    assert new.scope == memory.scope and new.visibility == memory.visibility
    assert new.owner_principal == memory.owner_principal and new.confidence >= 0.9
    assert any(
        e.source_type == "feedback" and e.source_id == corrected.feedback_id for e in new.evidence
    )
    assert new.system_metadata["corrected_by"] == corrected.feedback_id
    # a fresh row: nothing of the old row's lifecycle state comes along (the row folds its
    # nullable columns into system_metadata on the way out, so absent means None here)
    for inherited in ("expires_at", "derived_slot", "admission", "source_revisions"):
        assert new.system_metadata.get(inherited) is None
    listed = {m.memory_id for m in await _memories(uow_factory, ALICE, container)}
    assert new.memory_id in listed and memory.memory_id not in listed
    # the chain is readable both ways through the API service too
    memory_service = container.services["memory"]
    async with uow_factory() as uow:
        assert (
            await memory_service.get_memory(uow, ALICE, new.memory_id)
        ).temporal.supersedes == memory.memory_id


async def test_feedback_visibility_follows_the_target_and_the_author(
    container, uow_factory
) -> None:
    memory = await _first_memory(container, uow_factory)
    service = container.services["feedback"]
    record = await _submit(
        container,
        uow_factory,
        ALICE,
        target_kind=FeedbackTargetKind.MEMORY,
        target_id=memory.memory_id,
        verdict=FeedbackVerdict.CONFIRM,
    )
    async with uow_factory() as uow:
        mine = await service.list_for(
            uow, ALICE, target_kind=FeedbackTargetKind.MEMORY, target_id=memory.memory_id
        )
        assert [f.feedback_id for f in mine] == [record.feedback_id]
        with pytest.raises(ScopeDenied):  # bob cannot read alice's memory, so not its feedback
            await service.list_for(
                uow, BOB, target_kind=FeedbackTargetKind.MEMORY, target_id=memory.memory_id
            )
        with pytest.raises(ScopeDenied):
            await service.submit(
                uow,
                BOB,
                Feedback(
                    tenant_id="acme",
                    target_kind=FeedbackTargetKind.MEMORY,
                    target_id=memory.memory_id,
                    verdict=FeedbackVerdict.REJECT,
                ),
            )
        with pytest.raises(ValidationFailed):  # a record claiming another tenant
            await service.submit(
                uow,
                ALICE,
                Feedback(
                    tenant_id="globex",
                    target_kind=FeedbackTargetKind.RUN,
                    target_id="run_1",
                    verdict=FeedbackVerdict.CONFIRM,
                ),
            )


async def test_listing_pages_newest_first_through_the_keyset(container, uow_factory) -> None:
    service = container.services["feedback"]
    ids = []
    for n in range(5):
        record = await _submit(
            container,
            uow_factory,
            ALICE,
            feedback_id=f"fb_page_{n}",
            target_kind=FeedbackTargetKind.RUN,
            target_id="run_paged",
            verdict=FeedbackVerdict.CONFIRM,
        )
        ids.append(record.feedback_id)
    seen: list[str] = []
    before = None
    async with uow_factory() as uow:
        while True:
            rows = await service.list_for(
                uow,
                ALICE,
                target_kind=FeedbackTargetKind.RUN,
                target_id="run_paged",
                before=before,
                limit=2,
            )
            seen.extend(r.feedback_id for r in rows)
            if len(rows) < 2:
                break
            before = (rows[-1].created_at, rows[-1].feedback_id)
    assert seen == list(reversed(ids))


async def test_a_run_verdict_moves_the_cited_memories_and_labels_the_run(
    container, uow_factory
) -> None:
    """Confirmed: every memory the run's answer cited gains confidence and a reinforcement,
    and the run is labelled successful. Rejected: they lose confidence. A person's label
    outranks the judge's (``OUTCOME_PRECEDENCE``)."""
    memory = await _first_memory(container, uow_factory)
    agent = ALICE.model_copy(update={"agent_id": "helper", "agent_run_id": "run_answer_1"})
    cited = [{"source_type": "memory", "source_id": memory.memory_id}]
    confirmed = await _submit(
        container,
        uow_factory,
        agent,
        target_kind=FeedbackTargetKind.RUN,
        target_id="run_answer_1",
        verdict=FeedbackVerdict.CONFIRM,
        source=FeedbackSource.JUDGE,
        evidence_refs=cited,
    )
    assert confirmed.projection is not None
    assert confirmed.projection.action is ProjectionAction.RUN_LABELLED
    assert confirmed.projection.memory_ids == [memory.memory_id]
    assert confirmed.projection.run_id == "run_answer_1"
    async with uow_factory() as uow:
        raised = await uow.memories.get("acme", memory.memory_id)
        outcome = await uow.tools.outcome("acme", "run_answer_1")
    assert raised is not None and raised.confidence == pytest.approx(
        min(1.0, memory.confidence + 0.05)
    )
    assert raised.reinforcement_count == memory.reinforcement_count + 1
    assert outcome is not None and outcome.success and outcome.source == "judge"

    await _submit(
        container,
        uow_factory,
        agent,
        target_kind=FeedbackTargetKind.RUN,
        target_id="run_answer_1",
        verdict=FeedbackVerdict.APPROVE,
        source=FeedbackSource.HUMAN,
    )
    rejected = await _submit(
        container,
        uow_factory,
        agent,
        target_kind=FeedbackTargetKind.RUN,
        target_id="run_answer_1",
        verdict=FeedbackVerdict.REJECT,
        source=FeedbackSource.JUDGE,
        evidence_refs=cited,
    )
    assert rejected.projection is not None
    assert rejected.projection.action is ProjectionAction.NONE
    assert rejected.projection.memory_ids == [memory.memory_id]
    async with uow_factory() as uow:
        lowered = await uow.memories.get("acme", memory.memory_id)
        kept = await uow.tools.outcome("acme", "run_answer_1")
    assert lowered is not None and lowered.confidence == pytest.approx(raised.confidence - 0.05)
    assert kept is not None and kept.success and kept.source == "human"


async def test_a_run_verdict_may_only_cite_memories_its_reviewer_can_read(
    container, uow_factory
) -> None:
    memory = await _first_memory(container, uow_factory)
    service = container.services["feedback"]
    record = Feedback(
        tenant_id="acme",
        target_kind=FeedbackTargetKind.RUN,
        target_id="run_x",
        verdict=FeedbackVerdict.REJECT,
        evidence_refs=[{"source_type": "memory", "source_id": memory.memory_id}],
    )
    with pytest.raises(ScopeDenied):
        async with uow_factory() as uow:
            await service.submit(uow, BOB, record)


async def test_a_verdict_on_a_run_is_its_outcome(container, uow_factory) -> None:
    corrected = await _submit(
        container,
        uow_factory,
        ALICE,
        target_kind=FeedbackTargetKind.RUN,
        target_id="run_judged",
        verdict=FeedbackVerdict.CORRECT,
        correction="the PO should have gone to Globex",
    )
    assert corrected.projection is not None
    assert corrected.projection.action is ProjectionAction.RUN_LABELLED
    async with uow_factory() as uow:
        outcome = await uow.tools.outcome("acme", "run_judged")
    assert outcome is not None and outcome.success is False and outcome.source == "human"


async def test_a_vote_waits_for_review_and_changes_nothing_until_approved(
    container, uow_factory
) -> None:
    memory = await _first_memory(container, uow_factory)
    service = container.services["feedback"]
    vote = await _submit(
        container,
        uow_factory,
        ALICE,
        approve=False,
        target_kind=FeedbackTargetKind.MEMORY,
        target_id=memory.memory_id,
        verdict=FeedbackVerdict.CONFIRM,
    )
    assert vote.review is not None and vote.review.state is ReviewState.PENDING
    assert vote.projection is None
    async with uow_factory() as uow:
        untouched = await uow.memories.get("acme", memory.memory_id)
    assert untouched is not None and untouched.confidence == memory.confidence

    async with uow_factory() as uow:
        [(queued, record)] = await service.pending(uow, "acme")
    assert queued.feedback_id == vote.feedback_id
    assert record == {"pending": 1, "approved": 0, "dismissed": 0}

    async with uow_factory() as uow:
        approved = await service.review(
            uow,
            "acme",
            vote.feedback_id,
            approve=True,
            reviewed_by=REVIEWER,
            note="checked against the source",
        )
        await uow.commit()
    assert approved.review is not None and approved.review.reviewed_by == REVIEWER
    await container.tasks.drain()
    await container.tasks.drain()
    async with uow_factory() as uow:
        reinforced = await uow.memories.get("acme", memory.memory_id)
        applied = await service.get(uow, ALICE, vote.feedback_id)
        assert await service.pending(uow, "acme") == []
    assert reinforced is not None and reinforced.confidence > memory.confidence
    assert applied.projection is not None
    assert applied.projection.action is ProjectionAction.MEMORY_REINFORCED
    # reviewed once: a second decision is a conflict, not a second application
    with pytest.raises(Conflict):
        async with uow_factory() as uow:
            await service.review(uow, "acme", vote.feedback_id, approve=False, reviewed_by=REVIEWER)


async def test_a_dismissed_vote_is_kept_and_never_applied(container, uow_factory) -> None:
    memory = await _first_memory(container, uow_factory)
    service = container.services["feedback"]
    agent = ALICE.model_copy(update={"agent_id": "helper", "agent_run_id": "run_voted"})
    vote = await _submit(
        container,
        uow_factory,
        agent,
        approve=False,
        target_kind=FeedbackTargetKind.RUN,
        target_id="run_voted",
        verdict=FeedbackVerdict.REJECT,
        evidence_refs=[{"source_type": "memory", "source_id": memory.memory_id}],
    )
    assert vote.pending
    async with uow_factory() as uow:
        await service.review(
            uow,
            "acme",
            vote.feedback_id,
            approve=False,
            reviewed_by=REVIEWER,
            note="answer was right",
        )
        await uow.commit()
    await container.tasks.drain()
    async with uow_factory() as uow:
        kept = await service.get(uow, agent, vote.feedback_id)
        unchanged = await uow.memories.get("acme", memory.memory_id)
        outcome = await uow.tools.outcome("acme", "run_voted")
        # the author's record now shows the dismissal to whoever reviews their next vote
        counts = await uow.feedback.review_counts("acme", user_id="alice", agent_id="helper")
    assert kept.review is not None and kept.review.state is ReviewState.DISMISSED
    assert kept.review.note == "answer was right"
    assert kept.projection is not None and kept.projection.action is ProjectionAction.NONE
    assert unchanged is not None and unchanged.confidence == memory.confidence
    assert outcome is None
    assert counts["dismissed"] == 1


async def test_what_is_applied_as_it_arrives(container, uow_factory) -> None:
    """The judge's own verdict, a run reporting its own status, an owner's edit of a memory
    and what a tenant admin says need no review."""
    memory = await _first_memory(container, uow_factory)
    service = container.services["feedback"]
    run = ALICE.model_copy(update={"agent_id": "helper", "agent_run_id": "run_self"})
    await _admin(container, uow_factory)
    cases = [
        (
            run,
            {
                "target_kind": "run",
                "target_id": "run_self",
                "verdict": "confirm",
                "source": "system",
            },
            {},
        ),
        (
            ALICE,
            {"target_kind": "run", "target_id": "run_other", "verdict": "confirm"},
            {"trusted": True},
        ),
        (ROOT, {"target_kind": "run", "target_id": "run_admin", "verdict": "reject"}, {}),
        (ALICE, {"target_kind": "memory", "target_id": memory.memory_id, "verdict": "reject"}, {}),
    ]
    for ctx, fields, kwargs in cases:
        async with uow_factory() as uow:
            stored, _ = await service.submit(
                uow, ctx, Feedback(tenant_id="acme", **fields), **kwargs
            )
            await uow.commit()
        assert stored.review is None, fields
    # a run's status reported by anyone but that run waits like any other vote
    async with uow_factory() as uow:
        stored, _ = await service.submit(
            uow,
            run,
            Feedback(
                tenant_id="acme",
                target_kind=FeedbackTargetKind.RUN,
                target_id="run_someone_else",
                verdict=FeedbackVerdict.REJECT,
                source=FeedbackSource.SYSTEM,
            ),
        )
    assert stored.pending
