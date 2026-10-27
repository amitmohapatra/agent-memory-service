"""End users' thumbs wait for an admin, one vote per voter per target, and the admin works a
grouped queue in bulk (ADR 0039, amending 0028). Against the real database and projector.

A pending vote changes nothing: the memory's standing (which ranking reads), its index entry
and revisions, the run's outcome and the learning of skills from runs all stay as they were.
"""

from __future__ import annotations

import asyncio

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType, TemporalStatus, Visibility
from memory_service.domain.errors import Conflict, NotFound, ScopeDenied
from memory_service.domain.feedback import (
    REPLACED_REASON,
    Feedback,
    FeedbackSource,
    FeedbackTargetKind,
    FeedbackVerdict,
    ProjectionAction,
    ReviewState,
)
from memory_service.domain.ids import new_id
from memory_service.modules.jobs.names import TASK_MEMORY_INDEX
from memory_service.modules.tools.service import TASK_TOOLS_LEARN
from tests.integration.test_feedback_projection import (
    ALICE,
    BOB,
    REVIEWER,
    ROOT,
    _admin,
    _first_memory,
    _revisions,
)
from tests.integration.test_memory import _memories, _observe

pytestmark = pytest.mark.integration
#: Alice's agent, carrying her vote
HELPER = ALICE.model_copy(update={"agent_id": "helper", "agent_run_id": "run_helper"})
CAROL = MemoryExecutionContext(tenant_id="acme", user_id="carol", workspace_id="fin")


async def _vote(
    container, uow_factory, ctx, *, trusted: bool = False, run_vouched: bool = False, **fields
) -> Feedback:
    async with uow_factory() as uow:
        stored, _ = await container.services["feedback"].submit(
            uow,
            ctx,
            Feedback(tenant_id=ctx.tenant_id, **fields),
            trusted=trusted,
            run_vouched=run_vouched,
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()
    return stored


async def _get(container, uow_factory, feedback_id: str, tenant_id: str = "acme") -> Feedback:
    async with uow_factory() as uow:
        record = await uow.feedback.get(tenant_id, feedback_id)
    assert record is not None
    return record


def _job_count(container, task_name: str) -> int:
    return sum(1 for spec in container.tasks.payloads.values() if spec.task_name == task_name)


async def _bulk(container, uow_factory, *, approve: bool, **kw):
    async with uow_factory() as uow:
        result = await container.services["feedback"].review_many(
            uow, "acme", approve=approve, reviewed_by=REVIEWER, **kw
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()
    return result


async def test_a_thumbs_down_on_a_memory_waits_and_changes_nothing_until_approved(
    container, uow_factory
) -> None:
    memory = await _first_memory(container, uow_factory)
    before = await _revisions(uow_factory, "acme")
    index_jobs = _job_count(container, TASK_MEMORY_INDEX)
    down = await _vote(
        container,
        uow_factory,
        ALICE,  # the owner herself: a thumbs-down is a vote, not an edit
        target_kind=FeedbackTargetKind.MEMORY,
        target_id=memory.memory_id,
        verdict=FeedbackVerdict.REJECT,
    )
    assert down.review is not None and down.review.state is ReviewState.PENDING
    async with uow_factory() as uow:
        kept = await uow.memories.get("acme", memory.memory_id)
    assert kept is not None and kept.temporal.status is TemporalStatus.CURRENT
    assert (kept.confidence, kept.reinforcement_count) == (
        memory.confidence,
        memory.reinforcement_count,
    ), "standing, which ranking reads, is untouched"
    assert await _revisions(uow_factory, "acme") == before, "no cached bundle is invalidated"
    assert _job_count(container, TASK_MEMORY_INDEX) == index_jobs, "the index is not touched"
    assert (await _get(container, uow_factory, down.feedback_id)).projection is None

    await _bulk(container, uow_factory, approve=True, feedback_ids=[down.feedback_id])
    async with uow_factory() as uow:
        retracted = await uow.memories.get("acme", memory.memory_id)
    assert retracted is not None and retracted.temporal.status is TemporalStatus.RETRACTED
    applied = await _get(container, uow_factory, down.feedback_id)
    assert applied.projection is not None
    assert applied.projection.action is ProjectionAction.MEMORY_RETRACTED


async def test_an_owner_correction_with_text_applies_at_once(container, uow_factory) -> None:
    memory = await _first_memory(container, uow_factory)
    edit = await _vote(
        container,
        uow_factory,
        HELPER,  # an agent acting for the owner may correct too
        target_kind=FeedbackTargetKind.MEMORY,
        target_id=memory.memory_id,
        verdict=FeedbackVerdict.EDIT,
        correction={"content": "My timezone is Europe/Lisbon."},
    )
    assert edit.review is None
    applied = await _get(container, uow_factory, edit.feedback_id)
    assert applied.projection is not None
    assert applied.projection.action is ProjectionAction.MEMORY_SUPERSEDED


async def test_a_pending_run_vote_leaves_outcome_cited_memories_and_skills_alone(
    container, uow_factory
) -> None:
    memory = await _first_memory(container, uow_factory)
    learn_jobs = _job_count(container, TASK_TOOLS_LEARN)
    vote = await _vote(
        container,
        uow_factory,
        HELPER,  # an agent key carrying a person's thumbs: still a vote
        target_kind=FeedbackTargetKind.RUN,
        target_id="run_helper",
        verdict=FeedbackVerdict.CONFIRM,
        evidence_refs=[{"source_type": "memory", "source_id": memory.memory_id}],
    )
    assert vote.pending
    async with uow_factory() as uow:
        assert await uow.tools.outcome("acme", "run_helper") is None
        cited = await uow.memories.get("acme", memory.memory_id)
    assert cited is not None and cited.confidence == memory.confidence
    assert _job_count(container, TASK_TOOLS_LEARN) == learn_jobs, "no skill is learned from it"

    # dismissed: kept, and still nothing moves
    await _bulk(container, uow_factory, approve=False, feedback_ids=[vote.feedback_id])
    async with uow_factory() as uow:
        assert await uow.tools.outcome("acme", "run_helper") is None
    assert _job_count(container, TASK_TOOLS_LEARN) == learn_jobs
    # ... and it cannot be approved afterwards, one by one or in bulk
    with pytest.raises(Conflict):
        async with uow_factory() as uow:
            await container.services["feedback"].review(
                uow, "acme", vote.feedback_id, approve=True, reviewed_by=REVIEWER
            )
    decided, unchanged, _ = await _bulk(
        container, uow_factory, approve=True, feedback_ids=[vote.feedback_id]
    )
    assert decided == [] and [u.review.state for u in unchanged] == [ReviewState.DISMISSED]
    async with uow_factory() as uow:
        assert await uow.tools.outcome("acme", "run_helper") is None
        # kept for reporting
        [(listed, record)] = await container.services["feedback"].pending(
            uow, "acme", state=ReviewState.DISMISSED
        )
    assert listed.feedback_id == vote.feedback_id
    assert record == {"pending": 0, "approved": 0, "dismissed": 1}


async def test_what_is_not_a_vote_applies_at_once(container, uow_factory) -> None:
    """The run's own status, a tool-call decision and the admin (its credential, whatever
    source the record claims)."""
    await _admin(container, uow_factory)
    cases = [
        (HELPER, {"target_kind": "run", "target_id": "run_helper", "source": "system"}, False),
        (HELPER, {"target_kind": "tool_call", "target_id": "tiv_1", "source": "interrupt"}, False),
        (ALICE, {"target_kind": "run", "target_id": "run_judged", "source": "judge"}, True),
        (ROOT, {"target_kind": "run", "target_id": "run_root"}, False),
        # the tenant's administrator credential, on a memory
        (ALICE, {"target_kind": "memory", "target_id": "unused"}, True),
    ]
    memory = await _first_memory(container, uow_factory)
    for ctx, case, trusted in cases:
        fields = (
            {**case, "target_id": memory.memory_id} if case["target_kind"] == "memory" else case
        )
        stored = await _vote(
            container,
            uow_factory,
            ctx,
            trusted=trusted,
            # the harness's key may act as the agent whose run reports its status
            run_vouched=True,
            verdict=FeedbackVerdict.CONFIRM,
            **fields,
        )
        assert stored.review is None, fields
        assert (await _get(container, uow_factory, stored.feedback_id)).projection is not None


async def test_a_flipped_vote_replaces_the_earlier_and_is_applied_once(
    container, uow_factory
) -> None:
    memory = await _first_memory(container, uow_factory)
    common = {"target_kind": FeedbackTargetKind.MEMORY, "target_id": memory.memory_id}
    clicks = [
        await _vote(container, uow_factory, ALICE, verdict=FeedbackVerdict.CONFIRM, **common)
        for _ in range(3)
    ]
    down = await _vote(container, uow_factory, ALICE, verdict=FeedbackVerdict.REJECT, **common)
    up = await _vote(container, uow_factory, ALICE, verdict=FeedbackVerdict.CONFIRM, **common)
    for earlier in (*clicks, down):
        record = await _get(container, uow_factory, earlier.feedback_id)
        assert record.review is not None and record.review.state is ReviewState.DISMISSED
        assert record.review.replaced_by is not None and record.review.note == REPLACED_REASON
        assert record.projection is not None and record.projection.action is ProjectionAction.NONE
    async with uow_factory() as uow:
        queue = await container.services["feedback"].pending(uow, "acme")
        groups = await container.services["feedback"].pending_by_target(uow, "acme")
    assert [f.feedback_id for f, _ in queue] == [up.feedback_id], "five clicks, one vote"
    assert [(g.votes, g.thumbs_up, g.thumbs_down, g.voters) for g in groups] == [(1, 1, 0, 1)]

    # approving the whole target applies the one vote once; replaced votes stay unapplied
    decided, unchanged, more = await _bulk(
        container, uow_factory, approve=True, target=(FeedbackTargetKind.MEMORY, memory.memory_id)
    )
    assert [f.feedback_id for f in decided] == [up.feedback_id] and unchanged == [] and not more
    async with uow_factory() as uow:
        once = await uow.memories.get("acme", memory.memory_id)
    assert once is not None and once.reinforcement_count == memory.reinforcement_count + 1
    assert once.temporal.status is TemporalStatus.CURRENT, "the replaced thumbs-down never ran"
    # bulk approve again, naming replaced, approved and pending-less ids: nothing twice
    decided, unchanged, _ = await _bulk(
        container,
        uow_factory,
        approve=True,
        feedback_ids=[up.feedback_id, down.feedback_id, clicks[0].feedback_id],
    )
    assert decided == []
    assert [(u.review.state, u.review.replaced_by) for u in unchanged] == [
        (ReviewState.APPROVED, None),
        (ReviewState.DISMISSED, up.feedback_id),
        (ReviewState.DISMISSED, clicks[1].feedback_id),
    ]
    # the author's record counts what reviewers decided, not the author's changes of mind
    async with uow_factory() as uow:
        record = await uow.feedback.review_counts("acme", user_id="alice", agent_id=None)
    assert record == {"pending": 0, "approved": 1, "dismissed": 0}
    async with uow_factory() as uow:
        still = await uow.memories.get("acme", memory.memory_id)
    assert still is not None and still.reinforcement_count == once.reinforcement_count


async def test_the_grouped_queue_counts_votes_voters_and_the_latest_per_target(
    container, uow_factory
) -> None:
    for ctx, verdict in (
        (ALICE, FeedbackVerdict.CONFIRM),
        (HELPER, FeedbackVerdict.REJECT),  # Alice again, through her agent: replaces hers
        (BOB, FeedbackVerdict.REJECT),
        (CAROL, FeedbackVerdict.APPROVE),
    ):
        await _vote(
            container,
            uow_factory,
            ctx,
            target_kind=FeedbackTargetKind.RUN,
            target_id="run_popular",
            verdict=verdict,
        )
    last = await _vote(
        container,
        uow_factory,
        BOB,
        target_kind=FeedbackTargetKind.RUN,
        target_id="run_quiet",
        verdict=FeedbackVerdict.CONFIRM,
    )
    service = container.services["feedback"]
    async with uow_factory() as uow:
        groups = await service.pending_by_target(uow, "acme")
        first_page = await service.pending_by_target(uow, "acme", limit=1)
        g = first_page[0]
        rest = await service.pending_by_target(
            uow, "acme", before=(g.latest_at, g.target_kind.value, g.target_id)
        )
        other_tenant = await service.pending_by_target(uow, "globex")
    assert [(g.target_id, g.votes, g.thumbs_up, g.thumbs_down, g.voters) for g in groups] == [
        ("run_quiet", 1, 1, 0, 1),
        ("run_popular", 3, 1, 2, 3),
    ]
    assert groups[0].latest_at == last.created_at
    assert [g.target_id for g in first_page] == ["run_quiet"]
    assert [g.target_id for g in rest] == ["run_popular"]
    assert other_tenant == []

    # the admin dismisses the whole target: every vote on it, and only those
    decided, _, _ = await _bulk(
        container, uow_factory, approve=False, target=(FeedbackTargetKind.RUN, "run_popular")
    )
    assert len(decided) == 3
    async with uow_factory() as uow:
        left = await service.pending_by_target(uow, "acme")
        dismissed = await service.pending_by_target(uow, "acme", state=ReviewState.DISMISSED)
        assert await uow.tools.outcome("acme", "run_popular") is None
    assert [g.target_id for g in left] == ["run_quiet"]
    assert [(g.target_id, g.votes) for g in dismissed] == [("run_popular", 3)]


async def test_votes_on_a_retracted_or_deleted_memory(container, uow_factory) -> None:
    memory = await _first_memory(container, uow_factory)
    service = container.services["feedback"]
    vote = await _vote(
        container,
        uow_factory,
        ALICE,
        target_kind=FeedbackTargetKind.MEMORY,
        target_id=memory.memory_id,
        verdict=FeedbackVerdict.CONFIRM,
    )
    # the owner forgets it while the vote waits: erasure is immediate, the vote is not
    async with uow_factory() as uow:
        await container.services["memory"].forget(uow, ALICE, memory.memory_id)
        await uow.commit()
    await container.tasks.drain()
    with pytest.raises(NotFound):
        await _vote(
            container,
            uow_factory,
            ALICE,
            target_kind=FeedbackTargetKind.MEMORY,
            target_id=memory.memory_id,
            verdict=FeedbackVerdict.REJECT,
        )
    await _bulk(container, uow_factory, approve=True, feedback_ids=[vote.feedback_id])
    applied = await _get(container, uow_factory, vote.feedback_id)
    assert applied.projection is not None and applied.projection.action is ProjectionAction.NONE
    assert applied.projection.reason == "the memory is gone"

    # a vote cast on a memory that is retracted by the time it is approved changes nothing
    await _observe(container, uow_factory, ALICE, "The quarterly stock count is on Fridays.")
    other = next(
        m for m in await _memories(uow_factory, ALICE, container) if "stock count" in m.content
    )
    up = await _vote(
        container,
        uow_factory,
        ALICE,
        target_kind=FeedbackTargetKind.MEMORY,
        target_id=other.memory_id,
        verdict=FeedbackVerdict.CONFIRM,
    )
    await _vote(  # the administrator's own thumbs-down: applied at once
        container,
        uow_factory,
        ALICE,
        trusted=True,
        target_kind=FeedbackTargetKind.MEMORY,
        target_id=other.memory_id,
        verdict=FeedbackVerdict.REJECT,
    )
    async with uow_factory() as uow:
        await service.review(uow, "acme", up.feedback_id, approve=True, reviewed_by=REVIEWER)
        await uow.commit()
    await container.tasks.drain()
    late = await _get(container, uow_factory, up.feedback_id)
    assert late.projection is not None and late.projection.action is ProjectionAction.NONE
    assert "retracted" in (late.projection.reason or "")
    async with uow_factory() as uow:
        unchanged = await uow.memories.get("acme", other.memory_id)
    assert unchanged is not None and unchanged.reinforcement_count == other.reinforcement_count


async def test_another_tenants_vote_is_refused_and_left_alone(container, uow_factory) -> None:
    globex = MemoryExecutionContext(tenant_id="globex", user_id="gina")
    theirs = await _vote(
        container,
        uow_factory,
        globex,
        target_kind=FeedbackTargetKind.RUN,
        target_id="run_g",
        verdict=FeedbackVerdict.REJECT,
        source=FeedbackSource.HUMAN,
    )
    mine = await _vote(
        container,
        uow_factory,
        ALICE,
        target_kind=FeedbackTargetKind.RUN,
        target_id="run_g",
        verdict=FeedbackVerdict.REJECT,
    )
    with pytest.raises(NotFound):
        await _bulk(
            container,
            uow_factory,
            approve=True,
            feedback_ids=[mine.feedback_id, theirs.feedback_id],
        )
    with pytest.raises(NotFound):
        async with uow_factory() as uow:
            await container.services["feedback"].review(
                uow, "acme", theirs.feedback_id, approve=True, reviewed_by=REVIEWER
            )
    assert (await _get(container, uow_factory, mine.feedback_id)).pending, "nothing decided"
    assert (await _get(container, uow_factory, theirs.feedback_id, "globex")).pending
    # a target-wide review only ever takes the reviewing tenant's votes
    decided, _, _ = await _bulk(
        container, uow_factory, approve=False, target=(FeedbackTargetKind.RUN, "run_g")
    )
    assert [f.feedback_id for f in decided] == [mine.feedback_id]
    assert (await _get(container, uow_factory, theirs.feedback_id, "globex")).pending


async def _shared_memory(container, uow_factory) -> tuple[str, MemoryExecutionContext]:
    """Alice's memory shared with a workspace of hers and Carol's (fresh per run: a workspace
    id is never reused). Returns its id and Carol's context in that workspace."""
    tenancy = container.services["tenancy"]
    ws = f"fin-{new_id('turn')[-8:].lower()}"
    async with uow_factory() as uow:
        await tenancy.create_workspace(uow, "acme", name="Finance", workspace_id=ws)
        for member in ("user:alice", "user:carol"):
            await tenancy.set_member(uow, "acme", ws, member, role="member", added_by="test")
        await uow.commit()
    alice = ALICE.model_copy(update={"workspace_id": ws})
    carol = CAROL.model_copy(update={"workspace_id": ws})
    async with uow_factory() as uow:
        ack = await container.services["memory"].remember(
            uow,
            alice,
            content="The fin team closes the books on the third working day.",
            memory_type=MemoryType.SEMANTIC,
            lifetime=Lifetime.LONG_TERM,
            visibility=Visibility.WORKSPACE,
        )
        await uow.commit()
    async with uow_factory() as uow:
        await container.services["memory"].get_memory(uow, carol, ack.memory_id)  # readable
    return ack.memory_id, carol


async def test_who_may_edit_a_shared_memory_and_whose_thumbs_wait(container, uow_factory) -> None:
    memory_id, carol = await _shared_memory(container, uow_factory)
    edit = {
        "target_kind": FeedbackTargetKind.MEMORY,
        "target_id": memory_id,
        "verdict": FeedbackVerdict.CORRECT,
        "correction": "The fin team closes the books on the fifth working day.",
    }
    # a reader who is not the owner may not rewrite it; one who cannot read it is told
    # neither that it exists nor whose it is
    with pytest.raises(ScopeDenied):
        await _vote(container, uow_factory, carol, **edit)
    with pytest.raises(ScopeDenied):
        await _vote(container, uow_factory, carol, **{**edit, "verdict": FeedbackVerdict.EDIT})
    with pytest.raises((NotFound, ScopeDenied)):
        await _vote(container, uow_factory, BOB, **edit)
    # an agent acting for Carol is Carol
    with pytest.raises(ScopeDenied):
        await _vote(container, uow_factory, carol.model_copy(update={"agent_id": "helper"}), **edit)

    # the reader's thumbs-down is a vote: pending, the memory untouched
    down = await _vote(
        container,
        uow_factory,
        carol,
        target_kind=FeedbackTargetKind.MEMORY,
        target_id=memory_id,
        verdict=FeedbackVerdict.REJECT,
    )
    assert down.pending
    async with uow_factory() as uow:
        kept = await uow.memories.get("acme", memory_id)
    assert kept is not None and kept.temporal.status is TemporalStatus.CURRENT

    # a tenant admin in person edits it at once
    await _admin(container, uow_factory)
    admin = ROOT.model_copy(update={"workspace_id": carol.workspace_id})
    by_admin = await _vote(container, uow_factory, admin, **edit)
    assert by_admin.review is None
    applied = await _get(container, uow_factory, by_admin.feedback_id)
    assert applied.projection is not None
    assert applied.projection.action is ProjectionAction.MEMORY_SUPERSEDED
    # Carol's thumbs-down still waits for review; approved now, it finds nothing current
    assert (await _get(container, uow_factory, down.feedback_id)).pending


async def test_an_agent_edits_for_its_user_and_no_one_else(container, uow_factory) -> None:
    memory_id, carol = await _shared_memory(container, uow_factory)
    edit = {
        "target_kind": FeedbackTargetKind.MEMORY,
        "target_id": memory_id,
        "verdict": FeedbackVerdict.EDIT,
        "correction": {"content": "The fin team closes the books on the fourth working day."},
    }
    # the agent acting for Carol may not edit Alice's memory ...
    for_carol = carol.model_copy(update={"agent_id": "helper", "agent_run_id": "run_c"})
    with pytest.raises(ScopeDenied):
        await _vote(container, uow_factory, for_carol, **edit)
    # ... the same agent acting for Alice may, at once
    helper = HELPER.model_copy(update={"workspace_id": carol.workspace_id})
    for_alice = await _vote(container, uow_factory, helper, **edit)
    assert for_alice.review is None
    applied = await _get(container, uow_factory, for_alice.feedback_id)
    assert applied.projection is not None
    assert applied.projection.action is ProjectionAction.MEMORY_SUPERSEDED


# ---------------------------------------------------------------- review fixes, on PostgreSQL


async def _submit_alone(container, uow_factory, ctx, **fields) -> Feedback:
    """One submission in its own transaction, as a concurrent request would be."""
    async with uow_factory() as uow:
        stored, _ = await container.services["feedback"].submit(
            uow, ctx, Feedback(tenant_id=ctx.tenant_id, **fields)
        )
        await uow.commit()
    return stored


async def test_concurrent_votes_of_one_person_leave_one_pending(container, uow_factory) -> None:
    """Alice's web thumbs-down and her agent's thumbs-up on one run at the same moment, ten
    times over: the advisory lock is the voter's, so after each pair one vote is pending (a
    lock per agent let both through, each blind to the other's uncommitted row)."""
    run = {"target_kind": FeedbackTargetKind.RUN, "target_id": "run_race"}
    for _ in range(10):
        await asyncio.gather(
            _submit_alone(container, uow_factory, ALICE, verdict=FeedbackVerdict.REJECT, **run),
            _submit_alone(container, uow_factory, HELPER, verdict=FeedbackVerdict.CONFIRM, **run),
        )
        async with uow_factory() as uow:
            pending = await uow.feedback.pending_on(
                "acme", target_kind=FeedbackTargetKind.RUN, target_id="run_race"
            )
        assert len(pending) == 1, [(f.agent_id, f.verdict) for f in pending]
    async with uow_factory() as uow:
        everything = await uow.feedback.list_for(
            "acme", target_kind=FeedbackTargetKind.RUN, target_id="run_race", limit=100
        )
    assert len(everything) == 20
    newest = max(everything, key=lambda f: (f.created_at, f.feedback_id))
    assert pending[0].feedback_id == newest.feedback_id, "the last click is the vote"


async def _mid_confidence(container, uow_factory):
    """A memory whose confidence is away from the bounds, so a step and its undoing are
    exact (at 1.0 or the floor a step is clipped, and undoing it cannot know by how much)."""
    memory = await _first_memory(container, uow_factory)
    async with uow_factory() as uow:
        row = await uow.memories.get("acme", memory.memory_id)
        assert row is not None
        row.confidence = 0.5
        await uow.memories.update(row)
        await uow.commit()
        return row


async def test_a_vote_counts_once_across_approvals_and_a_flip_undoes_it(
    container, uow_factory
) -> None:
    memory = await _mid_confidence(container, uow_factory)
    service = container.services["feedback"]
    on_memory = {"target_kind": FeedbackTargetKind.MEMORY, "target_id": memory.memory_id}

    async def approve(vote: Feedback) -> Feedback:
        await _bulk(container, uow_factory, approve=True, feedback_ids=[vote.feedback_id])
        return await _get(container, uow_factory, vote.feedback_id)

    async def standing():
        async with uow_factory() as uow:
            m = await uow.memories.get("acme", memory.memory_id)
        assert m is not None
        return m.temporal.status, m.confidence, m.reinforcement_count

    up = await approve(await _vote(container, uow_factory, ALICE, verdict="confirm", **on_memory))
    after_up = await standing()
    assert after_up[2] == memory.reinforcement_count + 1

    # the same thumb again: stored dismissed under the id sent, pointing at the vote that
    # counts, nothing applied
    async with uow_factory() as uow:
        repeat, created = await service.submit(
            uow,
            ALICE,
            Feedback(tenant_id="acme", feedback_id="fb_again", verdict="approve", **on_memory),
        )
        await uow.commit()
    await container.tasks.drain()
    assert created and repeat.review is not None and repeat.review.replaced_by == up.feedback_id
    read_back = await _get(container, uow_factory, "fb_again")
    assert read_back.review is not None and read_back.review.state is ReviewState.DISMISSED
    assert await standing() == after_up

    # flipped: pending, and approving it undoes the thumbs-up before retracting
    down = await _vote(container, uow_factory, ALICE, verdict="reject", **on_memory)
    assert down.pending and await standing() == after_up
    down = await approve(down)
    assert down.projection is not None and down.projection.reversed == up.feedback_id
    assert down.projection.action is ProjectionAction.MEMORY_RETRACTED
    status, confidence, count = await standing()
    assert status is TemporalStatus.RETRACTED
    assert (confidence, count) == pytest.approx((memory.confidence, memory.reinforcement_count))

    # flipped back: the retraction is lifted and the thumbs-up counts once again
    back = await approve(await _vote(container, uow_factory, ALICE, verdict="confirm", **on_memory))
    assert back.projection is not None and back.projection.reversed == down.feedback_id
    assert await standing() == pytest.approx(after_up)
    assert (await standing())[0] is TemporalStatus.CURRENT


async def test_a_flipped_run_vote_moves_cited_memories_back_first(container, uow_factory) -> None:
    memory = await _mid_confidence(container, uow_factory)
    cited = [{"source_type": "memory", "source_id": memory.memory_id}]
    run = {"target_kind": FeedbackTargetKind.RUN, "target_id": "run_flip", "evidence_refs": cited}

    async def confidence() -> float:
        async with uow_factory() as uow:
            m = await uow.memories.get("acme", memory.memory_id)
        assert m is not None
        return m.confidence

    for verdict, expected in (
        ("confirm", memory.confidence + 0.05),
        ("reject", memory.confidence - 0.05),
        ("confirm", memory.confidence + 0.05),
    ):
        vote = await _vote(container, uow_factory, ALICE, verdict=verdict, **run)
        await _bulk(container, uow_factory, approve=True, feedback_ids=[vote.feedback_id])
        assert await confidence() == pytest.approx(min(1.0, expected)), verdict
    async with uow_factory() as uow:
        outcome = await uow.tools.outcome("acme", "run_flip")
    assert outcome is not None and outcome.success


async def test_a_run_status_counts_only_from_a_credential_that_may_act_as_the_agent(
    container, uow_factory
) -> None:
    """``source=system`` naming the caller's run: its status when the credential may act as
    that agent (an issued service key, the harness); a vote otherwise (a person's own token),
    which never overwrites the run's outcome on its word."""
    run_ctx = ALICE.model_copy(update={"agent_id": "helper", "agent_run_id": "run_spoof"})
    status = {"target_kind": FeedbackTargetKind.RUN, "target_id": "run_spoof", "source": "system"}
    service = container.services["feedback"]
    async with uow_factory() as uow:
        claimed, _ = await service.submit(
            uow, run_ctx, Feedback(tenant_id="acme", verdict="reject", **status)
        )
        await uow.commit()
    await container.tasks.drain()
    assert claimed.pending
    async with uow_factory() as uow:
        assert await uow.tools.outcome("acme", "run_spoof") is None
        vouched, _ = await service.submit(
            uow, run_ctx, Feedback(tenant_id="acme", verdict="confirm", **status), run_vouched=True
        )
        await uow.commit()
    await container.tasks.drain()
    assert vouched.review is None
    async with uow_factory() as uow:
        outcome = await uow.tools.outcome("acme", "run_spoof")
    assert outcome is not None and outcome.success and outcome.source == "system"


async def test_an_agent_for_an_admin_edits_only_its_users_memories(container, uow_factory) -> None:
    memory_id, carol = await _shared_memory(container, uow_factory)
    await _admin(container, uow_factory)
    ws = carol.workspace_id
    for_root = ROOT.model_copy(update={"workspace_id": ws, "agent_id": "helper"})
    edit = {
        "target_kind": FeedbackTargetKind.MEMORY,
        "target_id": memory_id,
        "verdict": FeedbackVerdict.CORRECT,
        "correction": "The fin team closes the books on the sixth working day.",
    }
    with pytest.raises((ScopeDenied, NotFound)):
        await _vote(container, uow_factory, for_root, **edit)
    async with uow_factory() as uow:
        kept = await uow.memories.get("acme", memory_id)
    assert kept is not None and kept.temporal.status is TemporalStatus.CURRENT


@pytest.mark.parametrize(
    ("query", "index"),
    [
        (
            "SELECT * FROM feedback WHERE tenant_id = 'acme' AND review_state = 'dismissed' "
            "ORDER BY created_at DESC, feedback_id DESC LIMIT 101",
            "ix_feedback_review",
        ),
        (
            "SELECT target_kind, target_id, count(*), max(created_at) FROM feedback "
            "WHERE tenant_id = 'acme' AND review_state = 'approved' "
            "AND review_replaced_by IS NULL GROUP BY target_kind, target_id",
            "ix_feedback_review_target",
        ),
    ],
)
async def test_review_reports_read_an_index(container, uow_factory, query, index) -> None:
    """``review=approved|dismissed`` and its grouped form read the rows of one state through
    an index (migration 0027), not the tenant's whole history. Sequential scans are switched
    off so the plan shows an index is usable at all; on a table this small the planner
    would otherwise rightly prefer a scan. Which of the two review indexes it picks for the
    grouped form depends on the statistics (the target one pays off on a large table), so
    either one passes; a plan through neither does not."""
    import psycopg

    from tests.conftest import DB_URL

    url = DB_URL.replace("postgresql+psycopg://", "postgresql://")
    with psycopg.connect(url) as conn:
        conn.execute("SET enable_seqscan = off")
        plan = "\n".join(row[0] for row in conn.execute(f"EXPLAIN {query}").fetchall())
    assert index in plan or "ix_feedback_review" in plan, plan
    assert "Seq Scan" not in plan, plan


async def test_a_claimed_judge_on_a_persons_thumbs_is_one_vote(container, uow_factory) -> None:
    """B1: a person's client writes ``source=judge`` on five thumbs-up. Without a credential
    vouched for an agent the claim makes no difference: one pending vote, and approving
    everything pending on the target applies it once."""
    memory = await _mid_confidence(container, uow_factory)
    on_memory = {"target_kind": FeedbackTargetKind.MEMORY, "target_id": memory.memory_id}
    for _ in range(5):
        await _vote(container, uow_factory, ALICE, verdict="confirm", source="judge", **on_memory)
    decided, _, _ = await _bulk(
        container, uow_factory, approve=True, target=(FeedbackTargetKind.MEMORY, memory.memory_id)
    )
    assert len(decided) == 1
    async with uow_factory() as uow:
        once = await uow.memories.get("acme", memory.memory_id)
    assert once is not None
    assert once.reinforcement_count == memory.reinforcement_count + 1
    assert once.confidence == pytest.approx(memory.confidence + 0.1)


async def test_an_approved_vote_whose_job_has_not_run_never_counts_twice(
    container, uow_factory
) -> None:
    """B2: 👍 applied; 👎 approved but its job not run yet; 👍 again, approved, and its job
    runs first. The 👎 that never applied is closed, nothing is undone, and the second 👍 is
    the first one counting already: the memory is reinforced once, not twice."""
    memory = await _mid_confidence(container, uow_factory)
    service = container.services["feedback"]
    on_memory = {"target_kind": FeedbackTargetKind.MEMORY, "target_id": memory.memory_id}
    v1 = await _vote(container, uow_factory, ALICE, verdict="confirm", **on_memory)
    await _bulk(container, uow_factory, approve=True, feedback_ids=[v1.feedback_id])

    async def approve_without_running(vote: Feedback) -> None:
        async with uow_factory() as uow:
            await service.review(uow, "acme", vote.feedback_id, approve=True, reviewed_by=REVIEWER)
            await uow.commit()  # its feedback.project job is queued, not run

    async def submit(verdict: str) -> Feedback:
        async with uow_factory() as uow:
            stored, _ = await service.submit(
                uow, ALICE, Feedback(tenant_id="acme", verdict=verdict, **on_memory)
            )
            await uow.commit()
        return stored

    v2 = await submit("reject")
    await approve_without_running(v2)
    v3 = await submit("confirm")
    assert v3.pending, "a flip against the approved 👎"
    await approve_without_running(v3)
    await service.project("acme", v3.feedback_id)  # v3's job first
    await container.tasks.drain()  # then v2's (and anything else queued)
    await container.tasks.drain()
    async with uow_factory() as uow:
        m = await uow.memories.get("acme", memory.memory_id)
    assert m is not None and m.temporal.status is TemporalStatus.CURRENT
    assert m.reinforcement_count == memory.reinforcement_count + 1, "v1 counts, once"
    assert m.confidence == pytest.approx(memory.confidence + 0.1)
    p2 = (await _get(container, uow_factory, v2.feedback_id)).projection
    p3 = (await _get(container, uow_factory, v3.feedback_id)).projection
    assert p2 is not None and p2.action is ProjectionAction.NONE
    assert p3 is not None and p3.action is ProjectionAction.NONE and p3.reversed is None
