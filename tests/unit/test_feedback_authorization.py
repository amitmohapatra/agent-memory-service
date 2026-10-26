"""Who may say what about a memory: reading grants affirmation, rewriting needs the owner;
which verdicts wait for review, one vote per voter per target, and bulk review."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType, ScopeLevel, Visibility
from memory_service.domain.errors import Conflict, NotFound, ScopeDenied, ValidationFailed
from memory_service.domain.evidence import EvidenceRef, EvidenceSource
from memory_service.domain.feedback import (
    REPLACED_REASON,
    Feedback,
    FeedbackSource,
    FeedbackTargetKind,
    FeedbackVerdict,
    ProjectionAction,
    ReviewState,
)
from memory_service.domain.grounding import ClaimReport, GroundingReport
from memory_service.domain.memory import CanonicalMemory, Scope, TemporalState
from memory_service.domain.tools import RunOutcome, ToolInvocation, stable_hash
from memory_service.modules.feedback.service import CALL_RECORD_CHECKS, FeedbackService
from memory_service.modules.feedback.votes import ALREADY_COUNTED_REASON, is_vote
from memory_service.modules.grounding.judge import judge_run

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
    """What the service asks of the repository, over a dict (one tenant per row key)."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], Feedback] = {}

    async def get(self, tenant_id, feedback_id):
        return self.rows.get((tenant_id, feedback_id))

    async def add(self, feedback) -> bool:
        self.rows[(feedback.tenant_id, feedback.feedback_id)] = feedback
        return True

    async def set_review(self, tenant_id, feedback_id, review) -> None:
        row = self.rows[(tenant_id, feedback_id)]
        self.rows[(tenant_id, feedback_id)] = row.model_copy(update={"review": review})

    async def set_projection(self, tenant_id, feedback_id, projection) -> None:
        row = self.rows[(tenant_id, feedback_id)]
        self.rows[(tenant_id, feedback_id)] = row.model_copy(update={"projection": projection})

    async def pending_on(
        self,
        tenant_id,
        *,
        target_kind,
        target_id,
        user_id=None,
        agent_id=None,
        voter_only=False,
        votes_only=False,
        limit=1000,
    ):
        found = [
            r
            for (t, _), r in self.rows.items()
            if t == tenant_id
            and r.pending
            and r.target_kind is target_kind
            and r.target_id == target_id
            and (not voter_only or _same_voter(r, user_id, agent_id))
            and (not votes_only or is_vote(r))
        ]
        return sorted(found, key=lambda r: (r.created_at, r.feedback_id))[:limit]

    async def list_for(self, tenant_id, *, target_kind, target_id, limit=100):
        return [
            r
            for (t, _), r in self.rows.items()
            if t == tenant_id and r.target_kind is target_kind and r.target_id == target_id
        ]

    async def applied_votes(
        self, tenant_id, *, target_kind, target_id, user_id, agent_id, limit=20
    ):
        found = [
            r
            for (t, _), r in self.rows.items()
            if t == tenant_id
            and r.target_kind is target_kind
            and r.target_id == target_id
            and r.review is not None
            and r.review.state is ReviewState.APPROVED
            and is_vote(r)
            and _same_voter(r, user_id, agent_id)
        ]
        return sorted(found, key=lambda r: (r.created_at, r.feedback_id), reverse=True)[:limit]


def _same_voter(r: Feedback, user_id, agent_id) -> bool:
    if user_id is not None:
        return r.user_id == user_id
    return r.user_id is None and r.agent_id == agent_id


class _Uow:
    def __init__(self) -> None:
        self.feedback = _FeedbackRepo()
        self.jobs: list = []
        self.locks: list[str] = []

    async def enqueue(self, spec):
        self.jobs.append(spec)

    async def serialize(self, *keys: str) -> None:
        self.locks.extend(keys)


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


async def test_a_reader_may_vote_a_memory_down_and_it_waits() -> None:
    """A thumbs-down changes nothing until a tenant admin approves it (ADR 0039), so reading
    is enough to cast one; it was owner-only while it retracted the memory at once."""
    stored, created = await _service().submit(_Uow(), READER, _record(FeedbackVerdict.REJECT))
    assert created and stored.pending


@pytest.mark.parametrize("verdict", [FeedbackVerdict.CORRECT, FeedbackVerdict.EDIT])
async def test_only_the_owner_or_an_admin_may_rewrite(verdict: FeedbackVerdict) -> None:
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
    """A vote waits, and so does the grounding judge's verdict; an owner's edit, a run's own
    status, a tool-call decision and an admin in person do not. An agent acting for an admin
    is still an agent."""
    run_ctx = OWNER.model_copy(update={"agent_id": "ref", "agent_run_id": "run_1"})
    admin_agent = READER.model_copy(update={"agent_id": "ref"})
    cases = [
        (READER, _record(FeedbackVerdict.CONFIRM), {}, True),
        # a thumbs-down on a memory is a vote, even its owner's (ADR 0039)
        (OWNER, _record(FeedbackVerdict.REJECT), {}, True),
        (AGENT_OF_OWNER, _record(FeedbackVerdict.REJECT), {}, True),
        # a correction carrying new text is the owner editing their memory
        (OWNER, _record(FeedbackVerdict.CORRECT), {}, False),
        (OWNER, _record(FeedbackVerdict.EDIT), {}, False),
        (AGENT_OF_OWNER, _record(FeedbackVerdict.CORRECT), {}, False),
        # ... and one carrying no text is a thumbs-down in other words
        (
            OWNER,
            Feedback(
                tenant_id="acme",
                target_kind=FeedbackTargetKind.MEMORY,
                target_id="mem_1",
                verdict=FeedbackVerdict.CORRECT,
                correction={"reason": "wrong"},
            ),
            {},
            True,
        ),
        # a person's vote on a run, even from the run itself, waits
        (
            run_ctx,
            Feedback(
                tenant_id="acme",
                target_kind=FeedbackTargetKind.RUN,
                target_id="run_1",
                verdict=FeedbackVerdict.REJECT,
            ),
            {},
            True,
        ),
        # a tool-call decision from an agent key applies as it arrives
        (
            run_ctx,
            Feedback(
                tenant_id="acme",
                target_kind=FeedbackTargetKind.TOOL_CALL,
                target_id="call_2",
                verdict=FeedbackVerdict.REJECT,
                source=FeedbackSource.INTERRUPT,
            ),
            {},
            False,
        ),
        # the admin's own word on a memory and on a run: no queue
        (READER, _record(FeedbackVerdict.REJECT), {"trusted": True}, False),
        # a run's own status, from a credential that may act as that agent
        (
            run_ctx,
            Feedback(
                tenant_id="acme",
                target_kind=FeedbackTargetKind.RUN,
                target_id="run_1",
                verdict=FeedbackVerdict.CONFIRM,
                source=FeedbackSource.SYSTEM,
            ),
            {"run_vouched": True},
            False,
        ),
        # the same claim from a credential that vouches for no agent (a person's own token)
        # is a vote: it never overwrites the run's outcome on its word
        (
            run_ctx,
            Feedback(
                tenant_id="acme",
                target_kind=FeedbackTargetKind.RUN,
                target_id="run_1",
                verdict=FeedbackVerdict.REJECT,
                source=FeedbackSource.SYSTEM,
            ),
            {},
            True,
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
            {"run_vouched": True},
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
            {"run_vouched": True},
            True,
        ),
        (READER, _record(FeedbackVerdict.CONFIRM), {"trusted": True}, False),
        (admin_agent, _record(FeedbackVerdict.CONFIRM), {}, True),
    ]
    for ctx, record, kwargs, waits in cases:
        uow = _Uow()
        admins = {"carol"} if ctx is admin_agent or kwargs.get("trusted") else frozenset()
        stored, _ = await _service(admins).submit(uow, ctx, record, **kwargs)
        assert stored.pending is waits, (ctx.principal_id, record.target_kind, record.source)
        assert bool(uow.jobs) is not waits
    stored, _ = await _service({"carol"}).submit(_Uow(), READER, _record(FeedbackVerdict.CONFIRM))
    assert not stored.pending, "a tenant admin in person"


# ---------------------------------------------------------------- one vote per voter (ADR 0039)


def _run_vote(verdict: FeedbackVerdict, run_id: str = "run_9", **fields) -> Feedback:
    return Feedback(
        tenant_id="acme",
        target_kind=FeedbackTargetKind.RUN,
        target_id=run_id,
        verdict=verdict,
        **fields,
    )


def _state(r: Feedback) -> str | None:
    """The review state, with "replaced" for a vote dismissed because its voter voted again:
    on the wire that is ``dismissed`` with ``replaced_by`` (no state older clients lack)."""
    if r.review is None:
        return None
    if r.review.replaced_by is not None:
        assert r.review.state is ReviewState.DISMISSED
        assert r.review.note in (REPLACED_REASON, ALREADY_COUNTED_REASON)
        return "replaced"
    return r.review.state.value


async def _states(uow: _Uow) -> dict[str, str | None]:
    return {fid: _state(r) for (_, fid), r in uow.feedback.rows.items()}


async def test_a_flipped_vote_replaces_the_earlier_one_and_repeats_count_once() -> None:
    service, uow = _service(), _Uow()
    up, _ = await service.submit(uow, READER, _run_vote(FeedbackVerdict.CONFIRM))
    again, _ = await service.submit(uow, READER, _run_vote(FeedbackVerdict.CONFIRM))
    down, _ = await service.submit(uow, READER, _run_vote(FeedbackVerdict.REJECT))
    states = await _states(uow)
    assert states == {
        up.feedback_id: "replaced",
        again.feedback_id: "replaced",
        down.feedback_id: "pending",
    }
    old = await uow.feedback.get("acme", up.feedback_id)
    assert old.review.replaced_by == again.feedback_id
    assert (
        await uow.feedback.get("acme", again.feedback_id)
    ).review.replaced_by == down.feedback_id
    assert old.projection is not None and old.projection.action.value == "none"
    assert old.projection.reason == REPLACED_REASON
    assert uow.jobs == [], "neither a pending nor a replaced vote schedules anything"
    # the same voter's click on another target is a vote of its own
    other, _ = await service.submit(uow, READER, _run_vote(FeedbackVerdict.REJECT, "run_10"))
    assert (await _states(uow))[other.feedback_id] == "pending"
    assert (await _states(uow))[down.feedback_id] == "pending"


async def test_two_voters_on_one_target_keep_a_vote_each() -> None:
    service, uow = _service(), _Uow()
    alice, _ = await service.submit(uow, OWNER, _run_vote(FeedbackVerdict.CONFIRM))
    carol, _ = await service.submit(uow, READER, _run_vote(FeedbackVerdict.REJECT))
    assert await _states(uow) == {alice.feedback_id: "pending", carol.feedback_id: "pending"}


async def test_a_person_is_one_voter_whichever_agent_carries_the_vote() -> None:
    service, uow = _service(), _Uow()
    direct, _ = await service.submit(uow, OWNER, _run_vote(FeedbackVerdict.CONFIRM))
    via_agent, _ = await service.submit(uow, AGENT_OF_OWNER, _run_vote(FeedbackVerdict.REJECT))
    assert await _states(uow) == {direct.feedback_id: "replaced", via_agent.feedback_id: "pending"}
    # an agent with no person behind it is its own voter
    bare = MemoryExecutionContext(tenant_id="acme", agent_id="bot")
    other_bot = MemoryExecutionContext(tenant_id="acme", agent_id="bot2")
    b1, _ = await service.submit(uow, bare, _run_vote(FeedbackVerdict.CONFIRM))
    b2, _ = await service.submit(uow, other_bot, _run_vote(FeedbackVerdict.CONFIRM))
    b3, _ = await service.submit(uow, bare, _run_vote(FeedbackVerdict.REJECT))
    states = await _states(uow)
    assert (states[b1.feedback_id], states[b2.feedback_id], states[b3.feedback_id]) == (
        "replaced",
        "pending",
        "pending",
    )
    assert states[via_agent.feedback_id] == "pending", "a person's vote is not an agent's"


async def test_a_retry_or_a_word_applied_at_once_replaces_nothing() -> None:
    service, uow = _service({"carol"}), _Uow()
    vote, _ = await service.submit(
        uow, OWNER, _run_vote(FeedbackVerdict.CONFIRM, feedback_id="fb_v1")
    )
    retried, created = await service.submit(
        uow, OWNER, _run_vote(FeedbackVerdict.CONFIRM, feedback_id="fb_v1")
    )
    assert not created and retried.pending
    # the owner's correction of a memory is an edit, not a vote: their pending thumbs-down
    # on it stays for the admin to decide
    down, _ = await service.submit(uow, OWNER, _record(FeedbackVerdict.REJECT))
    edit, _ = await service.submit(uow, OWNER, _record(FeedbackVerdict.CORRECT))
    states = await _states(uow)
    assert states[vote.feedback_id] == "pending" and states[down.feedback_id] == "pending"
    assert states[edit.feedback_id] is None
    # an admin in person is applied at once and leaves other voters' votes alone
    admin, _ = await service.submit(uow, READER, _run_vote(FeedbackVerdict.REJECT))
    assert admin.review is None and (await _states(uow))[vote.feedback_id] == "pending"


# ---------------------------------------------------------------- bulk review


async def _queue(service: FeedbackService, uow: _Uow) -> tuple[str, str, str]:
    a, _ = await service.submit(uow, OWNER, _run_vote(FeedbackVerdict.CONFIRM))
    b, _ = await service.submit(uow, READER, _run_vote(FeedbackVerdict.REJECT))
    c, _ = await service.submit(uow, READER, _run_vote(FeedbackVerdict.REJECT, "run_other"))
    return a.feedback_id, b.feedback_id, c.feedback_id


async def test_bulk_approve_of_a_mix_decides_each_pending_vote_once() -> None:
    service, uow = _service(), _Uow()
    a, b, c = await _queue(service, uow)
    await service.review(uow, "acme", b, approve=False, reviewed_by="key:admin")
    uow.jobs.clear()
    decided, unchanged, more = await service.review_many(
        uow, "acme", feedback_ids=[a, b, a], approve=True, reviewed_by="key:admin", note="ok"
    )
    assert [f.feedback_id for f in decided] == [a] and not more
    assert [(f.feedback_id, f.review.state.value) for f in unchanged] == [(b, "dismissed")]
    assert len(uow.jobs) == 1, "one projection, for the one vote decided now"
    # the same call again decides nothing and schedules nothing
    decided, unchanged, _ = await service.review_many(
        uow, "acme", feedback_ids=[a, b], approve=True, reviewed_by="key:admin"
    )
    assert decided == [] and {f.feedback_id for f in unchanged} == {a, b}
    assert len(uow.jobs) == 1
    assert (await _states(uow))[c] == "pending", "a vote not named is left alone"


async def test_bulk_review_of_a_target_takes_every_vote_pending_on_it() -> None:
    service, uow = _service(), _Uow()
    a, b, c = await _queue(service, uow)
    decided, unchanged, _ = await service.review_many(
        uow,
        "acme",
        target=(FeedbackTargetKind.RUN, "run_9"),
        approve=False,
        reviewed_by="key:admin",
    )
    assert {f.feedback_id for f in decided} == {a, b} and unchanged == []
    assert all(f.projection and f.projection.reason == "dismissed in review" for f in decided)
    assert uow.jobs == []
    states = await _states(uow)
    assert (states[a], states[b], states[c]) == ("dismissed", "dismissed", "pending")
    again, _, _ = await service.review_many(
        uow, "acme", target=(FeedbackTargetKind.RUN, "run_9"), approve=True, reviewed_by="k"
    )
    assert again == [], "nothing pending on it any more"


async def test_bulk_review_refuses_unknown_and_other_tenants_ids_before_deciding_any() -> None:
    service, uow = _service(), _Uow()
    a, _, _ = await _queue(service, uow)
    globex = MemoryExecutionContext(tenant_id="globex", user_id="gina")
    theirs, _ = await service.submit(
        uow,
        globex,
        Feedback(tenant_id="globex", target_kind="run", target_id="r", verdict="reject"),
    )
    for ids in ([a, theirs.feedback_id], [a, "fb_missing"]):
        with pytest.raises(NotFound):
            await service.review_many(
                uow, "acme", feedback_ids=ids, approve=True, reviewed_by="key:admin"
            )
    assert (await _states(uow))[a] == "pending" and uow.jobs == []
    with pytest.raises(NotFound):
        await service.review(uow, "acme", theirs.feedback_id, approve=True, reviewed_by="k")
    for bad in ({}, {"feedback_ids": [a], "target": (FeedbackTargetKind.RUN, "run_9")}):
        with pytest.raises(ValidationFailed):
            await service.review_many(uow, "acme", approve=True, reviewed_by="k", **bad)
    with pytest.raises(ValidationFailed):
        await service.review_many(uow, "acme", feedback_ids=[], approve=True, reviewed_by="k")


async def test_a_dismissed_or_replaced_vote_cannot_be_approved_one_by_one() -> None:
    service, uow = _service(), _Uow()
    a, b, _ = await _queue(service, uow)
    await service.review(uow, "acme", a, approve=False, reviewed_by="key:admin")
    with pytest.raises(Conflict):
        await service.review(uow, "acme", a, approve=True, reviewed_by="key:admin")
    flipped, _ = await service.submit(uow, READER, _run_vote(FeedbackVerdict.CONFIRM))
    with pytest.raises(Conflict):
        await service.review(uow, "acme", b, approve=True, reviewed_by="key:admin")
    assert (await _states(uow))[flipped.feedback_id] == "pending"


# ---------------------------------------------------------------- review fixes (ADR 0039)


async def test_one_voter_one_lock_whichever_agent_carries_the_vote() -> None:
    """The lock that serialises a voter's clicks is the voter's: Alice on the web and her
    agent take the same one, so their two votes on one run cannot both stay pending."""
    service, uow = _service(), _Uow()
    await service.submit(uow, OWNER, _run_vote(FeedbackVerdict.REJECT))
    await service.submit(uow, AGENT_OF_OWNER, _run_vote(FeedbackVerdict.CONFIRM))
    votes = [k for k in uow.locks if k.startswith("feedback-vote:")]
    assert votes == ["feedback-vote:acme:run:run_9:user:alice"] * 2


async def test_automated_records_never_replace_a_thumb_nor_the_reverse() -> None:
    """A run's status and a judge's word relayed by a credential vouched for the agent are not
    votes: they wait (they cite memories) without replacing the person's thumb on the run,
    and the thumb does not replace them."""
    service, uow = _service(), _Uow()
    run_ctx = OWNER.model_copy(update={"agent_id": "ref", "agent_run_id": "run_9"})
    cites = [{"source_type": "memory", "source_id": "mem_1"}]
    judge, _ = await service.submit(
        uow,
        run_ctx,
        _run_vote(FeedbackVerdict.REJECT, source="judge", evidence_refs=cites),
        run_vouched=True,
    )
    status, _ = await service.submit(
        uow,
        run_ctx,
        _run_vote(FeedbackVerdict.CONFIRM, source="system", evidence_refs=cites),
        run_vouched=True,
    )
    thumb, _ = await service.submit(uow, OWNER, _run_vote(FeedbackVerdict.CONFIRM))
    again, _ = await service.submit(uow, OWNER, _run_vote(FeedbackVerdict.REJECT))
    states = await _states(uow)
    assert (states[judge.feedback_id], states[status.feedback_id]) == ("pending", "pending")
    assert (states[thumb.feedback_id], states[again.feedback_id]) == ("replaced", "pending")
    assert (judge.vote, status.vote, thumb.vote) == (False, False, True)


async def test_a_claimed_source_does_not_make_a_thumb_anything_but_a_vote() -> None:
    """A person's client writing source=judge (or system) on its thumbs, five times: one
    vote, pending; the claim is kept, and counts for nothing (no credential vouched)."""
    service, uow = _service(), _Uow()
    clicks = [
        (await service.submit(uow, OWNER, _run_vote(FeedbackVerdict.CONFIRM, source=src)))[0]
        for src in ("judge", "judge", "system", "judge", "judge")
    ]
    states = await _states(uow)
    assert [states[c.feedback_id] for c in clicks] == ["replaced"] * 4 + ["pending"]
    assert all(c.vote for c in clicks) and clicks[-1].source is FeedbackSource.JUDGE


async def test_the_same_thumb_as_the_one_applied_changes_nothing() -> None:
    service, uow = _service(), _Uow()
    up, _ = await service.submit(uow, READER, _run_vote(FeedbackVerdict.CONFIRM))
    await service.review(uow, "acme", up.feedback_id, approve=True, reviewed_by="key:admin")
    uow.jobs.clear()
    # a flip waits; then the same thumb as the applied one withdraws the flip and is nothing:
    # stored under the id the client sent, dismissed, pointing at the vote that counts
    flip, _ = await service.submit(uow, READER, _run_vote(FeedbackVerdict.REJECT))
    repeat, created = await service.submit(
        uow, READER, _run_vote(FeedbackVerdict.APPROVE, feedback_id="fb_click_again")
    )
    assert created and repeat.feedback_id == "fb_click_again"
    assert repeat.review.state is ReviewState.DISMISSED
    assert repeat.review.replaced_by == up.feedback_id
    assert repeat.projection is not None and repeat.projection.action.value == "none"
    assert (await uow.feedback.get("acme", "fb_click_again")).review.replaced_by == up.feedback_id
    states = await _states(uow)
    assert states[flip.feedback_id] == "replaced" and uow.jobs == []
    assert (await uow.feedback.get("acme", flip.feedback_id)).review.replaced_by == up.feedback_id


async def test_an_agent_acting_for_an_admin_cannot_rewrite_a_colleagues_memory() -> None:
    """The admin's word is the admin's own: an agent acting for a tenant admin is an agent,
    and an agent edits only its own user's memories."""
    admin_agent = READER.model_copy(update={"agent_id": "ref"})  # carol is the admin here
    with pytest.raises(ScopeDenied, match="owner"):
        await _service({"carol"}).submit(_Uow(), admin_agent, _record(FeedbackVerdict.CORRECT))
    stored, _ = await _service({"carol"}).submit(_Uow(), READER, _record(FeedbackVerdict.CORRECT))
    assert stored.review is None, "the admin in person edits at once"


@pytest.mark.parametrize(
    ("mode", "claims", "agent", "vouches"),
    [
        ("api_key", {"role": "service", "may_act_as": ["*"]}, "buyer", True),
        ("api_key", {"role": "service", "may_act_as": ["user:u1", "agent:buyer"]}, "buyer", True),
        ("api_key", {"role": "service", "may_act_as": ["user:u1"]}, "buyer", False),
        ("api_key", {"role": "platform"}, "buyer", True),
        ("trusted_dev", {}, "buyer", True),
        # an identity provider's token names a person, never a run
        ("jwt", {"sub": "alice", "may_act_as": ["*"]}, "buyer", False),
        ("api_key", {"role": "service", "may_act_as": ["*"]}, None, False),
    ],
)
def test_which_credential_vouches_for_a_runs_own_status(mode, claims, agent, vouches) -> None:
    from types import SimpleNamespace

    from memory_service.api.deps import credential_vouches_for_agent
    from memory_service.modules.auth.authentication import ServicePrincipal

    principal = ServicePrincipal(service_id="k", mode=mode, claims=claims)
    request = SimpleNamespace(state=SimpleNamespace(service_principal=principal))
    assert credential_vouches_for_agent(request, agent) is vouches  # type: ignore[arg-type]


# ---------------------------------------------------------------- the grounding judge's verdict


class _Run:
    """The run outcome, memories and revisions a run verdict's projection touches."""

    def __init__(self) -> None:
        self.outcomes: dict[str, RunOutcome] = {}
        self.confidence = {"mem_1": 0.9}
        self.calls: list[ToolInvocation] = []
        self.counted: list[tuple[str, str, str]] = []

    async def outcome(self, tenant_id, run_id):
        return self.outcomes.get(run_id)

    async def set_outcome(self, outcome) -> None:
        self.outcomes[outcome.run_id] = outcome

    async def get_many(self, tenant_id, ids):
        return [_memory(i, self.confidence[i]) for i in ids if i in self.confidence]

    async def update(self, memory) -> None:
        self.confidence[memory.memory_id] = round(memory.confidence, 4)

    async def bump(self, *_: object) -> None:
        return None

    async def mark_used(self, *_: object) -> None:
        return None

    async def invocations_for_run(self, tenant_id, run_id) -> list:
        return [c for c in self.calls if c.run_id == run_id]

    async def count_verdict(self, *_: object) -> None:
        return None

    async def count_approval(self, tenant_id, agent_id, tool, shape, counter) -> None:
        self.counted.append((tool, shape, counter))


def _memory(memory_id: str, confidence: float) -> CanonicalMemory:
    when = datetime(2026, 9, 1, tzinfo=UTC)
    return CanonicalMemory(
        memory_id=memory_id,
        tenant_id="acme",
        scope=Scope(level=ScopeLevel.USER, tenant_id="acme", user_id="alice"),
        visibility=Visibility.USER,
        owner_principal="user:alice",
        lifetime=Lifetime.LONG_TERM,
        memory_type=MemoryType.SEMANTIC,
        content="S40 sells 90 a week.",
        normalized_hash="h",
        temporal=TemporalState(observed_at=when),
        evidence=[
            EvidenceRef(source_type=EvidenceSource.MESSAGE, source_id="msg_1", observed_at=when)
        ],
        confidence=confidence,
    )


class _ProjectingUow(_Uow):
    def __init__(self, run: _Run, feedback: _FeedbackRepo) -> None:
        super().__init__()
        self.feedback = feedback
        self.tools = self.memories = self.revisions = self.pulls = run

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def commit(self) -> None:
        return None


def _projecting() -> tuple[FeedbackService, _Run, _ProjectingUow]:
    run = _Run()
    uow = _ProjectingUow(run, _FeedbackRepo())
    service = _service({"carol"})
    service.uow_factory = lambda: uow
    return service, run, uow


def _judge(verdict: FeedbackVerdict, feedback_id: str) -> Feedback:
    return _run_vote(
        verdict,
        feedback_id=feedback_id,
        source="judge",
        reviewer="judge:grounding",
        evidence_refs=[{"source_type": "memory", "source_id": "mem_1"}],
    )


async def test_a_judge_verdict_moves_nothing_until_approved_and_then_once() -> None:
    """B2/B12: the grounding judge's verdict waits for a tenant admin, even when an admin's
    own credential carried the /v1/verify call; pending it changes nothing, approved it
    applies once, dismissed it never does."""
    service, run, uow = _projecting()
    admin = READER.model_copy(update={"agent_run_id": "run_9"})
    pending, _ = await service.submit(
        uow, admin, _judge(FeedbackVerdict.REJECT, "fb_j1"), run_vouched=True
    )
    assert pending.pending and not pending.vote and uow.jobs == []
    assert await service.project("acme", "fb_j1") is None
    assert (run.outcomes, run.confidence) == ({}, {"mem_1": 0.9})

    await service.review(uow, "acme", "fb_j1", approve=True, reviewed_by="carol")
    first = await service.project("acme", "fb_j1")
    again = await service.project("acme", "fb_j1")
    assert first == again and first is not None and first.memory_ids == ["mem_1"]
    assert run.outcomes["run_9"].source == "judge" and not run.outcomes["run_9"].success
    assert run.confidence == {"mem_1": 0.85}

    await service.submit(uow, admin, _judge(FeedbackVerdict.REJECT, "fb_j2"), run_vouched=True)
    await service.review(uow, "acme", "fb_j2", approve=False, reviewed_by="carol")
    assert await service.project("acme", "fb_j2") is not None
    assert run.confidence == {"mem_1": 0.85}


async def test_an_outranked_judge_verdict_moves_no_memory() -> None:
    """B12: a person decided the run; the judge's verdict, even approved, changes neither the
    outcome nor the confidence of what it cites."""
    service, run, uow = _projecting()
    run.outcomes["run_9"] = RunOutcome(
        tenant_id="acme", run_id="run_9", success=True, source="human"
    )
    await service.submit(uow, OWNER, _judge(FeedbackVerdict.REJECT, "fb_j3"), run_vouched=True)
    await service.review(uow, "acme", "fb_j3", approve=True, reviewed_by="carol")
    projection = await service.project("acme", "fb_j3")
    assert projection is not None and projection.memory_ids == []
    assert run.outcomes["run_9"].success and run.confidence == {"mem_1": 0.9}


async def test_the_judges_verdict_is_no_vote_of_the_user_it_judged_for() -> None:
    """The /v1/verify call is made in the user's scope, but the judge's verdict is not that
    user's vote: it neither replaces their pending thumb on the run nor is replaced by it,
    and it waits on its own."""
    service, _, uow = _projecting()
    ctx = OWNER.model_copy(update={"agent_id": "ref", "agent_run_id": "run_9"})
    thumb, _ = await service.submit(uow, OWNER, _run_vote(FeedbackVerdict.CONFIRM))
    report = GroundingReport(
        claims=[ClaimReport(claim="S40 sells 90 a week.", verdict="unsupported")],
        unsupported=1,
        evidence_count=2,
    )
    judged = await judge_run(service, lambda: uow, ctx, "run_9", "ctx_1", "90", report)  # type: ignore[arg-type,return-value]
    again, _ = await service.submit(uow, OWNER, _run_vote(FeedbackVerdict.REJECT))
    states = await _states(uow)
    assert judged is not None and states[judged] == "pending"
    assert uow.feedback.rows[("acme", judged)].vote is False
    assert (states[thumb.feedback_id], states[again.feedback_id]) == ("replaced", "pending")


# ---------------------------------------------------------------- what an approval decision teaches

ARGS = {"from": "W1", "to": "S12", "units": 4}


def _decision(verdict: FeedbackVerdict, feedback_id: str, **metadata) -> Feedback:
    return Feedback(
        feedback_id=feedback_id,
        tenant_id="acme",
        target_kind=FeedbackTargetKind.TOOL_CALL,
        target_id="call_transfer",
        verdict=verdict,
        source=FeedbackSource.INTERRUPT,
        reviewer="user:lead",
        metadata={"tool": "transfer_stock", "args": ARGS, **metadata},
    )


async def test_an_approval_counts_once_and_only_for_a_call_its_run_recorded() -> None:
    """B6: a key that may act as the agent sent approvals with forged ids (no interrupt, no
    call) and replayed one interrupt under new feedback ids: five of them made a learned rule
    that stopped an administrator's question. An approval now counts only on an interrupt of
    a run that recorded the call it approved, once per interrupt."""
    service, run, uow = _projecting()
    ctx = OWNER.model_copy(update={"agent_id": "ops", "agent_run_id": "run_7"})

    async def decided(verdict, feedback_id, **metadata):
        await service.submit(uow, ctx, _decision(verdict, feedback_id, **metadata))
        return await service.project("acme", feedback_id, attempt=CALL_RECORD_CHECKS)

    forged = await decided(FeedbackVerdict.APPROVE, "fb_forged")
    assert forged is not None and forged.action is ProjectionAction.NONE
    unbacked = await decided(FeedbackVerdict.APPROVE, "fb_unbacked", interrupt_id="int_1")
    assert unbacked is not None and unbacked.action is ProjectionAction.NONE
    assert run.counted == []

    run.calls.append(
        ToolInvocation(
            tenant_id="acme",
            tool_id="tol_t",
            tool_name="transfer_stock",
            run_id="run_7",
            args_hash=stable_hash(ARGS),
        )
    )
    backed = await decided(FeedbackVerdict.APPROVE, "fb_backed", interrupt_id="int_2")
    replayed = await decided(FeedbackVerdict.APPROVE, "fb_replayed", interrupt_id="int_2")
    assert backed is not None and backed.action is ProjectionAction.TOOL_CALL_COUNTED
    assert replayed is not None and replayed.action is ProjectionAction.NONE
    # a rejection only makes the tool ask more: it needs no recorded call
    rejected = await decided(FeedbackVerdict.REJECT, "fb_rejected", interrupt_id="int_3")
    assert rejected is not None and rejected.action is ProjectionAction.TOOL_CALL_COUNTED
    assert [c for _, _, c in run.counted] == ["approvals", "rejections"]


async def test_an_approval_before_its_call_is_recorded_is_looked_at_again() -> None:
    """The decision is sent as the run resumes, before the approved call runs: its projection
    waits, scheduled again, rather than counting nothing for good."""
    service, run, uow = _projecting()
    ctx = OWNER.model_copy(update={"agent_id": "ops", "agent_run_id": "run_7"})
    await service.submit(uow, ctx, _decision(FeedbackVerdict.APPROVE, "fb_a", interrupt_id="i"))
    uow.jobs.clear()
    assert await service.project("acme", "fb_a") is None
    [job] = uow.jobs
    assert job.payload["attempt"] == 1 and job.schedule_in_seconds
    assert uow.feedback.rows[("acme", "fb_a")].projection is None and run.counted == []
