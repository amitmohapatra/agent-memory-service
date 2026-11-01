"""Feedback: store a judgement, decide who may see it, and learn from it off the request path.

The projector runs as the ``feedback.project`` job enqueued in the transaction that stored
the record, so a verdict is never lost and never slows the request that carried it.

- memory: an affirming verdict reinforces, a rejection retracts, a correction writes a new
  memory that supersedes the old one;
- run: the run's outcome, unless a higher-ranked source decided it (a person over the judge
  over the run's own status); only then do the memories the verdict cites gain or lose
  confidence;
- tool call: counted on the tool's statistics and on its approval pattern - only a person's
  decision on an interrupt of a run, once per interrupt, and an approval only once the run
  recorded the call it approved (``_project_tool_call``);
- procedure: a rejection takes it out of what is offered.

A verdict that would change what was learned on a person's, an agent's or the grounding
judge's word alone waits in review (ADR 0028, amended by ADR 0039): it is stored ``pending``
and projected only once a tenant admin approves it; a dismissed one is kept for statistics
and changes nothing.
``_needs_review`` says which. A person counts once per target (``modules.feedback.votes``):
their newer vote replaces their earlier pending one (dismissed with ``replaced_by``), a vote
the same as the one of theirs already applied changes nothing, and a flipped one, once
approved, undoes the earlier one's effect before it applies its own.

Every projection that touches a memory re-indexes it (the index carries the confidence and
reinforcement retrieval ranks by) and bumps its revisions, so a cached bundle that showed the
old standing stops being served.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import TemporalStatus
from memory_service.domain.errors import Conflict, NotFound, ScopeDenied, ValidationFailed
from memory_service.domain.evidence import EvidenceRef, EvidenceSource
from memory_service.domain.feedback import (
    AFFIRMING_VERDICTS,
    APPROVAL_COUNTER,
    CORRECTING_VERDICTS,
    OUTCOME_PRECEDENCE,
    REPLACED_REASON,
    Feedback,
    FeedbackProjection,
    FeedbackReview,
    FeedbackSource,
    FeedbackTargetGroup,
    FeedbackTargetKind,
    FeedbackVerdict,
    ProjectionAction,
    ReviewState,
    outcome_teaches,
)
from memory_service.domain.ids import content_hash
from memory_service.domain.learning import (
    APPROVAL_DECISIONS_READ,
    CITED_CONFIDENCE_STEP,
    CITED_MEMORIES_MAX,
    CONFIDENCE_FLOOR,
)
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.revisions import RevisionKind
from memory_service.domain.tools import RunOutcome, stable_hash
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.feedback.votes import (
    ALREADY_COUNTED_REASON,
    classify,
    correction_text,
    is_vote,
    rewrites,
    same_thumb,
    thumbs_up,
    voter_key,
)
from memory_service.modules.jobs.names import TASK_MEMORY_INDEX
from memory_service.modules.memory.native import normalized_hash
from memory_service.modules.memory.replacement import reinstate
from memory_service.modules.memory.revisions import bump_memory_revisions, retract, supersede
from memory_service.modules.memory.service import MemoryService
from memory_service.modules.tools.service import TASK_TOOLS_LEARN
from memory_service.observability.logging import get_logger
from memory_service.ports.tasks import JobSpec, Queue
from memory_service.ports.uow import UnitOfWork, UnitOfWorkFactory
from trellis.memory.approval import arg_shape

log = get_logger(__name__)

TASK_FEEDBACK_PROJECT = "feedback.project"
#: How much one affirming verdict moves a memory's confidence.
REINFORCEMENT_STEP: Final = 0.1
#: How many verdicts one bulk review decides at most (ids sent, or pending on one target).
BULK_REVIEW_MAX: Final = 500
#: How often an approval is looked at again while the call it approved is not recorded yet
#: (the decision is sent as the run resumes, before the approved call has run), and how far
#: apart; after that it counts nothing.
CALL_RECORD_CHECKS: Final = 10
CALL_RECORD_WAIT_SECONDS: Final = 30
#: The record fields the trusted headers decide; a body that disagrees is refused.
IDENTITY_FIELDS: Final = ("tenant_id", "workspace_id", "user_id")
EVIDENCE_SOURCE: Final = EvidenceSource.FEEDBACK


#: kept for callers that import it from here
_correction_text = correction_text
#: Why a vote's projection did nothing because another vote of the same voter counts instead.
STOOD_DOWN: Final = frozenset({REPLACED_REASON, ALREADY_COUNTED_REASON})


class FeedbackService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        authz: AuthorizationService,
        memory: MemoryService,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.uow_factory = uow_factory
        self.authz = authz
        self.memory = memory
        self.clock = clock

    # ------------------------------------------------------------------ writes
    async def submit(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        feedback: Feedback,
        *,
        trusted: bool = False,
        run_vouched: bool = False,
    ) -> tuple[Feedback, bool]:
        """Store the record and schedule its projection, or leave it waiting for review.
        Returns the stored record and whether it was new (a retry with the same
        ``feedback_id`` returns the original; a vote the same as the voter's vote already
        applied on the target returns that vote, and nothing is stored).

        The record only claims who it is from: its identity fields must agree with the
        trusted headers, and a memory target must be readable by the caller. ``trusted`` is
        a verdict applied without review: one the tenant's administrator credential carries.
        ``run_vouched``: the credential may act as the agent the context names, so a
        ``system`` verdict on that agent's own run is its status rather than a client's claim
        (``_needs_review``); the grounding judge passes it too, its verdict being the
        service's own word and never a person's vote."""
        await self._admit(uow, ctx, feedback)
        voter = voter_key(feedback.user_id or ctx.user_id, feedback.agent_id or ctx.agent_id)
        if voter is not None:
            # one voter's clicks on one target apply one after another - whichever agent
            # carried them, a person is one voter - so two at once still leave one vote
            await uow.serialize(
                f"feedback-vote:{ctx.tenant_id}:{feedback.target_kind.value}:"
                f"{feedback.target_id}:{voter}"
            )
        existing = await uow.feedback.get(ctx.tenant_id, feedback.feedback_id)
        if existing is not None:
            return existing, False
        stored = feedback.model_copy(
            update={
                "workspace_id": feedback.workspace_id or ctx.workspace_id,
                "user_id": feedback.user_id or ctx.user_id,
                "agent_id": feedback.agent_id or ctx.agent_id,
                "agent_run_id": feedback.agent_run_id or ctx.agent_run_id,
                # provenance is the request's, never the body's (ADR 0022)
                "trace_id": ctx.trace_id,
                "created_at": self.clock(),
                "projection": None,
                "review": None,
            }
        )
        stored = stored.model_copy(
            update={"vote": classify(stored, trusted=trusted, run_vouched=run_vouched)}
        )
        if await self._needs_review(ctx, stored, trusted=trusted, run_vouched=run_vouched):
            stored = stored.model_copy(update={"review": FeedbackReview(state=ReviewState.PENDING)})
        if stored.pending and voter is not None and is_vote(stored):
            stored = await self._count_once(uow, stored)
        await uow.feedback.add(stored)
        if stored.review is None:  # applied as it arrives (a repeat is stored decided)
            await self._schedule(uow, stored)
        return stored, True

    async def _admit(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, feedback: Feedback
    ) -> None:
        """The record's identity agrees with the trusted headers, and the caller may judge
        its target and read what it cites."""
        for name in IDENTITY_FIELDS:
            claimed, header = getattr(feedback, name), getattr(ctx, name)
            if claimed is not None and header is not None and claimed != header:
                raise ValidationFailed(
                    f"feedback {name} does not match the trusted header", details={"field": name}
                )
        if feedback.target_kind is FeedbackTargetKind.MEMORY:
            await self._authorize_memory_verdict(uow, ctx, feedback)
        for memory_id in feedback.cited_memory_ids(CITED_MEMORIES_MAX):
            # an answer verdict moves the confidence of what it cites: only what the
            # reviewer may read (404 / 403 otherwise)
            await self.memory.get_memory(uow, ctx, memory_id)

    async def _count_once(self, uow: UnitOfWork, vote: Feedback) -> Feedback:
        """One vote per voter per target, across approvals. The same thumb as the voter's
        approved vote changes nothing: it is stored dismissed, ``replaced_by`` naming the vote
        that counts (so the id the client sent reads back, and points at it), and the voter's
        pending change of mind is withdrawn the same way. Otherwise ``vote`` waits and
        replaces their pending ones."""
        counted = await self._applied_vote(uow, vote)
        if counted is None or not same_thumb(counted, vote):
            await self._replace_earlier_votes(uow, vote, replaced_by=vote.feedback_id)
            return vote
        await self._replace_earlier_votes(uow, vote, replaced_by=counted.feedback_id)
        return vote.model_copy(
            update={
                "review": FeedbackReview(
                    state=ReviewState.DISMISSED,
                    reviewed_at=vote.created_at,
                    note=ALREADY_COUNTED_REASON,
                    replaced_by=counted.feedback_id,
                ),
                "projection": FeedbackProjection(
                    action=ProjectionAction.NONE,
                    reason=ALREADY_COUNTED_REASON,
                    projected_at=vote.created_at,
                ),
            }
        )

    @staticmethod
    async def _applied_vote(uow: UnitOfWork, vote: Feedback) -> Feedback | None:
        """The voter's vote on this target that counts now: their newest approved one, not
        counting ``vote`` itself or anything after it. Only reviewed votes take part: what the
        administrator says, applied as it arrives, is the administrator's own word."""
        for found in await uow.feedback.applied_votes(
            vote.tenant_id,
            target_kind=vote.target_kind,
            target_id=vote.target_id,
            user_id=vote.user_id,
            agent_id=vote.agent_id,
        ):
            if (
                found.feedback_id != vote.feedback_id
                and is_vote(found)
                and ((found.created_at, found.feedback_id) < (vote.created_at, vote.feedback_id))
                # a repeat that changed nothing is not the vote that counts
                and not (found.projection and found.projection.reason == ALREADY_COUNTED_REASON)
            ):
                return found
        return None

    async def _replace_earlier_votes(
        self, uow: UnitOfWork, vote: Feedback, *, replaced_by: str
    ) -> None:
        """One vote per voter per target: the voter's earlier pending votes on this target
        are dismissed with ``replaced_by`` (never applied), so repeated clicks count once
        and a changed mind is the vote that waits. Only people's votes: a run's status or a
        judge's word never replaces a thumb, nor the reverse. Not a review state of its own:
        a client that knows pending, approved and dismissed parses every record."""
        earlier = await uow.feedback.pending_on(
            vote.tenant_id,
            target_kind=vote.target_kind,
            target_id=vote.target_id,
            user_id=vote.user_id,
            agent_id=vote.agent_id,
            voter_only=True,
            votes_only=True,
        )
        # the lock a review takes, so a vote is never both approved and replaced
        await uow.serialize(
            *(
                f"feedback:{vote.tenant_id}:{f.feedback_id}"
                for f in sorted(earlier, key=lambda f: f.feedback_id)
            )
        )
        now = vote.created_at
        for found in earlier:
            old = await uow.feedback.get(vote.tenant_id, found.feedback_id)
            if old is None or not old.pending:
                continue  # decided in the meantime
            await uow.feedback.set_review(
                vote.tenant_id,
                old.feedback_id,
                FeedbackReview(
                    state=ReviewState.DISMISSED,
                    reviewed_at=now,
                    note=REPLACED_REASON,
                    replaced_by=replaced_by,
                ),
            )
            await uow.feedback.set_projection(
                vote.tenant_id,
                old.feedback_id,
                FeedbackProjection(
                    action=ProjectionAction.NONE,
                    reason=REPLACED_REASON,
                    projected_at=now,
                ),
            )

    async def _needs_review(
        self, ctx: MemoryExecutionContext, record: Feedback, *, trusted: bool, run_vouched: bool
    ) -> bool:
        """Whether ``record`` waits for a tenant admin before it changes anything: an end
        user's vote does, and so does the grounding judge's verdict (``grounding.judge``: a
        model's word, never trusted). Applied as it arrives: a run reporting its own final
        status, citing nothing (the lowest-ranked word on a run, which never
        overrides a person or the judge); an owner's correction of a memory carrying new text
        (an edit, authorised as one by ``_authorize_memory_verdict``; a thumbs-down is a vote
        and waits); a decision on a tool call (what is learned from it is a suggestion,
        applied only once accepted); and what a tenant admin says, in person or through the
        tenant's administrator credential (``trusted``).

        A run's status is only as good as who reports it: the run id is the caller's own
        (``agent_run_id`` of the context), and the credential must be one that may act as that
        agent (``run_vouched``); a ``system`` verdict from any other caller - a person's own
        token, another run, one citing memories - waits like a vote."""
        if trusted:
            return False
        if record.source is FeedbackSource.JUDGE:
            return True  # whoever's credential carried it, a model's verdict is no admin's
        if (
            run_vouched
            and record.source is FeedbackSource.SYSTEM
            and record.target_kind is FeedbackTargetKind.RUN
            and ctx.agent_run_id is not None
            and record.target_id == ctx.agent_run_id
            # citing memories would move their confidence: a fresh run each time would be a
            # vote nobody reviewed
            and not record.cited_memory_ids(1)
        ):
            return False
        if record.target_kind is FeedbackTargetKind.TOOL_CALL:
            return False
        if record.target_kind is FeedbackTargetKind.MEMORY and rewrites(record):
            return False
        # a tenant admin speaking in person (never an agent acting for one)
        return ctx.is_agent or not await self.authz.is_tenant_admin(ctx)

    @staticmethod
    async def _schedule(uow: UnitOfWork, record: Feedback, *, attempt: int = 0) -> None:
        key = f"feedback:{record.tenant_id}:{record.feedback_id}"
        await uow.enqueue(
            JobSpec(
                task_name=TASK_FEEDBACK_PROJECT,
                queue=Queue.RECONCILE,
                payload={
                    "tenant_id": record.tenant_id,
                    "feedback_id": record.feedback_id,
                    **({"attempt": attempt} if attempt else {}),
                },
                idempotency_key=f"{key}:{attempt}" if attempt else key,
                schedule_in_seconds=CALL_RECORD_WAIT_SECONDS if attempt else None,
                tenant_id=record.tenant_id,
            )
        )

    async def _authorize_memory_verdict(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, feedback: Feedback
    ) -> None:
        """A vote - thumbs up or down - needs what reading needs: it changes nothing until a
        tenant admin approves it (ADR 0039). A correction that rewrites the memory is the
        forget rule: the owner, the user an agent acts for, or a tenant admin in person; a
        reader may not rewrite what a colleague remembered, and an agent edits only its own
        user's memories, whoever that user is."""
        memory = await self.memory.get_memory(uow, ctx, feedback.target_id)  # 404 / 403
        if not rewrites(feedback):
            return
        owner_ok = memory.owner_principal in (ctx.principal_id, f"user:{ctx.user_id}")
        # an agent acting for an admin is still an agent: the admin's word is the admin's own
        admin = not ctx.is_agent and await self.authz.is_tenant_admin(ctx)
        if not owner_ok and not admin:
            raise ScopeDenied(
                f"only the owner (or a tenant admin) can {feedback.verdict.value} a memory",
                details={"principal": ctx.principal_id},
            )

    # ------------------------------------------------------------------ review
    async def review(
        self,
        uow: UnitOfWork,
        tenant_id: str,
        feedback_id: str,
        *,
        approve: bool,
        reviewed_by: str,
        note: str | None = None,
    ) -> Feedback:
        """Approve a pending verdict (it is projected as if it had just arrived) or dismiss
        it (kept, never applied); only once. Who may review is the caller's business: the API
        asks for the tenant's administrator credential."""
        await uow.serialize(f"feedback:{tenant_id}:{feedback_id}")
        record = await uow.feedback.get(tenant_id, feedback_id)
        if record is None:
            raise NotFound(f"feedback {feedback_id} not found")
        if not record.pending:
            raise Conflict(
                f"feedback {feedback_id} is not waiting for review",
                details={"review": record.review.state.value if record.review else None},
            )
        return await self._decide(uow, record, approve=approve, reviewed_by=reviewed_by, note=note)

    async def review_many(
        self,
        uow: UnitOfWork,
        tenant_id: str,
        *,
        feedback_ids: list[str] | None = None,
        target: tuple[FeedbackTargetKind, str] | None = None,
        approve: bool,
        reviewed_by: str,
        note: str | None = None,
    ) -> tuple[list[Feedback], list[Feedback], bool]:
        """Approve or dismiss several verdicts at once: the ``feedback_ids`` named, or every
        verdict pending on one ``target``. Idempotent: a verdict already decided (approved,
        or dismissed by a reviewer or by a newer vote) is left as it is and returned among
        the unchanged, never applied twice. An id this tenant does not have refuses the whole
        call (404) before anything is decided. Returns (decided, unchanged, more): ``more``
        when the target holds more pending verdicts than one call decides."""
        if (feedback_ids is None) == (target is None):
            raise ValidationFailed("send feedback_ids or a target, not both")
        more = False
        if feedback_ids is not None:
            ids = list(dict.fromkeys(feedback_ids))
            if not ids or len(ids) > BULK_REVIEW_MAX:
                raise ValidationFailed(
                    f"send between 1 and {BULK_REVIEW_MAX} feedback ids",
                    details={"count": len(ids)},
                )
        else:
            assert target is not None
            found = await uow.feedback.pending_on(
                tenant_id, target_kind=target[0], target_id=target[1], limit=BULK_REVIEW_MAX + 1
            )
            more = len(found) > BULK_REVIEW_MAX
            ids = [f.feedback_id for f in found[:BULK_REVIEW_MAX]]
        # one lock order for every caller, so two bulk reviews never deadlock
        await uow.serialize(*(f"feedback:{tenant_id}:{i}" for i in sorted(ids)))
        records: list[Feedback] = []
        missing: list[str] = []
        for feedback_id in ids:
            record = await uow.feedback.get(tenant_id, feedback_id)
            if record is None:
                missing.append(feedback_id)
            else:
                records.append(record)
        if missing:
            raise NotFound(
                f"{len(missing)} feedback id(s) not found", details={"feedback_ids": missing}
            )
        decided: list[Feedback] = []
        unchanged: list[Feedback] = []
        for record in records:
            if record.pending:
                decided.append(
                    await self._decide(
                        uow, record, approve=approve, reviewed_by=reviewed_by, note=note
                    )
                )
            else:
                unchanged.append(record)
        return decided, unchanged, more

    async def _decide(
        self,
        uow: UnitOfWork,
        record: Feedback,
        *,
        approve: bool,
        reviewed_by: str,
        note: str | None,
    ) -> Feedback:
        tenant_id, feedback_id = record.tenant_id, record.feedback_id
        now = self.clock()
        review = FeedbackReview(
            state=ReviewState.APPROVED if approve else ReviewState.DISMISSED,
            reviewed_by=reviewed_by,
            reviewed_at=now,
            note=note,
        )
        await uow.feedback.set_review(tenant_id, feedback_id, review)
        projection = None
        if approve:
            await self._schedule(uow, record)
        else:
            projection = FeedbackProjection(
                action=ProjectionAction.NONE, reason="dismissed in review", projected_at=now
            )
            await uow.feedback.set_projection(tenant_id, feedback_id, projection)
        log.info(
            "feedback.reviewed",
            tenant_id=tenant_id,
            feedback_id=feedback_id,
            state=review.state.value,
        )
        return record.model_copy(update={"review": review, "projection": projection})

    async def pending(
        self,
        uow: UnitOfWork,
        tenant_id: str,
        *,
        state: ReviewState = ReviewState.PENDING,
        before: tuple[datetime, str] | None = None,
        limit: int = 100,
    ) -> list[tuple[Feedback, dict[str, int]]]:
        """The review queue (or the verdicts in another review ``state``, for reporting),
        newest first, each verdict with how its author's verdicts fared in review
        (pending / approved / dismissed)."""
        rows = await uow.feedback.list_pending(tenant_id, state=state, before=before, limit=limit)
        history: dict[tuple[str | None, str | None], dict[str, int]] = {}
        for r in rows:
            author = (r.user_id, r.agent_id)
            if author not in history:
                history[author] = await uow.feedback.review_counts(
                    tenant_id, user_id=r.user_id, agent_id=r.agent_id
                )
        return [(r, history[(r.user_id, r.agent_id)]) for r in rows]

    async def pending_by_target(
        self,
        uow: UnitOfWork,
        tenant_id: str,
        *,
        state: ReviewState = ReviewState.PENDING,
        before: tuple[datetime, str, str] | None = None,
        limit: int = 100,
    ) -> list[FeedbackTargetGroup]:
        """The review queue grouped: one row per target with its thumbs up and down, how
        many voters, and when the latest vote came; the most recently voted target first."""
        return await uow.feedback.group_by_target(
            tenant_id, state=state, before=before, limit=limit
        )

    # ------------------------------------------------------------------ reads
    async def get(self, uow: UnitOfWork, ctx: MemoryExecutionContext, feedback_id: str) -> Feedback:
        record = await uow.feedback.get(ctx.tenant_id, feedback_id)
        if record is None or not await self._visible(uow, ctx, record):
            raise NotFound(f"feedback {feedback_id} not found")
        return record

    async def list_for(
        self,
        uow: UnitOfWork,
        ctx: MemoryExecutionContext,
        *,
        target_kind: FeedbackTargetKind,
        target_id: str,
        before: tuple[datetime, str] | None = None,
        limit: int = 100,
    ) -> list[Feedback]:
        """Feedback on one target. A memory target is checked once (as ``GET`` does) and then
        every record on it is visible; on other kinds each record is checked on its own."""
        if target_kind is FeedbackTargetKind.MEMORY:
            await self.memory.get_memory(uow, ctx, target_id)
            return await uow.feedback.list_for(
                ctx.tenant_id,
                target_kind=target_kind,
                target_id=target_id,
                before=before,
                limit=limit,
            )
        admin = await self.authz.is_tenant_admin(ctx)
        out: list[Feedback] = []
        while len(out) < limit:
            rows = await uow.feedback.list_for(
                ctx.tenant_id,
                target_kind=target_kind,
                target_id=target_id,
                before=before,
                limit=limit,
            )
            out.extend(r for r in rows if admin or self._own(ctx, r))
            if len(rows) < limit:
                break
            before = (rows[-1].created_at, rows[-1].feedback_id)
        return out[:limit]

    async def _visible(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, record: Feedback
    ) -> bool:
        if record.target_kind is FeedbackTargetKind.MEMORY:
            try:
                await self.memory.get_memory(uow, ctx, record.target_id)
            except (NotFound, ScopeDenied):
                return False
            return True
        return self._own(ctx, record) or await self.authz.is_tenant_admin(ctx)

    @staticmethod
    def _own(ctx: MemoryExecutionContext, record: Feedback) -> bool:
        """The author, or a teammate in the workspace the feedback was given in."""
        if ctx.user_id is not None and record.user_id == ctx.user_id:
            return True
        return ctx.workspace_id is not None and record.workspace_id == ctx.workspace_id

    # ------------------------------------------------------------------ projector
    async def project(
        self, tenant_id: str, feedback_id: str, *, attempt: int = 0
    ) -> FeedbackProjection | None:
        """The ``feedback.project`` job. Idempotent and serialised: the record and, for a
        memory target, the memory are locked for the transaction, so a re-dispatched job or
        two verdicts on one memory apply one after another and never fork the chain. An
        approval of a call the run has not recorded yet is looked at again later
        (``attempt``, at most ``CALL_RECORD_CHECKS`` times)."""
        async with self.uow_factory() as uow:
            await uow.serialize(f"feedback:{tenant_id}:{feedback_id}")
            record = await uow.feedback.get(tenant_id, feedback_id)
            if record is None:
                log.warning("feedback.missing", tenant_id=tenant_id, feedback_id=feedback_id)
                return None
            if record.projection is not None:
                return record.projection
            if record.pending:
                # waits for a reviewer; approval schedules this job again
                return None
            if (
                attempt < CALL_RECORD_CHECKS
                and _approves(record)
                and not await _backed(uow, record)
            ):
                await self._schedule(uow, record, attempt=attempt + 1)
                await uow.commit()
                return None
            if record.target_kind is FeedbackTargetKind.MEMORY:
                await uow.serialize(f"memory:{tenant_id}:{record.target_id}")
            now = self.clock()
            if record.review is not None and is_vote(record):
                projection = await self._project_vote(uow, record, now=now)
            else:
                projection = await self._project(uow, record, now=now)
            await uow.feedback.set_projection(tenant_id, feedback_id, projection)
            await uow.commit()
        log.info(
            "feedback.projected",
            tenant_id=tenant_id,
            feedback_id=feedback_id,
            action=projection.action.value,
        )
        return projection

    async def _project_vote(
        self, uow: UnitOfWork, record: Feedback, *, now: datetime
    ) -> FeedbackProjection:
        """A person's vote counts once per target: the same thumb as their vote already
        applied changes nothing, and a flipped one undoes that vote's effect before applying
        its own. Serialised per voter and target, so two of their votes apply in order."""
        voter = voter_key(record.user_id, record.agent_id)
        if voter is None:
            return await self._project(uow, record, now=now)
        await uow.serialize(
            f"feedback-effect:{record.tenant_id}:{record.target_kind.value}:"
            f"{record.target_id}:{voter}"
        )
        prior = await self._settle_earlier(uow, record, now=now)
        if prior is not None and same_thumb(prior, record):
            return FeedbackProjection(
                action=ProjectionAction.NONE,
                memory_id=record.target_id
                if record.target_kind is FeedbackTargetKind.MEMORY
                else None,
                run_id=record.target_id if record.target_kind is FeedbackTargetKind.RUN else None,
                reason=ALREADY_COUNTED_REASON,
                projected_at=now,
            )
        reversed_id = None
        if prior is not None:
            await self._reverse(uow, prior, now=now)
            reversed_id = prior.feedback_id
        projection = await self._project(uow, record, now=now)
        return projection.model_copy(update={"reversed": reversed_id})

    async def _settle_earlier(
        self, uow: UnitOfWork, record: Feedback, *, now: datetime
    ) -> Feedback | None:
        """The voter's vote whose effect stands: the newest of their earlier approved votes
        that was actually applied (its projection ran and did not stand down). An earlier one
        approved but not applied yet never will be - the newer vote stands - so it is closed
        here, before its own job can run after this one and count a second time."""
        applied = None
        for found in await uow.feedback.applied_votes(
            record.tenant_id,
            target_kind=record.target_kind,
            target_id=record.target_id,
            user_id=record.user_id,
            agent_id=record.agent_id,
        ):
            if found.feedback_id == record.feedback_id or not is_vote(found):
                continue
            if (found.created_at, found.feedback_id) > (record.created_at, record.feedback_id):
                continue
            if found.projection is None:
                await uow.feedback.set_projection(
                    record.tenant_id,
                    found.feedback_id,
                    FeedbackProjection(
                        action=ProjectionAction.NONE, reason=REPLACED_REASON, projected_at=now
                    ),
                )
            elif applied is None and found.projection.reason not in STOOD_DOWN:
                applied = found
        return applied

    async def _reverse(self, uow: UnitOfWork, prior: Feedback, *, now: datetime) -> None:
        """Undo what ``prior`` - the voter's vote that counted until now - did to its target:
        a reinforcement is taken back, a retraction it made is lifted, the confidence it moved
        on the memories a run cited moves back. Its run label needs no undoing: the newer
        vote labels the run again, at the same rank."""
        projection = prior.projection
        assert projection is not None
        if prior.target_kind is FeedbackTargetKind.RUN and projection.memory_ids:
            affirmed = thumbs_up(prior)
            await self._adjust(
                uow,
                prior.tenant_id,
                projection.memory_ids,
                step=-CITED_CONFIDENCE_STEP if affirmed else CITED_CONFIDENCE_STEP,
                reinforce=-1 if affirmed else 0,
                now=now,
            )
            return
        if prior.target_kind is FeedbackTargetKind.PROCEDURE:
            # a rejection this vote made is lifted: the procedure is offered again
            if projection.action is ProjectionAction.PROCEDURE_REJECTED and (
                await uow.procedures.restore(prior.tenant_id, prior.target_id)
            ):
                await uow.revisions.bump(prior.tenant_id, RevisionKind.TENANT, "")
            return
        if prior.target_kind is not FeedbackTargetKind.MEMORY:
            return
        memory = await uow.memories.get(prior.tenant_id, prior.target_id)
        if memory is None or memory.deleted_at is not None:
            return
        if projection.action is ProjectionAction.MEMORY_REINFORCED:
            if memory.temporal.status is TemporalStatus.CURRENT:
                memory.confidence = max(CONFIDENCE_FLOOR, memory.confidence - REINFORCEMENT_STEP)
                memory.reinforcement_count = max(0, memory.reinforcement_count - 1)
                memory.updated_at = now
                await uow.memories.update(memory)
                await self._reindex(uow, memory)
        elif projection.action is ProjectionAction.MEMORY_RETRACTED:
            await self._lift_retraction(uow, memory, projection.projected_at, now=now)

    async def _lift_retraction(
        self, uow: UnitOfWork, memory: CanonicalMemory, retracted_at: datetime, *, now: datetime
    ) -> None:
        """The memory, and the twins retracted with it, current again - only those this vote
        retracted (their validity closed at its instant), not ones withdrawn since for another
        reason."""
        twins = await uow.memories.twins(memory.tenant_id, memory, current_only=False)
        lifted = []
        for m in (memory, *twins):
            if (
                m.temporal.status is TemporalStatus.RETRACTED
                and m.temporal.valid_to == retracted_at
            ):
                m.temporal = m.temporal.model_copy(
                    update={"status": TemporalStatus.CURRENT, "valid_to": None}
                )
                m.updated_at = now
                await uow.memories.update(m)
                lifted.append(m)
        if lifted:
            await self._reindex(uow, *lifted)

    async def _project(
        self, uow: UnitOfWork, record: Feedback, *, now: datetime
    ) -> FeedbackProjection:
        handler = {
            FeedbackTargetKind.MEMORY: self._project_memory,
            FeedbackTargetKind.RUN: self._project_run,
            FeedbackTargetKind.TOOL_CALL: self._project_tool_call,
            FeedbackTargetKind.PROCEDURE: self._project_procedure,
        }[record.target_kind]
        return await handler(uow, record, now=now)

    async def _project_memory(
        self, uow: UnitOfWork, record: Feedback, *, now: datetime
    ) -> FeedbackProjection:
        memory = await uow.memories.get(record.tenant_id, record.target_id)
        if memory is None or memory.deleted_at is not None:
            return FeedbackProjection(
                action=ProjectionAction.NONE,
                memory_id=record.target_id,
                reason="the memory is gone",
                projected_at=now,
            )
        if memory.temporal.status is not TemporalStatus.CURRENT:
            return FeedbackProjection(
                action=ProjectionAction.NONE,
                memory_id=memory.memory_id,
                reason=f"the memory is {memory.temporal.status.value.lower()}",
                projected_at=now,
            )
        if record.verdict in AFFIRMING_VERDICTS:
            return await self._reinforce(uow, memory, now=now)
        if record.verdict is FeedbackVerdict.REJECT:
            return await self._retract(uow, memory, record, now=now)
        if record.verdict in CORRECTING_VERDICTS:
            return await self._correct(uow, memory, record, now=now)
        raise AssertionError(f"unhandled verdict {record.verdict}")  # pragma: no cover

    async def _project_run(
        self, uow: UnitOfWork, record: Feedback, *, now: datetime
    ) -> FeedbackProjection:
        """A verdict on a run: confirmed or approved succeeded, rejected or corrected did not.

        The run's outcome follows the verdict unless a higher-ranked source already decided
        it (``OUTCOME_PRECEDENCE``: a person over the judge over the run's own status). Only
        a verdict that labels the run moves the memories it cites (its evidence references):
        they gain or lose confidence - unless the verdict is the run's own unverified
        success, which is its outcome only (``outcome_teaches``). An outranked verdict
        changes nothing."""
        affirmed = record.verdict in AFFIRMING_VERDICTS
        teaches = outcome_teaches(record.source, success=affirmed)
        labelled = await self._label_run(uow, record, success=affirmed, learn=teaches, now=now)
        if not labelled:
            return FeedbackProjection(
                action=ProjectionAction.NONE,
                run_id=record.target_id,
                reason="a higher-ranked source already decided the run's outcome",
                projected_at=now,
            )
        adjusted = (
            await self._adjust_cited(uow, record, affirmed=affirmed, now=now) if teaches else []
        )
        return FeedbackProjection(
            action=ProjectionAction.RUN_LABELLED,
            memory_ids=adjusted,
            run_id=record.target_id,
            projected_at=now,
        )

    async def _project_tool_call(
        self, uow: UnitOfWork, record: Feedback, *, now: datetime
    ) -> FeedbackProjection:
        """Approve, reject or edit of a tool call: counted on the tool and on the approval
        pattern of (agent, tool, argument shape). The tool is ``metadata.tool`` and its
        arguments ``metadata.args`` (the edit's replacement arguments when absent). The run's
        outcome is not touched: that is what RUN feedback is for.

        A learned approval rule stops a person being asked, so only a decision that is one
        counts: a reviewer's on an interrupt (``metadata.interrupt_id``) of a run, once per
        interrupt whatever feedback id carries it again, and an approval only when the run
        recorded the call it approved, that tool with those arguments (``_backed``), each
        recorded call backing one counted approval (``_spent``: the interrupt id is the
        client's). A rejection or an edit only makes the tool ask more, so it needs no
        recorded call."""
        tool = str(record.metadata.get("tool") or "").strip()
        if not tool:
            return FeedbackProjection(
                action=ProjectionAction.NONE,
                reason="the verdict names no tool (metadata.tool)",
                projected_at=now,
            )
        interrupt = record.metadata.get("interrupt_id")
        if not (record.agent_run_id and interrupt and record.reviewer):
            return FeedbackProjection(
                action=ProjectionAction.NONE,
                reason="a decision counts only on an interrupt of a run, with its reviewer",
                projected_at=now,
            )
        if await _counted_before(uow, record):
            return FeedbackProjection(
                action=ProjectionAction.NONE,
                reason="a decision on this interrupt was counted already",
                projected_at=now,
            )
        if _approves(record) and not await _backed(uow, record):
            return FeedbackProjection(
                action=ProjectionAction.NONE,
                reason="the run recorded no call of that tool with those arguments",
                projected_at=now,
            )
        if _approves(record) and await _spent(uow, record):
            return FeedbackProjection(
                action=ProjectionAction.NONE,
                reason="each call of that tool with those arguments the run recorded backs a "
                "counted approval already",
                projected_at=now,
            )
        counter = APPROVAL_COUNTER[record.verdict]
        args = record.metadata.get("args")
        if not isinstance(args, dict) and isinstance(record.correction, dict):
            args = record.correction
        await uow.tools.count_verdict(record.tenant_id, tool, counter)
        await uow.tools.count_approval(
            record.tenant_id,
            record.agent_id or "",
            tool,
            arg_shape(args if isinstance(args, dict) else None),
            counter,
        )
        return FeedbackProjection(action=ProjectionAction.TOOL_CALL_COUNTED, projected_at=now)

    async def _project_procedure(
        self, uow: UnitOfWork, record: Feedback, *, now: datetime
    ) -> FeedbackProjection:
        """A rejected procedure is no longer offered (until its steps change)."""
        if record.verdict is not FeedbackVerdict.REJECT:
            return FeedbackProjection(
                action=ProjectionAction.NONE,
                reason=f"a {record.verdict.value} of a procedure is recorded",
                projected_at=now,
            )
        if not await uow.procedures.reject(record.tenant_id, record.target_id):
            return FeedbackProjection(
                action=ProjectionAction.NONE, reason="no such procedure", projected_at=now
            )
        await uow.revisions.bump(record.tenant_id, RevisionKind.TENANT, "")
        return FeedbackProjection(action=ProjectionAction.PROCEDURE_REJECTED, projected_at=now)

    async def _adjust_cited(
        self, uow: UnitOfWork, record: Feedback, *, affirmed: bool, now: datetime
    ) -> list[str]:
        return await self._adjust(
            uow,
            record.tenant_id,
            record.cited_memory_ids(CITED_MEMORIES_MAX),
            step=CITED_CONFIDENCE_STEP if affirmed else -CITED_CONFIDENCE_STEP,
            reinforce=1 if affirmed else 0,
            now=now,
        )

    async def _adjust(
        self,
        uow: UnitOfWork,
        tenant_id: str,
        ids: list[str],
        *,
        step: float,
        reinforce: int,
        now: datetime,
    ) -> list[str]:
        """Move the confidence (and reinforcement) of the current memories among ``ids``."""
        if not ids:
            return []
        await uow.serialize(*(f"memory:{tenant_id}:{i}" for i in sorted(ids)))
        memories = [
            m
            for m in await uow.memories.get_many(tenant_id, ids)
            if m.deleted_at is None and m.temporal.status is TemporalStatus.CURRENT
        ]
        for memory in memories:
            memory.confidence = min(1.0, max(CONFIDENCE_FLOOR, memory.confidence + step))
            memory.reinforcement_count = max(0, memory.reinforcement_count + reinforce)
            memory.updated_at = now
            await uow.memories.update(memory)
        if memories:
            await self._reindex(uow, *memories)
        return sorted(m.memory_id for m in memories)

    async def _label_run(
        self, uow: UnitOfWork, record: Feedback, *, success: bool, learn: bool, now: datetime
    ) -> bool:
        """Set the run's outcome unless a higher-ranked source decided it, and - when it is
        evidence (``learn``) - queue the run's calls to be learned again: a label is what
        turns a trajectory into evidence for a procedure. Serialised per run, so two verdicts
        apply one after the other."""
        run_id = record.target_id
        await uow.serialize(f"run-outcome:{record.tenant_id}:{run_id}")
        existing = await uow.tools.outcome(record.tenant_id, run_id)
        rank = OUTCOME_PRECEDENCE[record.source]
        if existing is not None and OUTCOME_PRECEDENCE[FeedbackSource(existing.source)] > rank:
            return False
        await uow.tools.set_outcome(
            RunOutcome(
                tenant_id=record.tenant_id,
                run_id=run_id,
                success=success,
                note=f"feedback {record.feedback_id}: {record.verdict.value}",
                source=record.source.value,
                recorded_at=now,
            )
        )
        if not learn:
            return True
        await uow.enqueue(
            JobSpec(
                task_name=TASK_TOOLS_LEARN,
                queue=Queue.RECONCILE,
                payload={"tenant_id": record.tenant_id},
                tenant_id=record.tenant_id,
            )
        )
        return True

    async def _reinforce(
        self, uow: UnitOfWork, memory: CanonicalMemory, *, now: datetime
    ) -> FeedbackProjection:
        memory.reinforcement_count += 1
        memory.confidence = min(1.0, memory.confidence + REINFORCEMENT_STEP)
        memory.updated_at = now
        await uow.memories.update(memory)
        # re-indexed, not only invalidated: the index carries confidence and reinforcement,
        # which retrieval ranks by
        await self._reindex(uow, memory)
        return FeedbackProjection(
            action=ProjectionAction.MEMORY_REINFORCED, memory_id=memory.memory_id, projected_at=now
        )

    async def _retract(
        self, uow: UnitOfWork, memory: CanonicalMemory, record: Feedback, *, now: datetime
    ) -> FeedbackProjection:
        # the same statement kept twice (a turn and a rule's reading in its words) is
        # withdrawn once: the twin left behind would still assert it
        twins = await uow.memories.twins(memory.tenant_id, memory)
        for wrong in (memory, *twins):
            await retract(uow, wrong, now=now)
        # what the withdrawn statement replaced in the user's other conversations holds again
        # (ADR 0038)
        restored = await reinstate(
            uow, memory.tenant_id, [memory.memory_id, *(t.memory_id for t in twins)], now=now
        )
        await self._reindex(uow, memory, *twins, *restored)
        return FeedbackProjection(
            action=ProjectionAction.MEMORY_RETRACTED, memory_id=memory.memory_id, projected_at=now
        )

    async def _correct(
        self, uow: UnitOfWork, memory: CanonicalMemory, record: Feedback, *, now: datetime
    ) -> FeedbackProjection:
        content = _correction_text(record.correction)
        if not content:
            return FeedbackProjection(
                action=ProjectionAction.NONE,
                memory_id=memory.memory_id,
                reason="the correction carries no text (a string or {content: ...})",
                projected_at=now,
            )
        evidence = [
            *memory.evidence,
            EvidenceRef(
                source_type=EVIDENCE_SOURCE,
                source_id=record.feedback_id,
                agent_id=record.agent_id,
                agent_run_id=record.agent_run_id,
                observed_at=now,
            ),
        ]
        corrected = memory.model_copy(
            update={
                "memory_id": CanonicalMemory.model_fields["memory_id"].get_default(
                    call_default_factory=True
                ),
                "content": content,
                "normalized_hash": normalized_hash(content),
                "confidence": max(memory.confidence, 0.9),
                "reinforcement_count": 1,
                "access_count": 0,
                "last_accessed_at": None,
                "evidence": evidence,
                "temporal": memory.temporal.model_copy(
                    update={"observed_at": now, "valid_from": now, "valid_to": None}
                ),
                # a fresh row: no inherited TTL, derived slot, index state or admission trail
                "system_metadata": {
                    "provider": memory.system_metadata.get("provider", "native"),
                    "corrected_by": record.feedback_id,
                    "reviewer": record.reviewer,
                },
                "created_at": now,
                "updated_at": now,
                "revision": 1,
            }
        )
        keys = list(memory.system_metadata.get("visibility_keys", []))
        # the same statement kept twice is corrected once: its twin closes at the correction
        twins = await uow.memories.twins(memory.tenant_id, memory)
        # link first: ``add`` copies the temporal state into the row as it is at that moment
        await supersede(uow, memory, corrected, now=now)
        for twin in twins:
            await supersede(uow, twin, corrected.model_copy(), now=now)
        await uow.memories.add(corrected, visibility_keys=keys)
        await self._reindex(uow, memory, corrected, *twins)
        return FeedbackProjection(
            action=ProjectionAction.MEMORY_SUPERSEDED,
            memory_id=memory.memory_id,
            superseded_by=corrected.memory_id,
            projected_at=now,
        )

    @staticmethod
    async def _reindex(uow: UnitOfWork, *memories: CanonicalMemory) -> None:
        ids = sorted(m.memory_id for m in memories)
        await uow.enqueue(
            JobSpec(
                task_name=TASK_MEMORY_INDEX,
                queue=Queue.EMBEDDING,
                payload={"tenant_id": memories[0].tenant_id, "memory_ids": ids},
                # the ids' digest: a verdict may cite any number of memories, and the key
                # must fit the outbox's column (300 characters: nine ids did not)
                idempotency_key=f"memidx:feedback:{content_hash('|'.join(ids))}",
                tenant_id=memories[0].tenant_id,
            )
        )
        await bump_memory_revisions(uow, memories)


def _approves(record: Feedback) -> bool:
    """Whether ``record`` approves a tool call: the decision a learned rule relaxes on."""
    return (
        record.target_kind is FeedbackTargetKind.TOOL_CALL
        and APPROVAL_COUNTER[record.verdict] == "approvals"
    )


async def _recorded(uow: UnitOfWork, record: Feedback) -> int:
    """How many calls of the decided tool with those arguments the decision's agent recorded
    in its run. A run id is the client's: two agents of one tenant may send the same one, so
    a call another agent recorded under it backs no decision of this one."""
    run_id, tool = record.agent_run_id or "", record.metadata.get("tool")
    wanted = stable_hash(record.metadata.get("args"))
    calls = await uow.tools.invocations_for_run(record.tenant_id, run_id)
    return sum(
        c.agent_id == record.agent_id and c.tool_name == tool and c.args_hash == wanted
        for c in calls
    )


async def _backed(uow: UnitOfWork, record: Feedback) -> bool:
    """Whether the decision's agent recorded, in its run, the call it decided: its tool,
    its arguments."""
    if not record.agent_run_id or not isinstance(record.metadata.get("args"), dict):
        return False
    return await _recorded(uow, record) > 0


async def _spent(uow: UnitOfWork, record: Feedback) -> bool:
    """Whether every call of the approved tool with those arguments that the agent recorded
    in the run backs a counted approval already. The interrupt id is the client's: without
    this, one recorded call would back any number of approvals, each under a new id.
    Serialised per run, so two approvals of one call are counted one after the other."""
    run_id, tool = record.agent_run_id or "", str(record.metadata.get("tool"))
    await uow.serialize(f"tool-approvals:{record.tenant_id}:{run_id}")
    wanted = stable_hash(record.metadata.get("args"))
    recorded = await _recorded(uow, record)
    decisions = await uow.feedback.counted_decisions(
        record.tenant_id,
        agent_id=record.agent_id or "",
        tool=tool,
        limit=APPROVAL_DECISIONS_READ,
    )
    counted = sum(
        d.agent_run_id == run_id and _approves(d) and stable_hash(d.metadata.get("args")) == wanted
        for d in decisions
    )
    return counted >= recorded


async def _counted_before(uow: UnitOfWork, record: Feedback) -> bool:
    """Whether another decision of the same agent on the same interrupt of the same run
    was counted. The run and interrupt ids are the client's: another agent's decision under
    the same ids neither stands for this one nor stops it."""
    interrupt = record.metadata.get("interrupt_id")
    return any(
        other.feedback_id != record.feedback_id
        and other.agent_id == record.agent_id
        and other.agent_run_id == record.agent_run_id
        and other.metadata.get("interrupt_id") == interrupt
        and other.projection is not None
        and other.projection.action is ProjectionAction.TOOL_CALL_COUNTED
        for other in await uow.feedback.list_for(
            record.tenant_id, target_kind=record.target_kind, target_id=record.target_id
        )
    )
