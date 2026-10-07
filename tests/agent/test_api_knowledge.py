"""Standing questions and judgements: the two things that learn off the request path.

A profile block with a standing question is kept answering it from what its scope may read;
feedback is a verdict on something the platform did - a memory it reinforces or retracts, a
run whose outcome follows the highest-ranked source that judged it. A vote changes nothing
until the tenant's administrator approves it (ADR 0028).
"""

from __future__ import annotations

import pytest

from tests.agent.conftest import BOOTSTRAP, sdk
from trellis.memory import MemoryError

pytestmark = pytest.mark.e2e

FACT = "The payments platform team runs its release review on Thursdays at 15:00."
QUESTION = "when does the payments platform team review releases"


#: the tenant administrator client of each harness, by the harness client
ADMINS: dict[int, object] = {}


async def _harness(app, tenant_id: str = "acme"):
    platform = sdk(app, BOOTSTRAP)
    tenant = await platform.admin.create_tenant(tenant_id.title(), tenant_id=tenant_id)
    admin = sdk(app, tenant.admin_key.token)
    service = await admin.tenant.keys.issue("service", f"{tenant_id}-harness")
    harness = sdk(app, service.token)
    ADMINS[id(harness)] = admin.bind(tenant_id=tenant_id)
    return harness


def _reviewer(harness):  # type: ignore[no-untyped-def]
    """The tenant's administrator, who reviews the votes a harness's users cast."""
    return ADMINS[id(harness)]


@pytest.mark.covers("profile.edit_profile_block", "profile.get_profile")
async def test_a_standing_question_is_answered_into_its_block(app, running) -> None:
    harness = await _harness(app)
    user = harness.bind(user_id="u1")
    await user.remember(FACT, visibility="USER")

    block = await user.profile.edit("user.releases", source_query=QUESTION)
    assert block.source_query == QUESTION and block.block == "user.releases"
    # the job answered it (inline here): without a model the answer is the evidence itself
    answered = next(b for b in await user.profile() if b.block == "user.releases")
    assert "Thursdays at 15:00" in answered.text
    assert answered.version > block.version

    # an edit of the text keeps the question; clearing the question keeps the text
    edited = await user.profile.edit("user.releases", "Thursdays, 15:00 (release review)")
    assert edited.source_query == QUESTION and edited.text.startswith("Thursdays")
    cleared = await user.profile.edit("user.releases", source_query=None)
    assert cleared.source_query is None and cleared.text == edited.text


@pytest.mark.covers("feedback.submit_feedback", "feedback.get_feedback", "feedback.list_feedback")
async def test_a_verdict_on_a_memory_is_stored_and_projected(app, running) -> None:
    harness = await _harness(app)
    user = harness.bind(user_id="u1")
    await user.remember(FACT, visibility="USER")
    memory = next(m for m in (await user.advanced.memories.page()).items if FACT in m.content)

    confirmed = await user.feedback(
        "memory", memory.memory_id, "confirm", comment="checked with the team", reviewer="u1"
    )
    assert confirmed.feedback_id and confirmed.verdict == "confirm"
    assert confirmed.target_kind == "memory" and confirmed.target_id == memory.memory_id
    # The identity of the judgement comes from the bound scope, never from the body.
    assert confirmed.user_id == "u1" and confirmed.tenant_id == "acme"
    assert confirmed.source == "human" and confirmed.created_at is not None

    # A retry with the same feedback_id is the stored record, not a second judgement.
    replay = await user.feedback(
        "memory", memory.memory_id, "confirm", feedback_id=confirmed.feedback_id
    )
    assert replay.feedback_id == confirmed.feedback_id

    # a vote waits for the tenant's administrator, and changes nothing until approved
    assert confirmed.review is not None and confirmed.review.state == "pending"
    one = await user.feedback.get(confirmed.feedback_id)
    assert one.feedback_id == confirmed.feedback_id and one.verdict == "confirm"
    assert one.projection is None
    approved = await _reviewer(harness).feedback.approve(confirmed.feedback_id)
    assert approved.review is not None and approved.review.state == "approved"
    one = await user.feedback.get(confirmed.feedback_id)
    assert one.projection is not None and one.projection.action == "memory_reinforced"

    corrected = await user.feedback(
        "memory",
        memory.memory_id,
        "correct",
        correction="The review moved to Wednesdays at 15:00.",
    )
    on_target = (await user.feedback.page_for("memory", memory.memory_id)).items
    assert {f.feedback_id for f in on_target} == {confirmed.feedback_id, corrected.feedback_id}
    assert [f.created_at for f in on_target] == sorted(
        (f.created_at for f in on_target), reverse=True
    ), "newest first"
    projected = await user.feedback.get(corrected.feedback_id)
    assert projected.projection is not None
    assert projected.projection.action == "memory_superseded"


@pytest.mark.covers("feedback.submit_feedback", "feedback.get_feedback")
async def test_a_run_s_outcome_follows_the_highest_ranked_verdict(app, running) -> None:
    """The harness reports how the run ended (system); the judge grounds its answer; a person
    has the last word. A lower-ranked verdict never overrides a higher one."""
    harness = await _harness(app)
    run = harness.bind(user_id="u1").agent("buyer", agent_run_id="run_outcome_1")

    system = await run.feedback("run", "run_outcome_1", "confirm", source="system")
    assert (await run.feedback.get(system.feedback_id)).projection.action == "run_labelled"  # type: ignore[union-attr]
    # a client that says "judge" is not the service's judge: its word waits like a vote
    judge = await run.feedback("run", "run_outcome_1", "reject", source="judge", score=0.2)
    assert judge.review is not None and judge.review.state == "pending"
    await _reviewer(harness).feedback.approve(judge.feedback_id)
    assert (await run.feedback.get(judge.feedback_id)).projection.action == "run_labelled"  # type: ignore[union-attr]
    human = await run.feedback("run", "run_outcome_1", "confirm", source="human")
    await _reviewer(harness).feedback.approve(human.feedback_id)
    assert (await run.feedback.get(human.feedback_id)).projection.action == "run_labelled"  # type: ignore[union-attr]
    late = await run.feedback("run", "run_outcome_1", "reject", source="system")
    projection = (await run.feedback.get(late.feedback_id)).projection
    assert projection is not None and projection.action == "none", (
        "the run's own status does not override a person"
    )


@pytest.mark.covers(
    "feedback.list_feedback",
    "feedback.pending_feedback",
    "feedback.approve_feedback",
    "feedback.dismiss_feedback",
)
@pytest.mark.covers_error(
    "feedback.list_feedback",
    "feedback.pending_feedback",
    "feedback.approve_feedback",
    "feedback.dismiss_feedback",
)
async def test_votes_wait_in_a_queue_only_the_tenant_administrator_reviews(app, running) -> None:
    harness = await _harness(app)
    admin = _reviewer(harness)
    user = harness.bind(user_id="u1")
    await user.remember(FACT, visibility="USER")
    memory = next(m for m in (await user.advanced.memories.page()).items if FACT in m.content)
    up = await user.feedback("memory", memory.memory_id, "confirm")
    down = await user.feedback("run", "run_voted", "reject", comment="wrong answer")

    queue = await admin.feedback.pending()
    assert [f.feedback_id for f in queue.items] == [down.feedback_id, up.feedback_id]
    assert queue.items[0].author_record == {"pending": 2, "approved": 0, "dismissed": 0}
    # the old route is a deprecated alias of GET /v1/feedback?review=pending: the same queue
    legacy = await admin._request("GET", "/v1/feedback/pending")
    assert [f["feedback_id"] for f in legacy["feedback"]] == [down.feedback_id, up.feedback_id]
    with pytest.raises(MemoryError) as refused_alias:
        await user._request("GET", "/v1/feedback/pending")
    assert refused_alias.value.status == 403

    # the service key that cast the votes may not review them
    for review in (user.feedback.pending(), user.feedback.approve(up.feedback_id)):
        with pytest.raises(MemoryError) as refused:
            await review
        assert refused.value.status == 403

    await admin.feedback.approve(up.feedback_id, note="matches the release calendar")
    dismissed = await admin.feedback.dismiss(down.feedback_id, note="the answer was right")
    assert dismissed.review is not None and dismissed.review.state == "dismissed"
    assert dismissed.review.note == "the answer was right"
    assert dismissed.projection is not None and dismissed.projection.action == "none"
    assert (await admin.feedback.pending()).items == []

    # decided once: a second decision is refused, an unknown verdict is not found
    with pytest.raises(MemoryError) as twice:
        await admin.feedback.approve(down.feedback_id)
    assert twice.value.status == 409
    with pytest.raises(MemoryError) as unknown:
        await admin.feedback.dismiss("fb_missing")
    assert unknown.value.status == 404

    # what the tenant's administrator says needs no review
    own = await admin.feedback("run", "run_admin", "confirm")
    assert own.review is None


@pytest.mark.covers_error(
    "feedback.get_feedback",
    "feedback.submit_feedback",
    "feedback.list_feedback",
    "profile.edit_profile_block",
)
async def test_verdicts_do_not_cross_a_tenant(app, running) -> None:
    acme = await _harness(app, "acme")
    globex = await _harness(app, "globex")
    mine = acme.bind(user_id="u1")
    await mine.remember(FACT, visibility="USER")
    memory = next(m for m in (await mine.advanced.memories.page()).items if FACT in m.content)
    verdict = await mine.feedback("memory", memory.memory_id, "confirm")

    theirs = globex.bind(user_id="u1")
    with pytest.raises(MemoryError) as refused:
        await theirs.feedback.get(verdict.feedback_id)
    assert refused.value.status == 404

    # Feedback is listed *by its target*, so the target is resolved in the caller's own
    # tenant first: the answer is "no such memory", never an empty page that would confirm
    # the id exists somewhere.
    with pytest.raises(MemoryError) as unknown_target:
        await theirs.feedback.page_for("memory", memory.memory_id)
    assert unknown_target.value.status == 404

    claiming = globex.bind(tenant_id="acme", user_id="u1")
    for call in (
        claiming.feedback("memory", memory.memory_id, "reject"),
        claiming.feedback.page_for("memory", memory.memory_id),
        claiming.profile.edit("user", "x"),
    ):
        with pytest.raises(MemoryError) as crossed:
            await call
        assert crossed.value.status == 403, crossed.value
