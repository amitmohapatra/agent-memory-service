"""Standing questions and judgements: the two groups that learn off the request path.

A brief is a question the platform keeps answering; feedback is a verdict on something it did,
and a verdict on a memory is what reinforces or retracts it. Eight operations.
"""

from __future__ import annotations

import pytest

from tests.agent.conftest import BOOTSTRAP, sdk
from trellis.memory import BriefSpec, MemoryError

pytestmark = pytest.mark.e2e

FACT = "The payments platform team runs its release review on Thursdays at 15:00."
SPEC = BriefSpec(
    kind="mental_model",
    title="Release cadence",
    question="when does the payments platform team review releases",
    refresh_seconds=3600,
)


async def _harness(app, tenant_id: str = "acme"):
    platform = sdk(app, BOOTSTRAP)
    tenant = await platform.admin.create_tenant(tenant_id.title(), tenant_id=tenant_id)
    admin = sdk(app, tenant.admin_key.token)
    service = await admin.tenant.keys.issue("service", f"{tenant_id}-harness")
    return sdk(app, service.token)


@pytest.mark.covers(
    "briefs.create_brief",
    "briefs.read_brief",
    "briefs.list_briefs",
    "briefs.update_brief",
    "briefs.delete_brief",
)
async def test_an_agent_keeps_a_standing_question_and_reads_it_back(app, running) -> None:
    harness = await _harness(app)
    user = harness.bind(user_id="u1")
    await user.remember(FACT, visibility="USER")

    created = await user.briefs.create(SPEC)
    assert created.brief_id and created.status in ("pending", "ready", "stale")
    assert created.spec.question == SPEC.question and created.spec.use_llm is False

    read = await user.briefs.get(created.brief_id)
    assert read.brief_id == created.brief_id
    if read.output is not None:
        # A read never generates text: without a model the answer is the evidence itself.
        assert read.output.generated is False
        assert all(item.text for item in read.output.sources)

    listed = await user.briefs.list()
    assert created.brief_id in {b.brief_id for b in listed}

    wider = SPEC.model_copy(update={"question": SPEC.question + " and who owns it"})
    updated = await user.briefs.update(created.brief_id, wider)
    assert updated.brief_id == created.brief_id
    assert updated.spec.question == wider.question
    assert (await user.briefs.get(created.brief_id)).spec.question == wider.question

    await user.briefs.delete(created.brief_id)
    with pytest.raises(MemoryError) as gone:
        await user.briefs.get(created.brief_id)
    assert gone.value.status == 404
    assert created.brief_id not in {b.brief_id for b in await user.briefs.list()}


@pytest.mark.covers("feedback.submit_feedback", "feedback.get_feedback", "feedback.list_feedback")
async def test_a_verdict_on_a_memory_is_stored_and_projected(app, running) -> None:
    harness = await _harness(app)
    user = harness.bind(user_id="u1")
    await user.remember(FACT, visibility="USER")
    memory = next(m for m in await user.memories() if FACT in m.content)

    confirmed = await user.feedback.submit(
        "memory", memory.memory_id, "confirm", comment="checked with the team", reviewer="u1"
    )
    assert confirmed.feedback_id and confirmed.verdict == "confirm"
    assert confirmed.target_kind == "memory" and confirmed.target_id == memory.memory_id
    # The identity of the judgement comes from the bound scope, never from the body.
    assert confirmed.user_id == "u1" and confirmed.tenant_id == "acme"
    assert confirmed.source == "human" and confirmed.created_at is not None

    # A retry with the same feedback_id is the stored record, not a second judgement.
    replay = await user.feedback.submit(
        "memory", memory.memory_id, "confirm", feedback_id=confirmed.feedback_id
    )
    assert replay.feedback_id == confirmed.feedback_id

    one = await user.feedback.get(confirmed.feedback_id)
    assert one.feedback_id == confirmed.feedback_id and one.verdict == "confirm"

    corrected = await user.feedback.submit(
        "memory",
        memory.memory_id,
        "correct",
        correction="The review moved to Wednesdays at 15:00.",
    )
    on_target = await user.feedback.list_for("memory", memory.memory_id)
    assert {f.feedback_id for f in on_target} == {confirmed.feedback_id, corrected.feedback_id}
    assert [f.created_at for f in on_target] == sorted(
        (f.created_at for f in on_target), reverse=True
    ), "newest first"
    # A correction is projected onto the memory it judges: that is the point of storing it.
    assert corrected.projection is None or corrected.projection.action in (
        "none",
        "memory_reinforced",
        "memory_retracted",
        "memory_superseded",
    )


@pytest.mark.covers_error(
    "briefs.read_brief",
    "briefs.create_brief",
    "briefs.update_brief",
    "briefs.delete_brief",
    "briefs.list_briefs",
    "feedback.get_feedback",
    "feedback.submit_feedback",
    "feedback.list_feedback",
)
async def test_neither_briefs_nor_verdicts_cross_a_tenant(app, running) -> None:
    acme = await _harness(app, "acme")
    globex = await _harness(app, "globex")
    mine = acme.bind(user_id="u1")
    await mine.remember(FACT, visibility="USER")
    memory = next(m for m in await mine.memories() if FACT in m.content)
    brief = await mine.briefs.create(SPEC)
    verdict = await mine.feedback.submit("memory", memory.memory_id, "confirm")

    theirs = globex.bind(user_id="u1")
    for call, expected in (
        (theirs.briefs.get(brief.brief_id), 404),
        (theirs.briefs.update(brief.brief_id, SPEC), 404),
        (theirs.briefs.delete(brief.brief_id), 404),
        (theirs.feedback.get(verdict.feedback_id), 404),
    ):
        with pytest.raises(MemoryError) as refused:
            await call
        assert refused.value.status == expected, refused.value

    # Nothing of the other tenant's leaks into a listing either. Feedback is listed *by its
    # target*, so the target is resolved in the caller's own tenant first: the answer is
    # "no such memory", never an empty page that would confirm the id exists somewhere.
    assert brief.brief_id not in {b.brief_id for b in await theirs.briefs.list()}
    with pytest.raises(MemoryError) as unknown_target:
        await theirs.feedback.list_for("memory", memory.memory_id)
    assert unknown_target.value.status == 404

    claiming = globex.bind(tenant_id="acme", user_id="u1")
    for call in (
        claiming.briefs.create(SPEC),
        claiming.briefs.list(),
        claiming.feedback.submit("memory", memory.memory_id, "reject"),
        claiming.feedback.list_for("memory", memory.memory_id),
    ):
        with pytest.raises(MemoryError) as crossed:
            await call
        assert crossed.value.status == 403, crossed.value
