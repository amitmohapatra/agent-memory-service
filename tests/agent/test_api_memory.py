"""The turn an agent actually takes: write a memory, ask for context, check an answer
against it, look the memory up, and forget it. The operations - memories, context, recall,
verify, the job behind the write, and the graph - driven the way a framework adapter drives
them, with no request built by hand.
"""

from __future__ import annotations

import pytest

from tests.agent.conftest import BOOTSTRAP, sdk
from trellis.memory import MemoryError

pytestmark = pytest.mark.e2e

FACT = "Priya Raman leads the payments platform team and reports to the CTO."
SECOND = "The payments platform team runs its release review on Thursdays at 15:00."


async def _tenant(app, tenant_id: str = "acme"):
    """A tenant, and a service key for it - the two things a harness is given once."""
    platform = sdk(app, BOOTSTRAP)
    tenant = await platform.admin.create_tenant(tenant_id.title(), tenant_id=tenant_id)
    admin = sdk(app, tenant.admin_key.token)
    service = await admin.tenant.keys.issue("service", f"{tenant_id}-harness")
    return admin, sdk(app, service.token)


@pytest.mark.covers(
    "memory.remember",
    "memory.supersede_memory",
    "memory.list_memories",
    "memory.get_memory",
    "memory.forget_memory",
    "jobs.get_job",
)
async def test_an_agent_writes_reads_and_forgets_one_memory(app, running) -> None:
    _, harness = await _tenant(app)
    agent = harness.bind(user_id="u1").agent("onboarding-bot")

    ack = await agent.remember(FACT, visibility="USER")
    assert ack.memory_id and not ack.deduplicated
    assert ack.job_ids, "the write is acknowledged with the work it queued"

    # The job behind the write is readable by the tenant that dispatched it, and only by it.
    job = await agent.advanced.job(ack.job_ids[0])
    assert job.job_id and job.status in ("PENDING", "RUNNING", "SUCCEEDED")

    # Replay: the same content under the same key is one memory, not two (the SDK derives
    # the idempotency key from scope + content, so a retried turn is safe by default).
    replay = await agent.remember(FACT, visibility="USER")
    assert replay.memory_id == ack.memory_id

    # The statement is the memory, stored before the call returned: nothing to wait for.
    one = await agent.advanced.memories.get(ack.memory_id)
    assert one.content == FACT and one.visibility == "USER"
    stored = next(m for m in await agent.advanced.memories.list() if m.memory_id == ack.memory_id)

    # A correction is a new version; the old one is closed, not deleted.
    updated = await agent.update(ack.memory_id, SECOND, reason="the team moved its review")
    assert updated.supersedes == ack.memory_id
    old = await agent.advanced.memories.get(ack.memory_id)
    assert old.temporal_status == "SUPERSEDED" and old.superseded_by == updated.memory_id
    with pytest.raises(MemoryError) as twice:
        await agent.update(ack.memory_id, "again", reason="stale")
    assert twice.value.status == 409
    stored = await agent.advanced.memories.get(updated.memory_id)

    await agent.forget(stored.memory_id)
    with pytest.raises(MemoryError) as gone:
        await agent.advanced.memories.get(stored.memory_id)
    assert gone.value.status == 404
    assert not any(m.memory_id == stored.memory_id for m in await agent.advanced.memories.list())


@pytest.mark.covers("retrieval.context", "retrieval.recall", "retrieval.verify")
async def test_an_agent_asks_for_context_then_has_its_answer_verified(app, running) -> None:
    _, harness = await _tenant(app)
    agent = harness.bind(user_id="u1").agent("answer-bot")
    await agent.remember(FACT, visibility="USER")
    await agent.remember(SECOND, visibility="USER")

    prompt = await agent.context("who leads the payments platform team", token_budget=2000)
    assert prompt.bundle_id and prompt.token_estimate <= 2000
    assert "Priya" in prompt.rendered, "the bundle is ready to prompt with, not a pile of rows"
    assert "[m1]" in prompt.rendered, "evidence is cited by its per-bundle handle"

    full = await agent.context(
        "who leads the payments platform team", token_budget=2000, format="full"
    )
    assert full.query_type and full.token_estimate <= full.token_budget
    assert any("Priya" in item.text for item in full.memories), full.memories
    assert full.handles and all(h[0] in "mfsd" for h in full.handles)

    ranked = await agent.search("release review", limit=5)
    assert ranked and any("Thursday" in item.text for item in ranked)
    assert len(ranked) <= 5 and all(item.citation and item.kind for item in ranked)

    report = await agent.verify(
        "Priya Raman leads the payments platform team [m1]", bundle_id=prompt.bundle_id
    )
    assert report.evidence_count >= 1
    assert report.supported + report.unsupported + report.contradicted + report.borderline >= 1
    # No key is configured in this suite, so the judge is never consulted: the model-free
    # path is the promise, and verify honours it.
    assert report.judge_consulted == 0 and report.llm_tokens == 0

    with pytest.raises(MemoryError) as unknown:
        await agent.verify("anything", bundle_id="ctx_never_built")
    assert unknown.value.status == 404


@pytest.mark.covers("graph.search_entities", "graph.entity_profile")
async def test_an_agent_traverses_what_the_writes_made_of_the_entities(app, running) -> None:
    _, harness = await _tenant(app)
    agent = harness.bind(user_id="u1").agent("graph-bot")
    await agent.remember(FACT, visibility="USER")

    # Entities the write produced can be searched by name and opened as a profile.
    found = await agent.advanced.graph.entities("priya", limit=5)
    assert found and all(e.canonical_name.startswith("priya") for e in found), found
    profile = await agent.advanced.graph.entity(found[0].entity_id)
    assert profile.entity.entity_id == found[0].entity_id and profile.neighborhood is None
    # with a depth, the graph around it, inside this scope's visibility
    around = await agent.advanced.graph.entity(found[0].entity_id, depth=1, layers=["entity"])
    assert around.neighborhood is not None and around.neighborhood.visited >= 1
    assert all(f.subject and f.layer == "entity" for f in around.neighborhood.facts)
    with pytest.raises(MemoryError) as missing:
        await agent.advanced.graph.entity("ent_never_written")
    assert missing.value.status == 404


@pytest.mark.covers_error(
    "retrieval.recall", "retrieval.verify", "memory.get_memory", "memory.list_memories"
)
async def test_a_foreign_tenant_is_refused_and_a_missing_memory_is_a_problem(app, running) -> None:
    _, acme = await _tenant(app, "acme")
    _, globex = await _tenant(app, "globex")
    mine = acme.bind(user_id="u1")
    await mine.remember(FACT, visibility="USER")
    stored = next(m for m in await mine.advanced.memories.list() if FACT in m.content)

    # Another tenant's key cannot name this tenant, whatever it claims in the header.
    with pytest.raises(MemoryError) as cross:
        await globex.bind(tenant_id="acme", user_id="u1").search("payments")
    assert cross.value.status == 403
    with pytest.raises(MemoryError) as judged:
        await globex.bind(tenant_id="acme", user_id="u1").verify("x", bundle_id="ctx_any")
    assert judged.value.status == 403

    # ... and cannot read the memory by id either: not found, never someone else's content.
    with pytest.raises(MemoryError) as hidden:
        await globex.bind(user_id="u1").advanced.memories.get(stored.memory_id)
    assert hidden.value.status == 404

    with pytest.raises(MemoryError) as listed:
        await globex.bind(tenant_id="acme", user_id="u1").advanced.memories.list()
    assert listed.value.status == 403
