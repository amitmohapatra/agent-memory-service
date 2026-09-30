"""The turn an agent actually takes: write what happened, ask for context, check an answer
against it, look the memory up, and forget it. Nine operations - observations, memories,
context, recall, verify, the job behind the write, and the graph - driven the way a framework
adapter drives them, with no request built by hand.
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
    "memory.submit_observation",
    "memory.list_memories",
    "memory.get_memory",
    "memory.forget_memory",
    "jobs.get_job",
)
async def test_an_agent_writes_reads_and_forgets_one_memory(app, running) -> None:
    _, harness = await _tenant(app)
    agent = harness.bind(user_id="u1").agent("onboarding-bot")

    ack = await agent.remember(FACT, visibility="USER")
    assert ack.observation_id
    assert ack.job_ids, "the write is acknowledged with the work it queued"

    # The job behind the write is readable by the tenant that dispatched it, and only by it.
    job = await agent.job(ack.job_ids[0])
    assert job.job_id and job.status in ("PENDING", "RUNNING", "SUCCEEDED")

    # Replay: the same content under the same key is one observation, not two (the SDK derives
    # the idempotency key from scope + content, so a retried turn is safe by default).
    replay = await agent.remember(FACT, visibility="USER")
    assert replay.observation_id == ack.observation_id

    inventory = await agent.memories()
    assert any(FACT in m.content for m in inventory), inventory
    stored = next(m for m in inventory if FACT in m.content)

    one = await agent.get_memory(stored.memory_id)
    assert one.memory_id == stored.memory_id and one.content == stored.content
    assert one.visibility == "USER"

    await agent.forget(stored.memory_id)
    with pytest.raises(MemoryError) as gone:
        await agent.get_memory(stored.memory_id)
    assert gone.value.status == 404
    assert not any(m.memory_id == stored.memory_id for m in await agent.memories())


@pytest.mark.covers("retrieval.context", "retrieval.recall", "retrieval.verify")
async def test_an_agent_asks_for_context_then_has_its_answer_verified(app, running) -> None:
    _, harness = await _tenant(app)
    agent = harness.bind(user_id="u1").agent("answer-bot")
    await agent.remember(FACT, visibility="USER")
    await agent.remember(SECOND, visibility="USER")

    bundle = await agent.context("who leads the payments platform team", token_budget=2000)
    assert bundle.query_type and bundle.token_estimate <= bundle.token_budget
    assert bundle.rendered, "the bundle is ready to prompt with, not a pile of rows"
    assert any("Priya" in item.text for item in bundle.memories), bundle.memories

    ranked = await agent.recall("release review", limit=5)
    assert ranked and any("Thursday" in item.text for item in ranked)
    assert len(ranked) <= 5

    report = await agent.verify("Priya Raman leads the payments platform team [1]", bundle=bundle)
    assert report.evidence_count >= 1
    assert report.supported + report.unsupported + report.contradicted + report.borderline >= 1
    # No key is configured in this suite, so the judge is never consulted: the model-free
    # path is the promise, and verify honours it.
    assert report.judge_consulted == 0 and report.llm_tokens == 0


@pytest.mark.covers("graph.graph_query", "graph.search_entities", "graph.entity_profile")
async def test_an_agent_traverses_what_the_writes_made_of_the_entities(app, running) -> None:
    _, harness = await _tenant(app)
    agent = harness.bind(user_id="u1").agent("graph-bot")
    await agent.remember(FACT, visibility="USER")

    answer = await agent.graph.query("who does Priya Raman report to", hops=1, layers=["entity"])

    # The graph is built off the write path; what matters to the contract is that a query
    # answers in the bundle's shape and stays inside this scope's visibility.
    assert answer.visited >= 0
    assert all(fact.subject and fact.layer == "entity" for fact in answer.facts)

    # Entities the write produced can be searched by name and opened as a profile.
    found = await agent.graph.entities("priya", limit=5)
    assert found and all(e.canonical_name.startswith("priya") for e in found), found
    profile = await agent.graph.entity(found[0].entity_id)
    assert profile.entity.entity_id == found[0].entity_id
    with pytest.raises(MemoryError) as missing:
        await agent.graph.entity("ent_never_written")
    assert missing.value.status == 404


@pytest.mark.covers_error("retrieval.recall", "memory.get_memory", "memory.list_memories")
async def test_a_foreign_tenant_is_refused_and_a_missing_memory_is_a_problem(app, running) -> None:
    _, acme = await _tenant(app, "acme")
    _, globex = await _tenant(app, "globex")
    mine = acme.bind(user_id="u1")
    await mine.remember(FACT, visibility="USER")
    stored = next(m for m in await mine.memories() if FACT in m.content)

    # Another tenant's key cannot name this tenant, whatever it claims in the header.
    with pytest.raises(MemoryError) as cross:
        await globex.bind(tenant_id="acme", user_id="u1").recall("payments")
    assert cross.value.status == 403

    # ... and cannot read the memory by id either: not found, never someone else's content.
    with pytest.raises(MemoryError) as hidden:
        await globex.bind(user_id="u1").get_memory(stored.memory_id)
    assert hidden.value.status == 404

    with pytest.raises(MemoryError) as listed:
        await globex.bind(tenant_id="acme", user_id="u1").memories()
    assert listed.value.status == 403
