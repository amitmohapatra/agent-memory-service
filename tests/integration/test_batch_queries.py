"""A batch is a fixed number of statements, however many items it holds.

Four loops asked the database once per item: retention forgot a batch of 500 memories one
at a time (a read, a write and the dependents' statement each), a bulk review read each of
up to 500 named votes on its own, the review queue counted each author's verdicts with a
query of its own, and the index job closed the graph facts of each memory in a transaction
of its own. Each is asked here with few items and with more, and must cost the same.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.feedback import Feedback, FeedbackTargetKind, FeedbackVerdict
from memory_service.modules.tenancy.retention import RetentionService
from tests.integration.test_concurrent_forget import TENANT, _memory
from tests.unit.test_graph_entities import entity, relation

pytestmark = pytest.mark.integration

LONG_AGO = datetime.now(UTC) - timedelta(days=400)


async def _old_memories(uow_factory, tag: str, count: int) -> list[str]:
    memories = [
        _memory(f"mem_{tag}_{i:03d}", tag, f"{tag} said thing {i}.", source=f"msg_{tag}_{i}")
        for i in range(count)
    ]
    async with uow_factory() as uow:
        for m in memories:
            m.created_at = LONG_AGO
            await uow.memories.add(m, visibility_keys=[f"user:{TENANT}/{tag}"])
        await uow.commit()
    return [m.memory_id for m in memories]


async def _swept(uow_factory, statements: list[str], tag: str, count: int) -> list[str]:
    """Forget ``count`` old memories in one retention batch; the statements that took."""
    ids = await _old_memories(uow_factory, tag, count)
    statements.clear()
    forgotten = await RetentionService(uow_factory)._sweep_batch(
        TENANT, datetime.now(UTC) - timedelta(days=30)
    )
    ran = list(statements)
    assert forgotten == count
    async with uow_factory() as uow:
        assert [i for i in ids if not await uow.memories.is_forgotten(TENANT, i)] == []
    return ran


async def test_a_retention_batch_costs_the_same_for_three_memories_as_for_thirty(
    uow_factory, statements
) -> None:
    few = await _swept(uow_factory, statements, "few", 3)
    many = await _swept(uow_factory, statements, "many", 30)
    assert len(many) == len(few), "\n".join(many)


async def test_forgetting_many_takes_their_dependents_and_skips_what_is_not_there(
    uow_factory,
) -> None:
    ids = await _old_memories(uow_factory, "dep", 3)
    async with uow_factory() as uow:
        assert await uow.memories.forget(TENANT, ids[0])
        gone = await uow.memories.forget_many(TENANT, [*ids, "mem_missing"])
        assert await uow.memories.forget_many("globex", ids) == []
        assert await uow.memories.forget_many(TENANT, []) == []
        await uow.commit()
    assert sorted(gone) == ids[1:], "an already forgotten memory is not forgotten again"
    async with uow_factory() as uow:
        assert [i for i in ids if not await uow.memories.is_forgotten(TENANT, i)] == []


async def _votes(container, uow_factory, authors: list[str]) -> list[str]:
    """One pending vote per author, on a run each; their feedback ids."""
    service = container.services["feedback"]
    ids = []
    async with uow_factory() as uow:
        for author in authors:
            ctx = MemoryExecutionContext(tenant_id=TENANT, user_id=author)
            stored, _ = await service.submit(
                uow,
                ctx,
                Feedback(
                    tenant_id=TENANT,
                    target_kind=FeedbackTargetKind.RUN,
                    target_id=f"run_{author}",
                    verdict=FeedbackVerdict.REJECT,
                ),
            )
            assert stored.pending
            ids.append(stored.feedback_id)
        await uow.commit()
    return ids


def _reads_of_feedback(statements: list[str]) -> list[str]:
    return [s for s in statements if s.startswith("SELECT") and " FROM feedback" in s]


async def test_the_review_queue_counts_every_authors_verdicts_in_one_query(
    container, uow_factory, statements
) -> None:
    authors = [f"author{i}" for i in range(6)]
    await _votes(container, uow_factory, authors)
    statements.clear()
    async with uow_factory() as uow:
        queue = await container.services["feedback"].pending(uow, TENANT)
    assert {record.user_id: counts["pending"] for record, counts in queue} == dict.fromkeys(
        authors, 1
    )
    assert len(_reads_of_feedback(statements)) == 2, "the page, and the counts of its authors"


async def test_a_bulk_review_reads_the_votes_it_names_in_one_query(
    container, uow_factory, statements
) -> None:
    ids = await _votes(container, uow_factory, [f"voter{i}" for i in range(6)])
    statements.clear()
    async with uow_factory() as uow:
        decided, unchanged, _ = await container.services["feedback"].review_many(
            uow, TENANT, feedback_ids=ids, approve=False, reviewed_by="key:admin"
        )
        await uow.commit()
    assert [f.feedback_id for f in decided] == ids and unchanged == []
    assert len(_reads_of_feedback(statements)) == 1, "\n".join(_reads_of_feedback(statements))


async def test_the_index_job_closes_the_facts_of_gone_memories_in_one_statement(
    container, statements
) -> None:
    store = container.graph_store
    gone = [f"mem_gone_{i}" for i in range(5)]
    await store.upsert_entities([entity("Acme Corp"), entity("Germany", kind="PLACE")])
    await store.upsert_relations(
        [
            relation(f"rel_{i}", "ent_acme_corp", f"fact_{i}", "ent_germany").model_copy(
                update={"memory_id": memory_id}
            )
            for i, memory_id in enumerate(gone)
        ]
    )
    statements.clear()
    await container.services["graph"].enrich_memories("acme", [*gone, "mem_never_there"])
    closing = [s for s in statements if s.startswith("UPDATE graph_relations")]
    assert len(closing) == 1, "\n".join(closing)
    left = await store.entity_relations(
        "acme", "ent_acme_corp", scope_keys=["user:acme/u1"], current=True, limit=10
    )
    assert left == []
    assert await store.supersede_for_memories("acme", gone, at=datetime.now(UTC)) == set()
