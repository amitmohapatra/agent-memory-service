"""Actor/topic retrieval must remain bounded, scoped, and optional on failure."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from memory_service.config.constants import RETRIEVAL
from memory_service.domain.errors import DependencyUnavailable
from memory_service.domain.script import Script
from memory_service.modules.context.builder import candidate_to_item
from memory_service.modules.rag.indexer import MEMORIES
from memory_service.modules.retrieval.engine import Candidate, QueryVectors
from memory_service.modules.retrieval.memory_queries import plan_memory_queries
from memory_service.ports.search import SearchHit, SearchRecord, VectorName
from tests.unit import test_llm_retrieval as base

parts = base.parts


def memory(subject, identifier="seed"):
    return Candidate(identifier, "memory", "Source evidence", 1, payload={"subject": subject})


def test_plan_separates_actors_from_topic_without_inventing_names():
    candidates = [memory("user:Alice"), memory("user:Bob"), memory("user:Carol")]
    plan = plan_memory_queries("What activities have Alice and Bob both enjoyed?", candidates)
    assert plan is not None
    assert plan.subjects == ("user:Alice", "user:Bob")
    assert plan.topic == "What activities have enjoyed"
    assert plan_memory_queries("What activities has Dave enjoyed?", candidates) is None


@pytest.mark.parametrize("name", ["ALICE's", "Alice\u2019s", "alice"])
def test_actor_matching_handles_case_and_possessives(name):
    plan = plan_memory_queries(f"What are {name} favourite activities?", [memory("user:Alice")])
    assert plan is not None
    assert "alice" not in plan.topic.lower()


def test_partial_names_and_ambiguous_many_actors_are_not_planned():
    assert plan_memory_queries("What did Annabelle do?", [memory("user:Anna")]) is None
    assert (
        plan_memory_queries(
            "What have Alice, Bob and Carol done?",
            [memory("user:Alice"), memory("user:Bob"), memory("user:Carol")],
        )
        is None
    )


@pytest.mark.parametrize(
    ("first", "second", "topic"),
    [
        ("Maya Patel", "Omar Aziz", "incidents resolved"),
        ("Zoë", "Renée", "artifacts restored"),
        ("田中葵", "佐藤蓮", "projects completed"),
        ("Orion-agent", "Vega-agent", "deployments verified"),
    ],
)
def test_actor_renaming_and_topic_changes_preserve_the_search_plan(first, second, topic):
    candidates = [memory(f"user:{first}"), memory(f"user:{second}")]
    plan = plan_memory_queries(f"What {topic} have {first} and {second} both?", candidates)
    assert plan is not None
    assert plan.topic == f"What {topic} have"
    assert set(plan.subjects) == {f"user:{first}", f"user:{second}"}


def test_display_names_are_not_guessed_from_opaque_canonical_ids():
    assert (
        plan_memory_queries(
            "What incidents did Maya and Omar resolve?",
            [memory("user:8f649abc"), memory("user:64fc2cfd")],
        )
        is None
    )


async def test_subject_search_keeps_tenant_visibility_and_current_filters(parts):
    engine = base._engine(parts)
    engine.store.search_hybrid = AsyncMock(return_value=[])
    await engine._hybrid(
        "hobbies",
        base.VISIBILITY,
        kind="memory",
        document_ids=None,
        encoded=QueryVectors(dense={}, sparse=None, script=Script.LATIN),
        subject="user:Alice",
    )
    flt = engine.store.search_hybrid.call_args.kwargs["flt"]
    assert flt.tenant_id == base.VISIBILITY.tenant_id
    assert flt.must_any["visibility_keys"] == list(base.VISIBILITY.keys)
    assert flt.must["current"] is True
    assert flt.must["subject"] == "user:Alice"


async def test_actor_views_share_one_encoding_and_preserve_original_evidence(parts):
    engine = base._engine(parts)
    engine._encode = AsyncMock(
        return_value=QueryVectors(dense={}, sparse=None, script=Script.LATIN)
    )
    engine._hybrid = AsyncMock(
        return_value=[
            SearchHit(
                record_id="bridge",
                score=1,
                retriever="fusion",
                payload={
                    "text": "Bridge",
                    "source_refs": [
                        {
                            "source_type": "message",
                            "source_id": "msg_bridge",
                            "observed_at": "2026-09-26T00:00:00Z",
                        }
                    ],
                },
            )
        ]
    )
    candidates = [memory("user:Alice", "a"), memory("user:Bob", "b")]
    diagnostics = {}
    result = await engine._entity_search(
        "What activities have Alice and Bob both enjoyed?",
        candidates,
        base.VISIBILITY,
        diagnostics,
    )
    assert {c.record_id for c in result} == {"a", "b", "bridge"}
    assert engine._encode.await_count == 1
    assert engine._hybrid.await_count == 2
    assert [c.score for c in candidates] == [1, 1]
    assert diagnostics["memory_entity_search"]["new_candidates"] == 1
    bridge = next(c for c in result if c.record_id == "bridge")
    assert candidate_to_item(bridge).evidence[0].source_id == "msg_bridge"


@pytest.mark.parametrize("failure", [DependencyUnavailable(), TimeoutError()])
async def test_optional_search_failure_returns_original_candidates(parts, failure):
    engine = base._engine(parts)
    engine._encode = AsyncMock(side_effect=failure)
    candidates = [memory("user:Alice")]
    diagnostics = {}
    assert (
        await engine._entity_search(
            "What activities has Alice enjoyed?", candidates, base.VISIBILITY, diagnostics
        )
        is candidates
    )
    assert diagnostics["memory_entity_search"]["fallback"] == type(failure).__name__


async def test_timeout_cancels_slow_search_and_keeps_primary_results(parts):
    engine = base._engine(parts)
    engine.cfg = RETRIEVAL.model_copy(update={"memory_entity_search_timeout_ms": 1})
    engine._encode = AsyncMock(
        return_value=QueryVectors(dense={}, sparse=None, script=Script.LATIN)
    )
    cancelled = asyncio.Event()

    async def slow(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    engine._hybrid = AsyncMock(side_effect=slow)
    candidates = [memory("user:Alice")]
    diagnostics = {}
    assert (
        await engine._entity_search(
            "What activities has Alice enjoyed?", candidates, base.VISIBILITY, diagnostics
        )
        is candidates
    )
    assert cancelled.is_set()
    assert diagnostics["memory_entity_search"]["fallback"] == "TimeoutError"


@pytest.mark.parametrize("limit", [None, 5])
async def test_only_opted_in_aggregate_queries_without_explicit_limits_expand(parts, limit):
    engine = base._engine(parts)
    engine.cfg = RETRIEVAL.model_copy(update={"memory_entity_search": True})
    engine._search_kind = AsyncMock(return_value=[memory("user:Alice")])
    engine._entity_search = AsyncMock(return_value=[memory("user:Alice")])
    await engine.retrieve(
        base.CTX,
        "What activities has Alice enjoyed?",
        kinds=("memory",),
        visibility=base.VISIBILITY,
        limit=limit,
    )
    assert engine._entity_search.await_count == (1 if limit is None else 0)


async def test_real_store_actor_search_cannot_cross_visibility_tenant_or_current_state(parts):
    engine = base._engine(parts)
    indexer, embedding, store = parts
    body = "Alice enjoys hiking and pottery."
    dense = (await embedding.embed_documents([body]))[0]
    sparse = indexer.sparse.encode_documents([body])[0]
    cases = [
        ("allowed", "t", ["tenant:t"], True, "user:Alice"),
        ("private", "t", ["principal:t/user:other"], True, "user:Alice"),
        ("foreign", "other", ["tenant:t"], True, "user:Alice"),
        ("superseded", "t", ["tenant:t"], False, "user:Alice"),
        ("wrong_actor", "t", ["tenant:t"], True, "user:Bob"),
    ]
    await store.upsert(
        [
            SearchRecord(
                record_id=identifier,
                collection=indexer.collection(MEMORIES),
                tenant_id=tenant,
                dense={VectorName.DENSE_ML: dense},
                sparse=sparse,
                payload={
                    "kind": "memory",
                    "text": body,
                    "visibility_keys": keys,
                    "current": current,
                    "subject": subject,
                },
            )
            for identifier, tenant, keys, current, subject in cases
        ]
    )
    hits = await engine._hybrid(
        "hiking", base.VISIBILITY, kind="memory", document_ids=None, subject="user:Alice"
    )
    assert [hit.record_id for hit in hits] == ["allowed"]


async def test_request_cancellation_is_propagated(parts):
    engine = base._engine(parts)
    entered = asyncio.Event()

    async def wait_forever(*args):
        entered.set()
        await asyncio.Event().wait()

    engine._encode = AsyncMock(side_effect=wait_forever)
    task = asyncio.create_task(
        engine._entity_search(
            "What activities has Alice enjoyed?", [memory("user:Alice")], base.VISIBILITY, {}
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_exact_identifier_lookup_is_not_replaced_by_actor_search(parts):
    engine = base._engine(parts)
    engine.cfg = RETRIEVAL.model_copy(update={"memory_entity_search": True})
    engine._exact = AsyncMock(return_value=[memory("user:Alice")])
    engine._entity_search = AsyncMock()
    result = await engine.retrieve(
        base.CTX,
        "What activities has Alice enjoyed? mem_000000000000",
        kinds=("memory",),
        visibility=base.VISIBILITY,
    )
    assert [c.record_id for c in result.candidates] == ["seed"]
    engine._entity_search.assert_not_awaited()


@pytest.mark.parametrize(
    ("enabled", "query", "kinds", "document_ids"),
    [
        (False, "What activities has Alice enjoyed?", ("memory",), None),
        (True, "When did Alice go hiking?", ("memory",), None),
        (True, "What activities has Alice enjoyed?", ("memory", "chunk"), None),
        (True, "What activities has Alice enjoyed?", ("memory",), ["doc_1"]),
    ],
)
async def test_noneligible_requests_do_not_make_extra_searches(
    parts, enabled, query, kinds, document_ids
):
    engine = base._engine(parts)
    engine.cfg = RETRIEVAL.model_copy(update={"memory_entity_search": enabled})

    async def search(*args, kind, **kwargs):
        return [Candidate(kind, kind, "Evidence", 1, payload={"document_id": "doc_1"})]

    engine._search_kind = AsyncMock(side_effect=search)
    engine._entity_search = AsyncMock()
    await engine.retrieve(
        base.CTX, query, kinds=kinds, visibility=base.VISIBILITY, document_ids=document_ids
    )
    engine._entity_search.assert_not_awaited()
