"""Semantic reuse preserves scope/revision barriers and never aliases risky near misses."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import orjson
import pytest

from memory_service.adapters.cache.memory_cache import MemoryCache
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.context.semantic_cache import SemanticBundleCache, guarded_signature
from memory_service.modules.llm.policy import model_call_policy
from tests.unit.test_context_datapath import CTX, _builder, _parts

pytestmark = pytest.mark.unit


def semantic_builder():
    builder = _builder(MemoryCache())
    original = builder.engine.retrieve

    async def retrieve(*args, **kwargs):
        result = await original(*args, **kwargs)
        result.query_embedding = [1.0, 0.0]
        return result

    builder.engine.retrieve = retrieve
    builder.engine.indexer.embedding = SimpleNamespace(
        embed_query=AsyncMock(return_value=[1.0, 0.0])
    )
    return builder


async def test_read_model_policy_separates_exact_and_semantic_cache_entries():
    builder = semantic_builder()
    assisted = await builder.build(CTX, "my timezone?")
    await builder.drain()
    with model_call_policy(False):
        native = await builder.build(CTX, "please my timezone")
        assert not native.cache_hit and native.bundle_id != assisted.bundle_id
        await builder.drain()
        repeated = await builder.build(CTX, "my timezone?")
        assert repeated.cache_hit
    assert builder.engine.calls == 2
    await builder.close()


async def test_semantic_hit_keeps_current_query_identity_without_retrieving_again():
    builder = semantic_builder()
    first = await builder.build(CTX, "my timezone?")
    await builder.drain()
    second = await builder.build(CTX, "please my timezone")
    assert builder.engine.calls == 1
    assert second.cache_hit and second.query == "please my timezone"
    assert second.bundle_id != first.bundle_id and second.memories == first.memories
    assert builder.engine.indexer.embedding.embed_query.await_count == 1
    await builder.close()


@pytest.mark.parametrize(
    "change",
    ["tenant", "user", "agent", "run", "thread", "membership", "revision", "budget", "document"],
)
async def test_semantic_hit_cannot_cross_scope_revision_or_request_constraints(change):
    builder = semantic_builder()
    context = CTX.model_copy(update={"agent_id": "worker", "agent_run_id": "run_first"})
    await builder.build(context, "my timezone?")
    await builder.drain()
    kwargs = {}
    if change in {"tenant", "user", "agent"}:
        context = context.model_copy(update={f"{change}_id": "other"})
    elif change == "run":
        context = context.model_copy(update={"agent_run_id": "run_second"})
    elif change == "thread":
        context = context.model_copy(update={"thread_id": "thread_other"})
    elif change == "membership":
        factory, _ = _parts(builder)
        await factory.revisions.bump(CTX.tenant_id, RevisionKind.MEMBERSHIP)
    elif change == "revision":
        factory, _ = _parts(builder)
        await factory.revisions.bump(CTX.tenant_id, RevisionKind.TENANT)
    elif change == "budget":
        kwargs["token_budget"] = 100
    else:
        kwargs["document_ids"] = ["doc_other"]
    second = await builder.build(context, "please my timezone", **kwargs)
    assert not second.cache_hit and builder.engine.calls == 2
    await builder.close()


@pytest.mark.parametrize(
    "left,right",
    [
        ("Alice called Bob", "Bob called Alice"),
        ("my budget is 20", "my budget is 200"),
        ("my timezone", "my timezone is not UTC"),
        ("my US address", "my us address"),
        ("my manager was Alice", "my manager is Alice"),
    ],
)
def test_high_similarity_is_insufficient_when_meaning_constraints_differ(left, right):
    assert guarded_signature(left) != guarded_signature(right)


@pytest.mark.parametrize(
    "query", ["my current timezone", "today", "`select`", 'the "exact phrase"']
)
def test_time_sensitive_and_quoted_queries_are_not_reused(query):
    assert guarded_signature(query) is None


async def test_similar_surface_with_different_embedding_misses_and_reuses_the_encoded_query():
    builder = semantic_builder()
    await builder.build(CTX, "my timezone?")
    await builder.drain()
    builder.engine.indexer.embedding.embed_query.return_value = [0.0, 1.0]
    original = builder.engine.retrieve
    builder.engine.retrieve = AsyncMock(side_effect=original)
    second = await builder.build(CTX, "please my timezone")
    assert not second.cache_hit and builder.engine.calls == 2
    assert builder.engine.retrieve.call_args.kwargs["query_embedding"] == (
        "please my timezone",
        [0.0, 1.0],
    )
    await builder.close()


@pytest.mark.parametrize("entry", [b"invalid", b"{}", b"[]", b'{"vector":[null],"bundle_key":"x"}'])
async def test_malformed_cache_entries_are_misses(entry):
    cache = SemanticBundleCache(MemoryCache(), ttl_seconds=30)
    assert await cache.lookup(entry, [1.0]) is None


async def test_vector_dimension_mismatch_and_expired_reference_are_misses():
    cache = SemanticBundleCache(MemoryCache(), ttl_seconds=30)
    raw = orjson.dumps({"vector": [1.0, 0.0], "bundle_key": "missing"})
    assert await cache.lookup(raw, [1.0]) is None
    assert await cache.lookup(raw, [1.0, 0.0]) is None


async def test_corrupt_referenced_bundle_retrieves_again_instead_of_failing_the_request():
    builder = semantic_builder()
    first = await builder.build(CTX, "my timezone?")
    await builder.drain()
    await builder.cache.set(builder._cache_key(CTX, first.bundle_id), b"invalid", ttl_seconds=30)
    second = await builder.build(CTX, "please my timezone")
    assert not second.cache_hit and builder.engine.calls == 2
    await builder.close()
