"""Semantic reuse preserves scope/revision barriers and never aliases risky near misses."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import orjson
import pytest

from memory_service.adapters.cache.memory_cache import MemoryCache
from memory_service.domain.revisions import RevisionKind
from memory_service.domain.tools import SkillView, ToolCandidate, ToolHints
from memory_service.modules.context.sections import Pinned, ToolsRequest
from memory_service.modules.context.semantic_cache import SemanticBundleCache, guarded_signature
from memory_service.modules.llm.policy import model_call_policy
from memory_service.modules.tools.hints import asks_change
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
    builder.engine.indexer.spaces = SimpleNamespace(
        embed_primary_query=AsyncMock(return_value=[1.0, 0.0])
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
    assert builder.engine.indexer.spaces.embed_primary_query.await_count == 1
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
    builder.engine.indexer.spaces.embed_primary_query.return_value = [0.0, 1.0]
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


# --------------------------------------------------------------------------- with tools
TOOLBOX = ["erp-create_po", "erp-get_stock", "mail-send", "calendar-list", "crm-find"]


class _Sections:
    """The pinned sections a toolbox brings: the skill learned for the task and the tool
    hints, each found for the words of the question (its guarded signature), as the real
    ``ContextSections`` finds them for its words."""

    def __init__(self) -> None:
        self.gathered = 0
        self.hinted = 0

    async def gather(self, ctx, query, visibility, *, procedures, vectors=None):
        self.gathered += 1
        found = [SkillView(id="proc_1", name=guarded_signature(query) or query)]
        return Pinned(procedures=found if procedures else [])

    async def tools(self, ctx, query, request, visibility, *, memories, profile, vectors):
        self.hinted += 1
        first = sorted(request.available)[0]
        # as the real hints: a tool that changes something never leads a question
        return ToolHints(
            candidates=[ToolCandidate(name=first, score=1.0)],
            next=first if asks_change(query) else None,
        )


def tooled_builder():
    builder = semantic_builder()
    builder.sections = _Sections()
    return builder


def _tools(names=TOOLBOX, **kwargs):
    return ToolsRequest(available=list(names), **kwargs)


async def test_a_request_with_tools_reuses_a_bundle_built_for_the_same_toolbox():
    builder = tooled_builder()
    first = await builder.build(CTX, "my timezone?", tools=_tools())
    await builder.drain()
    second = await builder.build(CTX, "please my timezone?", tools=_tools(reversed(TOOLBOX)))
    assert second.cache_hit and second.query == "please my timezone?"
    assert builder.engine.calls == 1 and builder.sections.gathered == 1
    assert second.bundle_id != first.bundle_id
    assert (second.tools, second.procedures) == (first.tools, first.procedures)
    await builder.close()


async def test_a_semantic_hit_with_tools_is_the_bundle_a_fresh_build_makes():
    """What a hit answers is what building the question afresh answers: the skills and the
    hints were found for the same words, the memories retrieved for them."""
    cached = tooled_builder()
    await cached.build(CTX, "my timezone?", tools=_tools())
    await cached.drain()
    hit = await cached.build(CTX, "please my timezone?", tools=_tools())
    fresh = await tooled_builder().build(CTX, "please my timezone?", tools=_tools())
    assert hit.cache_hit and not fresh.cache_hit

    def content(bundle):  # what the caller reads (evidence timestamps are the build's own)
        memories = [(m.item_id, m.text, m.citation) for m in bundle.memories]
        return (
            bundle.query,
            memories,
            bundle.tools,
            bundle.procedures,
            bundle.profile,
            bundle.evidence.status,
            bundle.token_estimate,
        )

    assert content(hit) == content(fresh)
    await cached.close()


async def test_a_question_and_a_request_to_act_are_different_bundles_with_tools():
    """The hints' next step depends on whether the task is a question, a mark the guarded
    words drop: with tools, a question's bundle is never served to a request to act, nor the
    reverse; without tools the two still share one."""
    builder = tooled_builder()
    await builder.build(CTX, "my timezone?", tools=_tools())
    await builder.drain()
    act = await builder.build(CTX, "please my timezone!", tools=_tools())
    fresh = await tooled_builder().build(CTX, "please my timezone!", tools=_tools())
    assert not act.cache_hit and builder.engine.calls == 2
    assert act.tools is not None and act.tools.next == sorted(TOOLBOX)[0]
    assert act.tools == fresh.tools
    await builder.drain()
    again = await builder.build(CTX, "my timezone.", tools=_tools())
    assert again.cache_hit and again.tools == act.tools
    await builder.close()


@pytest.mark.parametrize(
    "other",
    [
        _tools(TOOLBOX[:-1]),  # one tool fewer
        _tools([*TOOLBOX, "erp-cancel_po"]),  # one more
        _tools(k=3),  # fewer hints asked for
        _tools(hints=False),  # the skills alone
        None,  # no tools at all
    ],
)
async def test_a_semantic_hit_needs_the_same_toolbox(other):
    builder = tooled_builder()
    await builder.build(CTX, "my timezone?", tools=_tools())
    await builder.drain()
    second = await builder.build(CTX, "please my timezone?", tools=other)
    assert not second.cache_hit and builder.engine.calls == 2
    await builder.close()


def test_the_tools_digest_is_the_toolbox_not_its_order_or_repeats():
    digest = _tools().fingerprint()
    assert _tools(reversed(TOOLBOX)).fingerprint() == digest
    assert _tools([*TOOLBOX, TOOLBOX[0]]).fingerprint() == digest
    assert _tools(TOOLBOX[:-1]).fingerprint() != digest


@pytest.mark.parametrize("mark", ["?", "\uff1f", "\u061f"])
async def test_a_question_in_any_script_is_never_served_a_request_to_acts_bundle(mark):
    """The 354339d fix for every question mark the hints read (Latin, full-width, Arabic):
    with tools, a request to act built first is not served to the question, and the
    question's bundle, cached, is served to the same question asked again."""
    assert not asks_change(f"my timezone{mark}") and asks_change("please my timezone!")
    builder = tooled_builder()
    act = await builder.build(CTX, "please my timezone!", tools=_tools())
    await builder.drain()
    asked = await builder.build(CTX, f"my timezone{mark}", tools=_tools())
    assert not asked.cache_hit and builder.engine.calls == 2
    assert act.tools is not None and act.tools.next == sorted(TOOLBOX)[0]
    assert asked.tools is not None and asked.tools.next is None
    await builder.drain()
    again = await builder.build(CTX, f"please my timezone{mark}", tools=_tools())
    assert again.cache_hit and again.tools == asked.tools
    await builder.close()
