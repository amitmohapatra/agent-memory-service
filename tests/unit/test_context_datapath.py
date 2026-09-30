"""The query path's round trips, its bookkeeping and its serialisation.

Three things are asserted here, each of which was a per-request cost nobody was buying
anything with:

* the revisions are read once, and the authorization scope and the bundle are fetched in one
  cache read - and a cache hit resolves no scope at all;
* served-memory bookkeeping is buffered and flushed in bulk, and ``drain()`` still flushes
  (``tests/integration/test_access_tracking.py`` depends on exactly that);
* a cache hit is the stored bytes, not a parse followed by a re-serialisation.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import orjson
import pytest

from memory_service.adapters.authz.memory_provider import MemoryAuthorizationProvider
from memory_service.adapters.cache.memory_cache import MemoryCache
from memory_service.config.constants import CONTEXT, FROZEN_MODELS, RETRIEVAL
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.context_bundle import ContextBundle, ConversationWindow
from memory_service.domain.enums import QueryType
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.context.builder import ContextBuilder, bundle_to_api
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.retrieval.engine import Candidate, RetrievalResult
from memory_service.modules.retrieval.router import QueryRouter
from memory_service.ports.authorization import AuthorizedScope

CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1")
VISIBILITY = VisibilitySpecification(tenant_id="acme", keys=frozenset({"tenant:acme"}))
QUERY = "who owns the rollback plan?"

# ---------------------------------------------------------------------------
# fakes: a builder with no database, no store and no models
# ---------------------------------------------------------------------------


class _Revisions:
    def __init__(self) -> None:
        self.calls = 0
        self.values: dict[str, int] = {}

    async def get_many(self, tenant_id: str, keys: Any) -> dict[str, int]:
        self.calls += 1
        return {
            f"{kind.value}:{obj}": self.values.get(f"{kind.value}:{obj}", 0) for kind, obj in keys
        }

    async def bump(self, tenant_id: str, kind: Any, object_id: str = "") -> int:
        key = f"{kind.value}:{object_id}"
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]


class _Memories:
    def __init__(self) -> None:
        #: one entry per bump_access call: the ids it was given
        self.bumps: list[list[str]] = []

    async def bump_access(self, tenant_id: str, memory_ids: Any, *, at: datetime) -> int:
        self.bumps.append(list(memory_ids))
        return len(self.bumps[-1])


class _Threads:
    """No thread exists: the conversation window is empty."""

    async def get(self, tenant_id: str, thread_id: str) -> None:
        return None


class _UoW:
    def __init__(self, revisions: _Revisions, memories: _Memories) -> None:
        self.revisions = revisions
        self.memories = memories
        self.threads = _Threads()
        self.commits = 0

    async def __aenter__(self) -> _UoW:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def commit(self) -> None:
        self.commits += 1


class _Factory:
    def __init__(self) -> None:
        self.revisions = _Revisions()
        self.memories = _Memories()
        self.opened = 0

    def __call__(self) -> _UoW:
        self.opened += 1
        return _UoW(self.revisions, self.memories)


class _Indexer:
    fingerprint = "fp"


class _Engine:
    """Shaped like RetrievalEngine: an authz service, an indexer fingerprint, a retrieve."""

    def __init__(self, authz: AuthorizationService, memories: int = 2) -> None:
        self.authz = authz
        self.indexer = _Indexer()
        self.calls = 0
        self.count = memories

    #: the frozen multilingual encoder's floor
    relevance_floor = FROZEN_MODELS.dense_ml.relevance_floor

    async def score_similarity(self, result: RetrievalResult) -> None:
        """The fixture's memories carry no vectors: nothing to score."""

    async def retrieve(
        self, ctx: MemoryExecutionContext, query: str, **kwargs: Any
    ) -> RetrievalResult:
        self.calls += 1
        return RetrievalResult(
            routed=QueryRouter().routed(
                query,
                QueryType.USER_MEMORY,
                identifiers=[],
                signals={},
                has_thread=False,
            ),
            candidates=[
                Candidate(
                    record_id=f"mem_{i}",
                    kind="memory",
                    text=f"Priya owns the rollback plan, note {i}.",
                    score=0.9,
                    retrievers=["dense"],
                )
                for i in range(self.count)
            ],
            visibility=VisibilitySpecification(tenant_id="acme", keys=frozenset({"tenant:acme"})),
        )


class _Conversation:
    pass


def _builder(
    cache: MemoryCache | None = None, memories: int = 2, *, assist: LLMAssist | None = None
) -> ContextBuilder:
    authz = AuthorizationService(MemoryAuthorizationProvider(), cache)
    factory = _Factory()
    builder = ContextBuilder(
        factory,  # type: ignore[arg-type]
        _Engine(authz, memories),  # type: ignore[arg-type]
        _Conversation(),  # type: ignore[arg-type]
        cache,
        settings=CONTEXT,
        retrieval=RETRIEVAL,
        assist=assist,
    )
    return builder


@pytest.mark.parametrize(
    ("change", "models"),
    [
        ({"base_url": "http://another-gateway.test/v1"}, None),
        ({"api_key": "rotated-operator-key"}, None),
        ({}, {"query_expansion": "gemini/gemini-3.8-pro"}),
    ],
)
async def test_assisted_context_cache_is_not_reused_after_model_policy_changes(change, models):
    """A bundle built with query expansion depends on the gateway, the key and the model the
    tenant's policy names for the use: any of them changing is a different bundle."""
    from types import SimpleNamespace

    from tests.support_llm import bound_to, llm_settings

    cache = MemoryCache()
    provider = SimpleNamespace(enabled=True)
    first = _builder(cache, assist=LLMAssist(provider, llm_settings()))
    unchanged = _builder(cache, assist=LLMAssist(provider, llm_settings()))
    changed = _builder(cache, assist=LLMAssist(provider, llm_settings(**change)))
    try:
        with bound_to("acme", "user:u1", uses=["query_expansion"]):
            await first.build_api(CTX, QUERY, output="full")
            await first.drain()
            assert orjson.loads(await unchanged.build_api(CTX, QUERY, output="full"))["cache_hit"]
        with bound_to("acme", "user:u1", uses=["query_expansion"], models=models):
            assert not orjson.loads(await changed.build_api(CTX, QUERY, output="full"))["cache_hit"]
        assert changed.engine.calls == 1 and unchanged.engine.calls == 0
        assert "operator-key" not in changed._fingerprint
    finally:
        await first.close()
        await unchanged.close()
        await changed.close()


def test_unused_model_configuration_does_not_fragment_native_cache():
    from types import SimpleNamespace

    from tests.support_llm import llm_settings

    provider = SimpleNamespace(enabled=True)
    first = _builder(assist=LLMAssist(provider, llm_settings()))
    changed = _builder(assist=LLMAssist(provider, llm_settings(base_url="http://other.test/v1")))
    assert first._fingerprint == changed._fingerprint


def _parts(builder: ContextBuilder) -> tuple[_Factory, _Engine]:
    return builder.uow_factory, builder.engine  # type: ignore[return-value]


async def test_promoted_context_packs_100_memories_and_respects_a_smaller_budget():
    builder = _builder(memories=120)
    result = await builder.engine.retrieve(CTX, QUERY)
    bundle = builder._assemble(
        QUERY, result, ConversationWindow(), CONTEXT.token_budget, "revision"
    )
    assert [m.item_id for m in bundle.memories] == [f"mem_{i}" for i in range(100)]
    assert bundle.token_estimate <= CONTEXT.token_budget

    tight = builder._assemble(QUERY, result, ConversationWindow(), 200, "revision")
    assert 0 < len(tight.memories) < 100
    assert tight.token_estimate <= 200


@pytest.mark.parametrize("edge", ["GRAPH_EVIDENCE", "DERIVED_SOURCE"])
async def test_memory_companions_survive_a_full_primary_cap_but_obey_tokens(edge):
    builder = _builder(memories=3)
    builder.cfg = builder.cfg.model_copy(update={"memories_max": 2})
    result = await builder.engine.retrieve(CTX, QUERY)
    companion = Candidate(
        record_id="mem_bridge",
        kind="memory",
        text="The missing bridge evidence.",
        score=0.4,
        retrievers=["graph"],
        expansion_edge=edge,
        expanded_from="graph",
    )
    result.candidates.append(companion)
    bundle = builder._assemble(QUERY, result, ConversationWindow(), 500, "revision")
    assert [m.item_id for m in bundle.memories] == ["mem_0", "mem_1", "mem_bridge"]
    budget = sum(m.token_estimate for m in bundle.memories[:2])
    tight = builder._assemble(QUERY, result, ConversationWindow(), budget, "revision")
    assert [m.item_id for m in tight.memories] == ["mem_0", "mem_1"]
    assert sum(m.token_estimate for m in tight.memories) <= budget


# ---------------------------------------------------------------------------
# 2. round trips
# ---------------------------------------------------------------------------


async def test_the_revisions_are_read_once_for_the_bundle_and_the_scope() -> None:
    """The scope cache key and the bundle cache key both depend on revisions. They were two
    reads of the same table in two units of work; one read now answers both."""
    cache = MemoryCache()
    builder = _builder(cache)
    factory, _ = _parts(builder)
    await builder.build(CTX, QUERY)
    assert factory.revisions.calls == 1, "the revisions were read more than once"


async def test_the_scope_and_the_bundle_are_one_cache_read() -> None:
    cache = MemoryCache()
    builder = _builder(cache)
    reads: list[list[str]] = []
    original = cache.mget

    async def spy(keys: Any) -> list[bytes | None]:
        reads.append(list(keys))
        return await original(keys)

    cache.mget = spy  # type: ignore[method-assign]
    gets: list[str] = []
    plain = cache.get

    async def spy_get(key: str) -> bytes | None:
        gets.append(key)
        return await plain(key)

    cache.get = spy_get  # type: ignore[method-assign]

    await builder.build(CTX, QUERY)
    assert len(reads) == 1 and len(reads[0]) == 2, reads
    assert reads[0][0].startswith("authz:scope:acme:") and reads[0][1].startswith("ctx:acme:")
    assert gets == [], "the authorization scope was fetched a second time on its own"


async def test_a_cache_hit_resolves_no_authorization_scope() -> None:
    """On a hit the caller is served from the bundle, so the scope is never needed. Resolving
    it - a list-objects call when the scope cache is cold - was work for nobody."""
    cache = MemoryCache()
    builder = _builder(cache)
    _, engine = _parts(builder)
    provider = engine.authz.provider
    await builder.build(CTX, QUERY)
    await builder.drain()
    before = provider.check_calls, engine.calls

    again = await builder.build(CTX, QUERY)
    assert again.cache_hit is True
    assert (provider.check_calls, engine.calls) == before


async def test_the_bundle_write_is_off_the_request_path() -> None:
    """The response must not wait on a 30-80 KB cache write for the *next* caller."""
    cache = MemoryCache()
    builder = _builder(cache)
    key_before = len(cache._data)
    await builder.build(CTX, QUERY)
    # the scope was written by authz inline; the bundle has not been written yet
    assert not any(k.startswith("ctx:") for k in cache._data), cache._data.keys()
    await builder.drain()
    assert any(k.startswith("ctx:") for k in cache._data)
    assert len(cache._data) > key_before


async def test_a_config_fingerprint_is_computed_once() -> None:
    builder = _builder(MemoryCache())
    computed = 0
    original = builder._config_fingerprint

    def spy() -> str:
        nonlocal computed
        computed += 1
        return original()

    builder._config_fingerprint = spy  # type: ignore[method-assign]
    await builder.build(CTX, QUERY)
    await builder.build(CTX, QUERY + " again")
    assert computed == 0, "the fingerprint is fixed at construction; nothing may recompute it"


# ---------------------------------------------------------------------------
# 3. coalesced access bumps
# ---------------------------------------------------------------------------


async def test_access_bumps_are_buffered_and_drain_flushes_them() -> None:
    """``tests/integration/test_access_tracking.py`` builds one bundle, calls ``drain()`` and
    asserts the counter moved. Buffering must not break that contract."""
    builder = _builder(MemoryCache())
    factory, _ = _parts(builder)
    bundle = await builder.build(CTX, QUERY)
    served = [item.item_id for item in bundle.memories]
    assert served, "nothing was served, so the rest of this proves nothing"
    assert factory.memories.bumps == [], "the bump was on the request path"
    await builder.drain()
    assert factory.memories.bumps == [served]


async def test_many_builds_become_one_statement_per_tenant() -> None:
    builder = _builder(MemoryCache())
    factory, _ = _parts(builder)
    for i in range(5):
        await builder.build(CTX, f"{QUERY} {i}")
    await builder.drain()
    assert len(factory.memories.bumps) == 1, factory.memories.bumps
    assert sorted(factory.memories.bumps[0]) == ["mem_0", "mem_1"], (
        "an id served by several queries in one window is bumped once"
    )


async def test_a_full_buffer_flushes_before_the_timer() -> None:
    """At ``access_flush_max_ids`` the buffer flushes immediately: the window is a latency
    budget, not a licence to hold an unbounded list of ids."""
    per_bundle = CONTEXT.memories_max
    builder = _builder(MemoryCache(), memories=per_bundle)
    factory, _ = _parts(builder)
    for i in range(CONTEXT.access_flush_max_ids // per_bundle):
        await builder.build(CTX, f"{QUERY} {i}")
    for _ in range(20):
        if factory.memories.bumps:
            break
        await asyncio.sleep(0)
    assert len(factory.memories.bumps) == 1, "the buffer filled and nothing flushed it"
    assert len(factory.memories.bumps[0]) == per_bundle
    await builder.drain()


async def test_two_tenants_are_bumped_separately() -> None:
    builder = _builder(MemoryCache())
    factory, _ = _parts(builder)
    await builder.build(CTX, QUERY)
    await builder.build(MemoryExecutionContext(tenant_id="globex", user_id="u2"), QUERY)
    await builder.drain()
    assert len(factory.memories.bumps) == 2, "a bump statement is per tenant"


async def test_a_failing_bump_never_reaches_the_caller() -> None:
    builder = _builder(MemoryCache())
    factory, _ = _parts(builder)

    async def boom(*args: Any, **kwargs: Any) -> int:
        raise RuntimeError("database down")

    factory.memories.bump_access = boom  # type: ignore[method-assign]
    await builder.build(CTX, QUERY)
    await builder.drain()  # must not raise


async def test_the_decision_cache_switch_still_disables_the_scope_cache() -> None:
    """``decision_cache=False`` is the switch reached for during a stale-permission incident.

    The builder reads ``authz:scope:*`` from its own cache handle and hands the bytes to
    ``scope()``, which used to trust them before it ever looked at ``self.cache`` - so on the
    context path, the one path where the scope cache matters, turning the decision cache off
    would have changed nothing.
    """
    stale = AuthorizedScope(tenant_id="acme", principal="u1", thread_ids=["thr_stale"])
    provider = MemoryAuthorizationProvider()
    off = AuthorizationService(provider, MemoryCache(), decision_cache=False)
    resolved = await off.scope(
        CTX, revision_fingerprint="fp", cached_scope=stale.model_dump_json().encode()
    )
    assert resolved.thread_ids != ["thr_stale"], "the cached scope was used with the cache off"

    on = AuthorizationService(provider, MemoryCache(), decision_cache=True)
    trusted = await on.scope(
        CTX, revision_fingerprint="fp", cached_scope=stale.model_dump_json().encode()
    )
    assert trusted.thread_ids == ["thr_stale"]


async def test_container_shutdown_flushes_what_is_buffered() -> None:
    """``close()`` existed and nothing called it.

    ``Container.close()`` walks the dependencies, and the builder is a service, so at SIGTERM
    every worker silently dropped up to ``access_flush_seconds`` of served-memory ids - the
    counter the forgetting policy reads - plus whatever bundle write was in flight. A window
    per worker on every rolling deploy.
    """
    from memory_service.application.container import Container
    from memory_service.config.settings import Settings

    builder = _builder(MemoryCache())
    factory, _ = _parts(builder)
    container = Container(settings=Settings(_env_file=None), version="test")  # type: ignore[call-arg]
    container.services["context_builder"] = builder
    container.add_closer("context_builder", builder.close)

    await builder.build(CTX, QUERY)
    assert factory.memories.bumps == [], "the bump was on the request path"
    await container.close()
    assert factory.memories.bumps, "shutdown dropped the buffered access ids"


async def test_a_service_that_fails_to_close_does_not_stop_the_shutdown() -> None:
    from memory_service.application.container import Container
    from memory_service.config.settings import Settings

    closed: list[str] = []

    async def boom() -> None:
        raise RuntimeError("flush failed")

    async def fine() -> None:
        closed.append("second")

    container = Container(settings=Settings(_env_file=None), version="test")  # type: ignore[call-arg]
    container.add_closer("second", fine)
    container.add_closer("first", boom)
    await container.close()
    assert closed == ["second"]


def test_the_wiring_registers_the_builder_for_shutdown() -> None:
    """The test above proves the flush happens when something calls it; this proves the
    composition root is what calls it."""
    from memory_service.adapters.wiring import _wire_retrieval
    from memory_service.application.container import Container
    from memory_service.config.settings import Settings
    from memory_service.modules.rag.spaces import DenseSpaces

    container = Container(settings=Settings(_env_file=None), version="test")  # type: ignore[call-arg]
    container.services["uow_factory"] = _Factory()
    container.services["conversation"] = _Conversation()
    container.services["llm_assist"] = LLMAssist.disabled()
    container.services["authz"] = AuthorizationService(MemoryAuthorizationProvider(), None)

    class _Model:
        def fingerprint(self) -> str:
            return "fp"

    container.dense_spaces = DenseSpaces.single(_Model())  # type: ignore[arg-type]
    container.embedding = container.dense_spaces.primary
    container.sparse = _Model()
    _wire_retrieval(container)

    assert container.closers["context_builder"] == container.services["context_builder"].close


async def test_close_flushes_what_is_buffered() -> None:
    builder = _builder(MemoryCache())
    factory, _ = _parts(builder)
    await builder.build(CTX, QUERY)
    await builder.close()
    assert factory.memories.bumps


# ---------------------------------------------------------------------------
# 4. one serialisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("output", ["prompt", "full"])
async def test_a_cache_hit_returns_the_stored_bytes(output) -> None:
    cache = MemoryCache()
    builder = _builder(cache)
    first = await builder.build_api(CTX, QUERY, output=output)
    await builder.drain()
    prefix = "ctxp:" if output == "prompt" else "ctx:"
    stored = next(v for k, (v, _) in cache._data.items() if k.startswith(prefix))

    again = await builder.build_api(CTX, QUERY, output=output)
    assert again is stored, "a hit must be the stored bytes, not a parse and a re-serialisation"
    assert orjson.loads(again) == {
        **orjson.loads(first),
        **({"cache_hit": True} if output == "full" else {}),
    }
    assert orjson.loads(again)["rendered"]


async def test_the_prompt_form_is_three_fields_and_debug_is_never_a_cache_hit() -> None:
    cache = MemoryCache()
    builder = _builder(cache)
    prompt = orjson.loads(await builder.build_api(CTX, QUERY))
    assert set(prompt) == {"rendered", "bundle_id", "token_estimate"}
    await builder.drain()
    debug = orjson.loads(await builder.build_api(CTX, QUERY, output="full", debug=True))
    assert debug["cache_hit"] is False and "timings_ms" in debug["diagnostics"]
    assert "diagnostics" not in orjson.loads(await builder.build_api(CTX, QUERY, output="full"))
    await builder.close()


async def test_the_bytes_are_what_the_router_would_have_built() -> None:
    """``build_api`` replaces ``ContextResponse.model_validate(bundle_to_api(bundle))``; the
    content it sends must be identical to what that produced."""
    builder = _builder(MemoryCache())
    payload = orjson.loads(await builder.build_api(CTX, QUERY, output="full"))
    fresh = _builder(MemoryCache())
    expected = bundle_to_api(await fresh.build(CTX, QUERY))
    for key in ("query", "query_type", "evidence", "rendered", "token_budget", "cache_hit"):
        assert payload[key] == expected[key], key
    # every field but the evidence refs, whose observed_at is the moment each was built
    trimmed = [{k: v for k, v in m.items() if k != "evidence"} for m in payload["memories"]]
    assert trimmed == [
        {k: v for k, v in m.items() if k != "evidence"} for m in expected["memories"]
    ]
    assert [e["source_id"] for m in payload["memories"] for e in m["evidence"]] == [
        e["source_id"] for m in expected["memories"] for e in m["evidence"]
    ]


async def test_a_built_bundle_leaves_its_handles_and_evidence_for_later_calls() -> None:
    """``/v1/verify`` and the handle-taking calls read what a bundle carried by its id: the
    record is kept for the scope the bundle was built for, and only for it."""
    cache = MemoryCache()
    builder = _builder(cache)
    payload = orjson.loads(await builder.build_api(CTX, QUERY, output="full"))
    await builder.drain()
    record = await builder.records.load(CTX, payload["bundle_id"])
    assert record is not None
    assert (
        record.handles
        == payload["handles"]
        == {f"m{i}": m["item_id"] for i, m in enumerate(payload["memories"], start=1)}
    )
    assert [e.citation for e in record.evidence] == list(payload["handles"])
    other = MemoryExecutionContext(tenant_id="acme", user_id="u2")
    assert await builder.records.load(other, payload["bundle_id"]) is None
    assert "[m1]" in payload["rendered"] and "memory_id:" not in payload["rendered"]


async def test_the_two_entry_points_share_one_cache() -> None:
    cache = MemoryCache()
    builder = _builder(cache)
    await builder.build(CTX, QUERY)
    await builder.drain()
    hit = await builder.build_api(CTX, QUERY, output="full")
    assert orjson.loads(hit)["cache_hit"] is True


# ---------------------------------------------------------------------------
# 4. the verification rules tokenise each record once
# ---------------------------------------------------------------------------


async def test_content_terms_are_computed_once_per_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """``overlaps`` walks the evidence once, ``unsupported_subject`` walks it again per name
    and a third time for the other-terms check. Each walk re-tokenised the full text of every
    item - a few hundred regex passes over text that had not changed since the first one."""
    from memory_service.modules.context import evidence as ev

    seen: list[str] = []
    original = ev.content_terms

    def counted(text: str) -> set[str]:
        seen.append(text)
        return original(text)

    monkeypatch.setattr(ev, "content_terms", counted)
    memories = [
        Candidate(
            record_id=f"mem_{i}",
            kind="memory",
            text=f"Priya told Ravi that Acme owns the rollback plan, note {i}.",
            score=0.9,
        )
        for i in range(6)
    ]
    routed = QueryRouter().routed(
        "what did Priya and Ravi decide about the Acme rollback plan?",
        QueryType.USER_MEMORY,
        identifiers=[],
        signals={},
        has_thread=False,
    )
    stage = ev.VerificationStage(_Factory(), settings=RETRIEVAL)  # type: ignore[arg-type]
    await stage(CTX, routed, list(memories), VISIBILITY, {})

    repeated = {m.record_id: seen.count(m.text) for m in memories if seen.count(m.text) != 1}
    assert not repeated, f"tokenised more than once: {repeated}"


# ---------------------------------------------------------------------------
# 5. the route sends those bytes
# ---------------------------------------------------------------------------


async def test_the_bytes_still_satisfy_the_documented_response_contract() -> None:
    """``/v1/context`` returns the builder's bytes, so FastAPI no longer validates them.

    ``ContextResponse`` and the bodies under it forbid extra fields, and that validation was
    the only thing keeping the domain models from silently growing a field into the public
    contract. The check moves here: what the route sends must still be exactly a
    ContextResponse.
    """
    from memory_service.api.routers.v1.retrieval import ContextResponse, PromptContextResponse

    builder = _builder(MemoryCache())
    ContextResponse.model_validate(orjson.loads(await builder.build_api(CTX, QUERY, output="full")))
    debug = await builder.build_api(CTX, QUERY, output="full", debug=True)
    ContextResponse.model_validate(orjson.loads(debug))
    PromptContextResponse.model_validate(orjson.loads(await builder.build_api(CTX, QUERY)))


def test_the_context_route_sends_the_builder_bytes(settings: Any, overrides: Any) -> None:
    """The route is the consumer this change exists for.

    It used to parse the cached bundle, dump it, validate the dump into ContextResponse and
    serialise that - four passes over 30-80 KB for content the cache already held in exactly
    the form the caller wanted. The route now sends ``build_api``'s bytes; this asserts they
    reach the client unchanged, ``cache_hit`` and all.
    """
    from fastapi.testclient import TestClient

    from memory_service.api.app import create_app
    from memory_service.api.deps import get_container
    from memory_service.modules.auth.authentication import ServicePrincipal

    builder = _builder(MemoryCache())
    sent: list[bytes] = []

    class _Authenticator:
        async def authenticate(self, headers: dict[str, str]) -> ServicePrincipal:
            return ServicePrincipal(service_id="svc", mode="trusted_dev", claims={})

    class _Builder:
        async def build_api(self, ctx: Any, query: str, **kwargs: Any) -> bytes:
            payload = await builder.build_api(ctx, query, **kwargs)
            sent.append(payload)
            return payload

        async def build(self, *args: Any, **kwargs: Any) -> ContextBundle:
            raise AssertionError("the unverified arm must not build a ContextBundle")

    app_settings = settings

    class _Container:
        # `settings` because build_context reads authentication.tenant_claim to decide
        # whether the asserted tenant has to agree with the credential presenting it
        services = {
            "authenticator": _Authenticator(),
            "context_builder": _Builder(),
            "llm_assist": LLMAssist.disabled(),
        }
        settings = app_settings

    app = create_app(settings, overrides=overrides)
    app.dependency_overrides[get_container] = _Container
    with TestClient(app, raise_server_exceptions=False) as c:
        response = c.post(
            "/v1/context",
            headers={"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "u1"},
            json={"scope": {"thread_id": "thr_1"}, "query": QUERY},
        )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert response.content == sent[0], "the route re-serialised what the builder had built"
    assert set(response.json()) == {"rendered", "bundle_id", "token_estimate"}


async def test_a_ranked_memory_under_the_relevance_floor_is_not_packed() -> None:
    """A fusion score always fills the budget; the dense similarity decides what is worth a
    slot. Exact hits and companions are judged by what brought them in."""
    builder = _builder(memories=4)
    _, engine = _parts(builder)
    similarities = {"mem_0": 0.62, "mem_1": 0.05, "mem_2": engine.relevance_floor}

    async def score(result: RetrievalResult) -> None:
        for c in result.candidates:
            c.similarity = similarities.get(c.record_id)
        result.candidates[3].retrievers = ["exact"]
        result.candidates[3].similarity = 0.01

    engine.score_similarity = score  # type: ignore[method-assign]
    bundle = await builder.build(CTX, "who owns the rollback plan?")
    packed = {m.item_id: m.relevance for m in bundle.memories}
    assert set(packed) == {"mem_0", "mem_2", "mem_3"}, "mem_1 is under the floor"
    assert packed["mem_0"] == pytest.approx(0.62), "relevance is the similarity"
    assert packed["mem_3"] == 1.0, "an exact hit is exempt"
    assert bundle.diagnostics["below_relevance_floor"] == 1
    await builder.close()


def test_the_relevance_floor_belongs_to_the_encoder_that_measured_it() -> None:
    from memory_service.adapters.models.embeddings import HashEmbedding

    assert FROZEN_MODELS.dense_ml.relevance_floor == 0.2
    assert FROZEN_MODELS.dense.relevance_floor == 0.0, "never the space every query reads"
    assert HashEmbedding().relevance_floor == 0.0, "a stand-in has no calibrated cosine"
