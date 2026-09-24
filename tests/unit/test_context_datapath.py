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
from memory_service.config.constants import CONTEXT, RETRIEVAL
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.context_bundle import ContextBundle
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


class _UoW:
    def __init__(self, revisions: _Revisions, memories: _Memories) -> None:
        self.revisions = revisions
        self.memories = memories
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


def _builder(cache: MemoryCache | None = None, memories: int = 2) -> ContextBuilder:
    authz = AuthorizationService(MemoryAuthorizationProvider(), cache)
    factory = _Factory()
    builder = ContextBuilder(
        factory,  # type: ignore[arg-type]
        _Engine(authz, memories),  # type: ignore[arg-type]
        _Conversation(),  # type: ignore[arg-type]
        cache,
        settings=CONTEXT,
        retrieval=RETRIEVAL,
    )
    return builder


def _parts(builder: ContextBuilder) -> tuple[_Factory, _Engine]:
    return builder.uow_factory, builder.engine  # type: ignore[return-value]


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

    container = Container(settings=Settings(_env_file=None), version="test")  # type: ignore[call-arg]
    container.services["uow_factory"] = _Factory()
    container.services["conversation"] = _Conversation()
    container.services["llm_assist"] = LLMAssist.disabled()
    container.services["authz"] = AuthorizationService(MemoryAuthorizationProvider(), None)

    class _Model:
        def fingerprint(self) -> str:
            return "fp"

    container.embedding = _Model()
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


async def test_a_cache_hit_returns_the_stored_bytes() -> None:
    cache = MemoryCache()
    builder = _builder(cache)
    first = await builder.build_api(CTX, QUERY)
    await builder.drain()
    stored = next(v for k, (v, _) in cache._data.items() if k.startswith("ctx:"))

    again = await builder.build_api(CTX, QUERY)
    assert again is stored, "a hit must be the stored bytes, not a parse and a re-serialisation"
    body = orjson.loads(again)
    assert body["cache_hit"] is True and orjson.loads(first)["cache_hit"] is False
    assert body["rendered"] and body["query"] == QUERY


async def test_the_bytes_are_what_the_router_would_have_built() -> None:
    """``build_api`` replaces ``ContextResponse.model_validate(bundle_to_api(bundle))``; the
    content it sends must be identical to what that produced."""
    builder = _builder(MemoryCache())
    payload = orjson.loads(await builder.build_api(CTX, QUERY))
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


async def test_a_bundle_cached_by_the_api_path_is_still_a_bundle() -> None:
    """``/v1/verify`` looks a bundle up by id while it is cached; the stored API dict must
    still validate as a ContextBundle."""
    cache = MemoryCache()
    builder = _builder(cache)
    payload = orjson.loads(await builder.build_api(CTX, QUERY))
    await builder.drain()
    found = await builder.cached(CTX, payload["bundle_id"])
    assert isinstance(found, ContextBundle)
    assert [i.item_id for i in found.memories] == [m["item_id"] for m in payload["memories"]]


async def test_the_two_entry_points_share_one_cache() -> None:
    cache = MemoryCache()
    builder = _builder(cache)
    await builder.build(CTX, QUERY)
    await builder.drain()
    hit = await builder.build_api(CTX, QUERY)
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
    stage = ev.VerificationStage(_Factory(), None, settings=RETRIEVAL)  # type: ignore[arg-type]
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
    from memory_service.api.routers.v1.retrieval import ContextResponse

    builder = _builder(MemoryCache())
    payload = orjson.loads(await builder.build_api(CTX, QUERY))
    ContextResponse.model_validate(payload)


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
        services = {"authenticator": _Authenticator(), "context_builder": _Builder()}
        settings = app_settings

    app = create_app(settings, overrides=overrides)
    app.dependency_overrides[get_container] = _Container
    with TestClient(app, raise_server_exceptions=False) as c:
        response = c.post(
            "/v1/context",
            headers={"X-API-Key": "test-key", "X-Memory-Tenant": "acme", "X-Memory-User": "u1"},
            json={"scope": {"thread_id": "thr_1"}, "query": QUERY},
        )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert response.content == sent[0], "the route re-serialised what the builder had built"
    assert response.json()["cache_hit"] is False
