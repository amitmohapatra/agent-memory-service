"""Two encoders, one query: the script decides which spaces are searched, the spaces are
encoded at the same time, and a vector a caller already has is not encoded again."""

# ruff: noqa: RUF001 - literal multilingual fixtures intentionally use non-Latin letters.

from __future__ import annotations

import asyncio
import time

import pytest

from memory_service.adapters.models.embeddings import HashEmbedding
from memory_service.domain.script import Script
from memory_service.modules.rag.spaces import DenseSpace, DenseSpaces
from memory_service.ports.search import VectorName

pytestmark = pytest.mark.unit


class _Spy(HashEmbedding):
    def __init__(self, dimension: int = 8, *, delay: float = 0.0) -> None:
        super().__init__(dimension=dimension)
        self.queries: list[str] = []
        self.delay = delay
        self.closed = False

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        if self.delay:
            await asyncio.sleep(self.delay)
        return await super().embed_query(text)

    def fingerprint(self) -> str:
        return f"spy-d{self.dimension}"

    def close(self) -> None:
        """The real encoders own a thread each; this records that it was released."""
        self.closed = True


def _spaces(delay: float = 0.0) -> tuple[DenseSpaces, _Spy, _Spy]:
    english, multilingual = _Spy(8, delay=delay), _Spy(16, delay=delay)
    spaces = DenseSpaces(
        [
            DenseSpace(VectorName.DENSE_EN, english, query_scripts=frozenset({Script.LATIN})),
            DenseSpace(VectorName.DENSE_ML, multilingual),
        ]
    )
    return spaces, english, multilingual


async def test_a_latin_query_searches_both_spaces() -> None:
    spaces, english, multilingual = _spaces()
    vectors = await spaces.embed_query("where is the office")
    assert set(vectors) == {VectorName.DENSE_EN, VectorName.DENSE_ML}
    assert english.queries == multilingual.queries == ["where is the office"]
    assert len(vectors[VectorName.DENSE_EN]) == 8 and len(vectors[VectorName.DENSE_ML]) == 16


@pytest.mark.parametrize(
    "query",
    [
        "Где находится офис?",
        "أين يقع المكتب؟",
        "สำนักงานอยู่ที่ไหน",
        "कार्यालय कहाँ है",
        "办公室在哪里",
        "Πού είναι το γραφείο;",
    ],
)
async def test_a_non_latin_query_never_reaches_the_english_encoder(query: str) -> None:
    spaces, english, multilingual = _spaces()
    vectors = await spaces.embed_query(query)
    assert set(vectors) == {VectorName.DENSE_ML}
    assert english.queries == [] and multilingual.queries == [query]


async def test_the_spaces_are_encoded_concurrently() -> None:
    spaces, _, _ = _spaces(delay=0.05)
    started = time.perf_counter()
    await spaces.embed_query("concurrent")
    assert time.perf_counter() - started < 0.09, "two 50 ms encodes ran one after the other"


async def test_a_known_vector_is_reused_and_a_known_pruned_space_is_dropped() -> None:
    spaces, english, multilingual = _spaces()
    known = {VectorName.DENSE_ML: [1.0] * 16}
    vectors = await spaces.embed_query("reuse me", known=known)
    assert vectors[VectorName.DENSE_ML] == [1.0] * 16 and multilingual.queries == []
    assert english.queries == ["reuse me"]
    # a known English vector on a Cyrillic query is not searched: the script rules it out
    vectors = await spaces.embed_query("Где офис", known={VectorName.DENSE_EN: [0.5] * 8})
    assert set(vectors) == {VectorName.DENSE_ML}


def test_the_fingerprint_names_every_space_in_name_order() -> None:
    spaces, _, _ = _spaces()
    assert spaces.fingerprint() == "dense_en=spy-d8+dense_ml=spy-d16"
    assert spaces.dimensions == {VectorName.DENSE_EN: 8, VectorName.DENSE_ML: 16}
    assert spaces.primary_space.name is VectorName.DENSE_ML


def test_a_single_space_answers_every_script() -> None:
    spaces = DenseSpaces.single(_Spy(4), VectorName.DENSE_EN)
    assert spaces.for_script(Script.THAI)[0].name is VectorName.DENSE_EN
    assert spaces.primary_space.name is VectorName.DENSE_EN


async def test_the_container_closes_every_space_not_only_the_primary() -> None:
    """The defect this covers: ``Container._close_models`` closed ``embedding``, which is the
    primary space alone, so the English specialist's runner thread outlived every container.
    One leaked thread per arm, on the host where the thread budget is what is being measured.
    """
    from memory_service.application.container import Container
    from memory_service.config.settings import Settings

    spaces, english, multilingual = _spaces()
    container = Container(settings=Settings(_env_file=None), version="test")  # type: ignore[call-arg]
    container.dense_spaces = spaces
    container.embedding = spaces.primary

    await container.close()

    assert english.closed and multilingual.closed


def test_invalid_layouts_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        DenseSpaces([])
    with pytest.raises(ValueError, match="unique"):
        DenseSpaces(
            [DenseSpace(VectorName.DENSE_ML, _Spy()), DenseSpace(VectorName.DENSE_ML, _Spy())]
        )
    with pytest.raises(ValueError, match="every script"):
        DenseSpaces(
            [DenseSpace(VectorName.DENSE_EN, _Spy(), query_scripts=frozenset({Script.LATIN}))]
        )


async def test_query_vectors_are_cached_per_space_and_survive_a_cache_outage() -> None:
    from memory_service.adapters.cache.memory_cache import MemoryCache
    from memory_service.ports.cache import CacheUnavailable

    spaces, english, multilingual = _spaces()
    cache = MemoryCache()
    spaces.use_cache(cache, ttl_seconds=60)
    first = await spaces.embed_query("where is the office")
    again = await spaces.embed_query("where is the office")
    assert english.queries == multilingual.queries == ["where is the office"]
    for name in first:
        assert again[name] == pytest.approx(first[name], abs=1e-6)
    keys = [k for k in cache._data if k.startswith("emb:")]
    assert len(keys) == 2 and all(":q:" in k for k in keys)
    # the primary vector the semantic cache asks for is the same cached entry
    assert await spaces.embed_primary_query("where is the office") == pytest.approx(
        first[VectorName.DENSE_ML], abs=1e-6
    )
    assert multilingual.queries == ["where is the office"]

    async def down(*_: object, **__: object) -> None:
        raise CacheUnavailable("down")

    cache.mget = down  # type: ignore[method-assign]
    cache.mset = down  # type: ignore[method-assign]
    assert set(await spaces.embed_query("another question")) == set(first)
    assert english.queries[-1] == "another question"
