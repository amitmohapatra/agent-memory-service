"""The dense spaces a record is indexed in and a query is searched in.

One encoder used to be the whole dense side. Two specialists replace it - an English
encoder that wins on English and a multilingual one that reads every script - and the
measured ensemble (SciFact 0.7557 nDCG@10 against 0.7409 for English alone; XQuAD paragraph
R@10 0.9883 against 0.6596) fuses their two ranked lists with BM25 rather than concatenating
their vectors. So a record carries one vector per space, and a query searches the spaces
its script calls for: the English space only answers Latin-script queries, the multilingual
space answers all of them.

Each encoder owns its own single-thread runner, so the spaces are encoded concurrently:
``asyncio.gather`` over two ``SerialRunner`` executors is two forward passes at once, not
one after the other.

Query vectors are cached (``emb:<encoder fingerprint>:q:<query hash>``, beside the document
vectors the indexer caches) once a cache is attached: every bundle-cache miss used to encode
the same query again - a revision bump, a different budget or a second reader asking the
same thing each cost a forward pass per space on the request path.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from memory_service.domain.script import Script, detect_script
from memory_service.ports.cache import CacheProvider, CacheUnavailable
from memory_service.ports.models import EmbeddingProvider
from memory_service.ports.search import VectorName


@dataclass(frozen=True)
class DenseSpace:
    name: VectorName
    encoder: EmbeddingProvider
    #: the query scripts this space is searched for; ``None`` means every script
    query_scripts: frozenset[Script] | None = None

    def queries(self, script: Script) -> bool:
        return self.query_scripts is None or script in self.query_scripts


class DenseSpaces:
    """The spaces of one deployment, in the order their vectors are fused."""

    def __init__(self, spaces: Sequence[DenseSpace]) -> None:
        if not spaces:
            raise ValueError("a deployment needs at least one dense space")
        names = [space.name for space in spaces]
        if len(set(names)) != len(names):
            raise ValueError(f"dense space names must be unique: {names}")
        if not any(space.query_scripts is None for space in spaces):
            raise ValueError("one dense space must be searched for every script")
        self.spaces = tuple(spaces)
        self._cache: CacheProvider | None = None
        self._query_ttl = 0

    def use_cache(self, cache: CacheProvider | None, *, ttl_seconds: int) -> None:
        """Cache query vectors in ``cache`` for ``ttl_seconds`` (``None``: no cache)."""
        self._cache = cache
        self._query_ttl = ttl_seconds

    @staticmethod
    def query_key(space: DenseSpace, text: str) -> str:
        """A query's vector key. ``q`` keeps it apart from the document vector of the same
        text: an asymmetric encoder embeds a query and a passage differently."""
        digest = hashlib.sha256(text.encode()).hexdigest()[:32]
        return f"emb:{space.encoder.fingerprint()}:q:{digest}"

    @classmethod
    def single(
        cls, encoder: EmbeddingProvider, name: VectorName = VectorName.DENSE_ML
    ) -> DenseSpaces:
        """One encoder searched for every script: the stand-in and the single-encoder arm."""
        return cls([DenseSpace(name, encoder)])

    @property
    def names(self) -> tuple[VectorName, ...]:
        return tuple(space.name for space in self.spaces)

    @property
    def primary_space(self) -> DenseSpace:
        """The space searched for every script: the one vector that stands for a text
        wherever a single vector is wanted (the semantic cache, the dedup cosine)."""
        return next(space for space in self.spaces if space.query_scripts is None)

    @property
    def primary(self) -> EmbeddingProvider:
        return self.primary_space.encoder

    @property
    def dimensions(self) -> dict[VectorName, int]:
        return {space.name: space.encoder.dimension for space in self.spaces}

    def fingerprint(self) -> str:
        """The name of the vector spaces together: every space's own fingerprint, in name
        order, so adding a space or swapping one encoder mints a new collection."""
        return "+".join(
            f"{space.name}={space.encoder.fingerprint()}"
            for space in sorted(self.spaces, key=lambda space: space.name)
        )

    def for_script(self, script: Script) -> tuple[DenseSpace, ...]:
        return tuple(space for space in self.spaces if space.queries(script))

    async def embed_query(
        self,
        text: str,
        *,
        script: Script | None = None,
        known: Mapping[VectorName, Sequence[float]] | None = None,
    ) -> dict[VectorName, list[float]]:
        """The query vectors the script calls for, encoded concurrently.

        ``known`` carries vectors a caller already has (the semantic cache encodes the
        primary space before retrieval runs); those spaces are not encoded again.
        """
        wanted = self.for_script(script if script is not None else detect_script(text))
        out: dict[VectorName, list[float]] = {
            space.name: list(known[space.name])
            for space in wanted
            if known is not None and space.name in known
        }
        pending = [space for space in wanted if space.name not in out]
        out.update(await self._cached_queries(text, pending))
        return out

    async def embed_primary_query(self, text: str) -> list[float]:
        """The primary space's query vector, through the same cache."""
        space = self.primary_space
        return (await self._cached_queries(text, [space]))[space.name]

    async def _cached_queries(
        self, text: str, spaces: Sequence[DenseSpace]
    ) -> dict[VectorName, list[float]]:
        if not spaces:
            return {}
        out: dict[VectorName, list[float]] = {}
        keys = [self.query_key(space, text) for space in spaces]
        if self._cache is not None:
            try:
                found = await self._cache.mget(keys)
            except CacheUnavailable:
                found = [None] * len(keys)
            for space, raw in zip(spaces, found, strict=True):
                if raw:
                    out[space.name] = list(struct.unpack(f"<{len(raw) // 4}f", raw))
        missing = [space for space in spaces if space.name not in out]
        vectors = await asyncio.gather(*(space.encoder.embed_query(text) for space in missing))
        fresh = {space.name: list(v) for space, v in zip(missing, vectors, strict=True)}
        out.update(fresh)
        if self._cache is not None and fresh:
            with contextlib.suppress(CacheUnavailable):
                await self._cache.mset(
                    {
                        self.query_key(space, text): struct.pack(
                            f"<{len(fresh[space.name])}f", *fresh[space.name]
                        )
                        for space in missing
                    },
                    ttl_seconds=self._query_ttl,
                )
        return out

    def close(self) -> None:
        for space in self.spaces:
            close = getattr(space.encoder, "close", None)
            if close is not None:
                close()
