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
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from memory_service.domain.script import Script, detect_script
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
        vectors = await asyncio.gather(*(space.encoder.embed_query(text) for space in pending))
        out.update(zip((space.name for space in pending), vectors, strict=True))
        return out

    def close(self) -> None:
        for space in self.spaces:
            close = getattr(space.encoder, "close", None)
            if close is not None:
                close()
