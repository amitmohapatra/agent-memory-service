"""Which learned procedures a task is about: the one place that decides it, for the skills of a
context and the plan of the tool hints alike (``ToolHintsService.procedures``).

Two tests, in order:

- words, the fast path and no model: the task's pattern shares enough tokens with the
  procedure's (``PROCEDURE_MATCH``, a Jaccard of ``patterns.similarity``). "update quote
  Q-9006 with APAC price for SKU-7" is the procedure learned from "update quote Q-1183 with
  EMEA price for SKU-22";
- meaning, for a procedure the words miss: the cosine of the task's query vector to the
  pattern's, in the space every query is searched in (the multilingual one, so a Hindi or an
  Arabic task is read too), at least that encoder's own ``procedure_cosine``. "MFA lockout
  again" shares one word with "reset multi-factor authentication for a locked-out user" and
  is the same task. Without that floor (a stand-in encoder) only the words count.

A word match outranks a meaning match: it is the stronger evidence, and an unchanged fast
path keeps every procedure it found where it was.

A pattern is stable text, so its vector is computed once per process (and behind the
indexer's content-hash cache, once per deployment): an in-process LRU keyed by the encoder's
fingerprint and the pattern's hash. The task's vector is the one the request already has
(the pushed context's retrieval, the tool search) whenever it has one.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np

from memory_service.domain.ids import content_hash
from memory_service.domain.tools import StoredProcedure
from memory_service.modules.rag.indexer import Indexer
from memory_service.modules.retrieval.engine import QueryVectors
from memory_service.modules.tools.patterns import similarity, task_pattern

#: Pattern similarity (token Jaccard) at which a stored procedure is taken to be about the task.
PROCEDURE_MATCH: Final = 0.6
#: Pattern vectors one process keeps (~1.5 KB each at 384 dimensions).
PATTERN_VECTORS_MAX: Final = 4096
#: Keeps a pattern's cached vector apart from a memory or a tool of the same text.
_PATTERN_KEY: Final = ":procedure1"

#: The task's vectors: in hand, still being encoded by the request's retrieval (resolved by
#: then), or none (encoded here when a pattern's words miss).
TaskVectors = QueryVectors | asyncio.Future[QueryVectors | None] | None


def rank_procedures(
    task: str,
    procedures: Sequence[StoredProcedure],
    k: int,
    meaning: Mapping[str, float] | None = None,
) -> list[StoredProcedure]:
    """The procedures about this task, best first: those whose pattern shares enough of its
    words (``PROCEDURE_MATCH``), the closest first, then those ``meaning`` keeps (procedure id
    -> cosine of its pattern to the task, the ones at the encoder's floor), closest first."""
    pattern = task_pattern(task)
    meaning = meaning or {}
    worded: list[tuple[float, StoredProcedure]] = []
    meant: list[tuple[float, StoredProcedure]] = []
    for p in procedures:
        if (score := similarity(pattern, p.pattern)) >= PROCEDURE_MATCH:
            worded.append((score, p))
        elif p.procedure_id in meaning:
            meant.append((meaning[p.procedure_id], p))
    out: list[StoredProcedure] = []
    for tier in (worded, meant):
        tier.sort(key=lambda sp: (-sp[0], -sp[1].success_rate, -sp[1].support, sp[1].procedure_id))
        out.extend(p for _, p in tier)
    return out[:k]


def missed_by_words(task: str, procedures: Sequence[StoredProcedure]) -> list[StoredProcedure]:
    pattern = task_pattern(task)
    return [p for p in procedures if similarity(pattern, p.pattern) < PROCEDURE_MATCH]


def _unit(vector: Sequence[float]) -> np.ndarray:
    out = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(out))
    return out / norm if norm else out


class ProcedureMatcher:
    """Ranks the procedures a caller may read by how much each is about a task.
    ``indexer``: the dense spaces and their cache; ``None`` matches by words alone."""

    def __init__(self, indexer: Indexer | None, *, capacity: int = PATTERN_VECTORS_MAX) -> None:
        self.indexer = indexer
        self.capacity = capacity
        self._patterns: OrderedDict[str, np.ndarray] = OrderedDict()

    @property
    def floor(self) -> float:
        """The least cosine of a meaning match; 0: none (the encoder has no calibrated one)."""
        if self.indexer is None:
            return 0.0
        return self.indexer.spaces.primary.procedure_cosine

    async def rank(
        self,
        task: str,
        procedures: Sequence[StoredProcedure],
        k: int,
        *,
        vectors: TaskVectors = None,
    ) -> list[StoredProcedure]:
        """The procedures about ``task``, best first (``rank_procedures``). The task is read
        for its meaning only when the words leave fewer than ``k`` and miss a procedure."""
        missed = missed_by_words(task, procedures)
        floor = self.floor
        if not missed or floor <= 0 or len(procedures) - len(missed) >= k:
            return rank_procedures(task, procedures, k)
        task_vector = await self._task_vector(task, vectors)
        patterns = await self._pattern_vectors([p.pattern for p in missed])
        cosines = np.stack(patterns) @ task_vector
        meaning = {
            p.procedure_id: float(c) for p, c in zip(missed, cosines, strict=True) if c >= floor
        }
        return rank_procedures(task, procedures, k, meaning)

    def _key(self, pattern: str) -> str:
        assert self.indexer is not None
        return f"{self.indexer.spaces.primary.fingerprint()}:{content_hash(pattern)}"

    async def _task_vector(self, task: str, vectors: TaskVectors) -> np.ndarray:
        assert self.indexer is not None
        if isinstance(vectors, asyncio.Future):
            vectors = await vectors
        spaces = self.indexer.spaces
        dense = vectors.dense.get(spaces.primary_space.name) if vectors is not None else None
        return _unit(dense if dense else await spaces.embed_primary_query(task))

    async def _pattern_vectors(self, patterns: Sequence[str]) -> list[np.ndarray]:
        """Each pattern's unit vector: the ones this process has, the rest encoded together."""
        assert self.indexer is not None
        keys = [self._key(p) for p in patterns]
        todo = list(
            dict.fromkeys(
                p for p, key in zip(patterns, keys, strict=True) if key not in self._patterns
            )
        )
        if todo:
            fresh = await self.indexer.embed_primary_cached(
                todo, [content_hash(p) + _PATTERN_KEY for p in todo]
            )
            for pattern, vector in zip(todo, fresh, strict=True):
                self._patterns[self._key(pattern)] = _unit(vector)
        out = []
        for key in keys:
            self._patterns.move_to_end(key)
            out.append(self._patterns[key])
        while len(self._patterns) > self.capacity:
            self._patterns.popitem(last=False)
        return out
