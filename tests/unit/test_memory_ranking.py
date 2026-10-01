"""The memory search end to end over the in-process store: what the rerankers read, and that
a reranker that fails or misses its deadline costs the query its signal, not its answer."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from memory_service.domain.errors import DependencyUnavailable
from memory_service.ports.models import ProviderInfo
from tests.unit.test_engine_script_pruning import CTX, VISIBILITY, _parts

pytestmark = pytest.mark.unit


class _Reranker:
    info = ProviderInfo(name="fake", license="Apache-2.0", origin="internal", locality="local")

    def __init__(self, name: str, *, fail: bool = False, delay: float = 0.0) -> None:
        self.name = name
        self.fail = fail
        self.delay = delay
        self.calls: list[tuple[str, list[str]]] = []

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise DependencyUnavailable("down")
        return [float("Berlin" in t) for t in texts]


async def test_the_rerankers_read_the_question_and_each_memory_as_indexed() -> None:
    engine, *_ = await _parts()
    rerankers = (_Reranker("ml"), _Reranker("en"))
    engine.rerankers = rerankers
    result = await engine.retrieve(
        CTX, "What opened in Berlin?", kinds=("memory",), visibility=VISIBILITY
    )
    for reranker in rerankers:
        [(query, texts)] = reranker.calls
        assert query == "What opened in Berlin?"
        assert any(t.endswith(": The Berlin office opened in May.") for t in texts), texts
    assert result.diagnostics["memory_fusion"]["reranked"] == [True, True]
    assert result.candidates[0].record_id == "chk_berlin_memory"


@pytest.mark.parametrize("broken", [{"fail": True}, {"delay": 0.5}])
async def test_a_reranker_that_fails_or_is_late_leaves_a_ranked_answer(broken) -> None:
    engine, *_ = await _parts(memory_rerank_timeout_ms=100)
    engine.rerankers = (_Reranker("ml", **broken), _Reranker("en"))
    result = await engine.retrieve(
        CTX, "What opened in Berlin?", kinds=("memory",), visibility=VISIBILITY
    )
    assert result.diagnostics["rerank_unavailable"] == ["ml"]
    assert result.diagnostics["memory_fusion"]["reranked"] == [False, True]
    assert result.candidates[0].record_id == "chk_berlin_memory"
