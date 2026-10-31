"""A tool hint's confidence says whether the tool fits the task, not only that it ranked first.

The tool search's fusion score is relative to the task's best hit, so the first tool read 1.0
whatever the task: "What's the weather on Mars?" hinted a stock tool at 0.63, the harness took
that as a tool answering the question, narrowed its tools and dropped its "I don't know" note.
The search now also reads how close each hit's indexed text is to the task (its cosine in the
space every query is searched in), and a hit below the encoder's calibrated ``tool_cosine``
still ranks where it ranked, but its match counts for nothing in its confidence.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from memory_service.adapters.models.embeddings import HashEmbedding
from memory_service.config.constants import FROZEN_MODELS
from memory_service.domain.context_bundle import TOOL_ANSWERS_CONFIDENCE
from memory_service.domain.script import Script
from memory_service.domain.tools import StoredProcedure, ToolStats
from memory_service.modules.retrieval.engine import QueryVectors
from memory_service.modules.tools.hints import _candidates
from memory_service.modules.tools.index import ToolIndex, ToolMatch
from memory_service.ports.search import Retriever, SearchHit, VectorName

pytestmark = pytest.mark.unit

TASK_VECTOR = [1.0, 0.0]


class _Store:
    """The fused hits of one search and each hit's cosine to the task."""

    def __init__(self, hits: list[tuple[str, str, float]], cosines: dict[str, float]) -> None:
        self.hits = [
            SearchHit(
                record_id=rid, score=score, retriever=Retriever.DENSE_ML, payload={"text": name}
            )
            for rid, name, score in hits
        ]
        self.cosines = cosines
        self.similarity_reads: list[tuple[VectorName, list[str]]] = []

    async def ensure_collection(self, spec: Any) -> None:
        return None

    async def search_hybrid(self, *_: object, **__: object) -> list[SearchHit]:
        return self.hits

    async def similarity(
        self, collection: str, name: VectorName, vector: Any, record_ids: list[str]
    ) -> dict[str, float]:
        self.similarity_reads.append((name, list(record_ids)))
        return {r: self.cosines[r] for r in record_ids if r in self.cosines}


def _indexer(tool_cosine: float) -> Any:
    primary = SimpleNamespace(tool_cosine=tool_cosine)
    return SimpleNamespace(
        spaces=SimpleNamespace(
            primary=primary,
            primary_space=SimpleNamespace(name=VectorName.DENSE_ML, encoder=primary),
            dimensions={},
        ),
        sparse=SimpleNamespace(server_side_idf=True),
        collection=lambda name: name,
    )


def _vectors() -> QueryVectors:
    return QueryVectors(dense={VectorName.DENSE_ML: TASK_VECTOR}, sparse=None, script=Script.LATIN)


HITS = [("tol_a", "excess_stock", 0.5), ("tol_b", "lost_sales", 0.33), ("tol_c", "app_help", 0.25)]


async def test_a_hit_close_to_the_task_matches_and_one_far_from_it_does_not() -> None:
    store = _Store(HITS, {"tol_a": 0.46, "tol_b": 0.31, "tol_c": 0.12})
    index = ToolIndex(None, _indexer(0.3), store)  # type: ignore[arg-type]
    found = await index.search(
        "acme", None, "Which styles are overstocked at S40?", vectors=_vectors()
    )
    # the order and the relevance are the fusion's, as before
    assert [(m.name, round(m.relevance, 2)) for m in found] == [
        ("excess_stock", 1.0),
        ("lost_sales", 0.66),
        ("app_help", 0.5),
    ]
    assert [m.matches for m in found] == [True, True, False]
    # one bounded read of the hits, in the space every query is searched in
    assert store.similarity_reads == [(VectorName.DENSE_ML, ["tol_a", "tol_b", "tol_c"])]


async def test_a_task_no_tool_is_about_finds_its_tools_but_none_matches() -> None:
    store = _Store(HITS, {"tol_a": 0.07, "tol_b": 0.05, "tol_c": 0.02})
    index = ToolIndex(None, _indexer(0.3), store)  # type: ignore[arg-type]
    found = await index.search("acme", None, "What's the weather on Mars?", vectors=_vectors())
    assert found[0].name == "excess_stock" and found[0].relevance == 1.0
    assert not any(m.matches for m in found)


async def test_an_encoder_with_no_calibrated_tool_cosine_reads_no_similarity() -> None:
    """The stand-in encoder's cosine means nothing (``HashEmbedding``): every hit counts, as
    it did, and no similarity is read."""
    store = _Store(HITS, {"tol_a": 0.01})
    index = ToolIndex(None, _indexer(0.0), store)  # type: ignore[arg-type]
    found = await index.search("acme", None, "anything", vectors=_vectors())
    assert all(m.matches for m in found) and store.similarity_reads == []


def test_the_space_every_query_reads_is_calibrated_and_the_stand_in_is_not() -> None:
    assert FROZEN_MODELS.dense_ml.tool_cosine > 0
    assert HashEmbedding().tool_cosine == 0.0


def _confidences(found: list[ToolMatch], **kwargs: Any) -> list[tuple[str, float]]:
    ranked = _candidates(
        found,
        kwargs.get("plan"),
        kwargs.get("step"),
        kwargs.get("stats", {}),
        None,
        5,
    )
    return [(c.name, c.confidence) for c in ranked]


UNRELATED = [ToolMatch("find_sku", 1.0, False), ToolMatch("get_stock", 0.5, False)]
MATCHING = [ToolMatch("find_sku", 1.0, True), ToolMatch("get_stock", 0.5, True)]


def test_a_task_no_tool_is_about_gets_a_confidence_below_the_floor_in_the_same_order() -> None:
    """Before: ``find_sku`` at 0.63 for "What's my sister's name?", above the 0.5 from which
    a tool is taken to answer the question (here and in the harness)."""
    assert _confidences(UNRELATED) == [("find_sku", 0.0), ("get_stock", 0.0)]
    ranked = _candidates(UNRELATED, None, None, {}, None, 5)
    assert all("matches the task" not in c.why for c in ranked)
    # a tool used often and recently is still no answer to a question it is not about
    recent = ToolStats(
        tool_name="find_sku",
        calls=10,
        successes=10,
        last_used_at=datetime.now(UTC),
    )
    best = max(c for _, c in _confidences(UNRELATED, stats={"find_sku": recent}))
    assert best < TOOL_ANSWERS_CONFIDENCE


def test_a_matching_task_keeps_its_confidence_and_its_order() -> None:
    assert _confidences(MATCHING) == [("find_sku", 0.63), ("get_stock", 0.39)]
    ranked = _candidates(MATCHING, None, None, {}, None, 5)
    assert ranked[0].why.startswith("matches the task (1.00)")
    # the score, the order, is the same whether or not the search's match counts
    assert [c.score for c in ranked] == [
        c.score for c in _candidates(UNRELATED, None, None, {}, None, 5)
    ]


def test_the_learned_plan_s_next_step_stays_confident_whatever_the_search_found() -> None:
    plan = StoredProcedure(
        tenant_id="acme",
        scope_key="k",
        pattern="find sku",
        steps=[{"ordinal": 0, "tool": "find_sku"}],
    )
    (name, confidence), *_ = _confidences(UNRELATED, plan=plan, step="find_sku")
    assert name == "find_sku" and confidence >= TOOL_ANSWERS_CONFIDENCE
