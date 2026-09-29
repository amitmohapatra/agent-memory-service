"""Multi-hop question decomposition: what it costs, what it fuses, and that it is off by
default.

The measured finding it exists for is in docs/PHASE7-RESULTS-2026-09-28.md: multi-hop source
recall 0.3977 at depth 10 against 0.6338 at depth 50. These tests pin the mechanism; whether it
moves that number is an arm, not a unit test.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from memory_service.modules.retrieval.decomposition import (
    DIAGNOSTIC,
    LLM_USE,
    QueryDecomposer,
    decompose_and_retrieve,
)
from memory_service.modules.retrieval.engine import Candidate
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.unit

QUESTION = "where does the person who led the payments migration live"
PARTS = ["who led the payments migration", "where does Priya Raman live"]


def _candidate(record_id: str, score: float = 1.0) -> Candidate:
    return Candidate(record_id=record_id, kind="memory", text=record_id, score=score)


@dataclass
class _Cfg:
    final_k: int = 10


@dataclass
class _Result:
    candidates: list[Candidate]
    visibility: str = "vis"
    diagnostics: dict[str, Any] = field(default_factory=dict)


class _Engine:
    """Answers each query with the candidates scripted for it, and counts the calls."""

    def __init__(self, answers: dict[str, list[str]], *, failing: str | None = None) -> None:
        self.answers = answers
        self.failing = failing
        self.queries: list[str] = []
        self.options: list[dict[str, Any]] = []
        self.cfg = _Cfg()

    async def retrieve(self, ctx, query, **options):
        self.queries.append(query)
        self.options.append(options)
        if query == self.failing:
            raise RuntimeError("store unavailable")
        return _Result([_candidate(r) for r in self.answers.get(query, [])])


async def test_a_default_read_never_decomposes_and_never_calls_a_model() -> None:
    engine = _Engine({QUESTION: ["a", "b"]})
    with mocked_gateway([{"sub_questions": PARTS}]) as gateway:
        decomposer = QueryDecomposer(gateway.assist(uses=["query_expansion"]))
        assert decomposer.enabled is False

        result = await decompose_and_retrieve(engine, decomposer, None, QUESTION)

        assert gateway.route.call_count == 0
        assert engine.queries == [QUESTION]
        assert DIAGNOSTIC not in result.diagnostics
        assert [c.record_id for c in result.candidates] == ["a", "b"]


async def test_without_any_model_key_the_path_is_the_plain_retrieval() -> None:
    engine = _Engine({QUESTION: ["a"]})
    result = await decompose_and_retrieve(engine, QueryDecomposer(), None, QUESTION)
    assert engine.queries == [QUESTION] and [c.record_id for c in result.candidates] == ["a"]


async def test_each_sub_question_is_retrieved_and_the_lists_are_fused() -> None:
    engine = _Engine({QUESTION: ["a", "b"], PARTS[0]: ["bridge"], PARTS[1]: ["answer", "b"]})
    with mocked_gateway([{"sub_questions": PARTS}]) as gateway:
        decomposer = QueryDecomposer(gateway.assist(uses=[LLM_USE]))

        result = await decompose_and_retrieve(engine, decomposer, None, QUESTION)

    assert engine.queries == [QUESTION, *PARTS]
    # "b" is retrieved by two questions, so it outranks "bridge" and "answer", which each
    # come from one. That is the behaviour a multi-hop question needs.
    assert result.candidates[0].record_id == "b"
    assert {c.record_id for c in result.candidates} == {"a", "b", "bridge", "answer"}
    assert result.diagnostics[DIAGNOSTIC]["sub_questions"] == PARTS
    assert result.diagnostics[DIAGNOSTIC]["retrievals"] == 3
    # What the decomposition cost is reported beside what it changed. The counter is
    # request-scoped, so outside a request it reads zero; the field is the contract.
    assert result.diagnostics[DIAGNOSTIC]["llm_tokens"] >= 0
    fused = next(c for c in result.candidates if c.record_id == "b")
    assert fused.payload["fused_from_questions"] == 2


async def test_the_extra_retrievals_reuse_the_first_one_s_visibility() -> None:
    engine = _Engine({QUESTION: ["a"], PARTS[0]: ["b"]})
    with mocked_gateway([{"sub_questions": [PARTS[0]]}]) as gateway:
        decomposer = QueryDecomposer(gateway.assist(uses=[LLM_USE]))
        await decompose_and_retrieve(engine, decomposer, None, QUESTION, visibility=None)
    assert engine.options[1]["visibility"] == "vis", "authorization is resolved once"
    assert engine.options[1]["query_embedding"] is None


async def test_a_single_hop_question_costs_one_model_call_and_one_retrieval() -> None:
    engine = _Engine({QUESTION: ["a"]})
    with mocked_gateway([{"sub_questions": []}]) as gateway:
        decomposer = QueryDecomposer(gateway.assist(uses=[LLM_USE]))

        result = await decompose_and_retrieve(engine, decomposer, None, QUESTION)

    assert engine.queries == [QUESTION]
    assert result.diagnostics[DIAGNOSTIC] == {
        "sub_questions": [],
        "retrievals": 1,
        "llm_tokens": result.diagnostics[DIAGNOSTIC]["llm_tokens"],
    }


async def test_a_failing_sub_question_does_not_lose_the_question() -> None:
    engine = _Engine({QUESTION: ["a"], PARTS[1]: ["b"]}, failing=PARTS[0])
    with mocked_gateway([{"sub_questions": PARTS}]) as gateway:
        decomposer = QueryDecomposer(gateway.assist(uses=[LLM_USE]))

        result = await decompose_and_retrieve(engine, decomposer, None, QUESTION)

    assert {c.record_id for c in result.candidates} == {"a", "b"}
    assert result.diagnostics[DIAGNOSTIC]["failed"] == 1
    assert result.diagnostics[DIAGNOSTIC]["retrievals"] == 2


async def test_restatements_duplicates_and_blanks_are_never_retrieved() -> None:
    reply = {
        "sub_questions": [
            QUESTION.upper() + "?",  # the question again
            "who led the payments migration",
            "  who   led the payments migration ",  # the same sub-question again
            "",
            "where does Priya Raman live",
        ]
    }
    with mocked_gateway([reply]) as gateway:
        decomposer = QueryDecomposer(gateway.assist(uses=[LLM_USE]))
        assert await decomposer.sub_questions(QUESTION) == PARTS


async def test_a_reply_that_does_not_match_the_schema_is_discarded_whole() -> None:
    """``LLMAssist.structured`` validates against the schema, so a reply with a non-string
    sub-question is not partly used: the read falls back to the one retrieval."""
    engine = _Engine({QUESTION: ["a"]})
    with mocked_gateway([{"sub_questions": ["who led the payments migration", 42]}]) as gateway:
        decomposer = QueryDecomposer(gateway.assist(uses=[LLM_USE]))

        result = await decompose_and_retrieve(engine, decomposer, None, QUESTION)

    assert engine.queries == [QUESTION]
    assert result.diagnostics[DIAGNOSTIC]["sub_questions"] == []


async def test_the_sub_question_count_is_bounded() -> None:
    reply = {"sub_questions": [f"question number {i}" for i in range(9)]}
    with mocked_gateway([reply]) as gateway:
        decomposer = QueryDecomposer(gateway.assist(uses=[LLM_USE]), max_sub_questions=2)
        assert len(await decomposer.sub_questions(QUESTION)) == 2


async def test_a_gateway_that_fails_leaves_the_read_model_free() -> None:
    engine = _Engine({QUESTION: ["a"]})
    with mocked_gateway(failing=True) as gateway:
        decomposer = QueryDecomposer(gateway.assist(uses=[LLM_USE]))

        result = await decompose_and_retrieve(engine, decomposer, None, QUESTION)

    assert engine.queries == [QUESTION]
    assert result.diagnostics[DIAGNOSTIC]["sub_questions"] == []


def test_fusion_respects_the_limit_and_keeps_the_best_copy() -> None:
    decomposer = QueryDecomposer()
    first = [_candidate("a", score=0.2), _candidate("b", score=0.9)]
    second = [_candidate("b", score=0.4), _candidate("c", score=0.1)]

    fused = decomposer.fuse([first, second], limit=2)

    assert [c.record_id for c in fused] == ["b", "a"]
    assert fused[0].payload["fused_from_questions"] == 2
    assert all(c.score > 0 for c in fused)


def test_a_decomposer_refuses_an_unbounded_configuration() -> None:
    with pytest.raises(ValueError, match="bounded"):
        QueryDecomposer(max_sub_questions=0)


def test_a_decomposing_deployment_cannot_share_cached_bundles_with_one_that_is_not() -> None:
    """The bundle cache is keyed on the model profile, and the builder lists this use in it:
    without that, a bundle fused from sub-questions could be served to a deployment that does
    not decompose at all."""
    from memory_service.modules.context.builder import CACHED_MODEL_USES

    assert LLM_USE in CACHED_MODEL_USES
    with mocked_gateway() as gateway:
        decomposing = gateway.assist(uses=[LLM_USE]).cache_fingerprint(CACHED_MODEL_USES)
        plain = gateway.assist(uses=["reflection"]).cache_fingerprint(CACHED_MODEL_USES)
    assert decomposing and decomposing != plain
