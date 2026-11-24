"""Which learned procedures a task is about (``modules/tools/matching.py``): the pattern's
words first, then, for a procedure they miss, the meaning - the cosine of the task's vector to
the pattern's in the space every query is searched in, at that encoder's own floor."""

# ruff: noqa: RUF001 - literal multilingual fixtures intentionally use non-Latin letters.

from __future__ import annotations

import asyncio
import re

import pytest

from memory_service.adapters.models.embeddings import HashEmbedding
from memory_service.adapters.models.sparse import Bm25SparseEncoder
from memory_service.config.constants import FROZEN_MODELS
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.script import Script
from memory_service.domain.tools import StoredProcedure, ToolHints
from memory_service.modules.context.sections import Pinned, ToolsRequest
from memory_service.modules.rag.indexer import Indexer
from memory_service.modules.rag.spaces import DenseSpace, DenseSpaces
from memory_service.modules.retrieval.engine import QueryVectors
from memory_service.modules.retrieval.router import QueryRouter
from memory_service.modules.tools.hints import ToolHintsService
from memory_service.modules.tools.matching import (
    PROCEDURE_MATCH,
    ProcedureMatcher,
    rank_procedures,
)
from memory_service.modules.tools.patterns import similarity, task_pattern
from memory_service.ports.search import VectorName
from tests.unit.test_context_datapath import CTX, _builder

pytestmark = pytest.mark.unit

#: What a word means, for the stand-in encoder: a paraphrase in other words (or another
#: language) lands on the same axes, an unrelated task on others.
_CONCEPTS = {
    "mfa": ("mfa", "multi-factor", "two-factor", "authentication", "authenticator", "मल्टी-फ़ैक्टर"),
    "lockout": ("locked-out", "locked", "lockout", "lockd", "bloqueado", "लॉक"),
    "user": ("user", "usr", "usuario", "यूज़र"),
    "reset": ("reset", "re-enrol", "restablece", "रीसेट"),
    "pos": ("pos", "terminal", "till"),
    "store": ("store", "store's", "s12"),
    "stock": ("stock", "order", "reorder"),
}
_AXES = list(_CONCEPTS)
_WORDS = re.compile(r"[\s.,!?\u0964]+")


class _Meaning:
    """A dense encoder with a meaning: one axis per concept, so cosines are what a reader
    expects. Counts what it encodes."""

    info = HashEmbedding.info
    dimension = len(_AXES)
    relevance_floor = 0.0

    def __init__(self, procedure_cosine: float = 0.6, name: str = "meaning") -> None:
        self.procedure_cosine = procedure_cosine
        self.name = name
        self.documents: list[str] = []
        self.queries: list[str] = []

    def _vector(self, text: str) -> list[float]:
        words = [w for w in _WORDS.split(text.casefold()) if w]
        return [float(sum(w in _CONCEPTS[axis] for w in words)) for axis in _AXES]

    async def embed_documents(self, texts):
        self.documents.extend(texts)
        return [self._vector(t) for t in texts]

    async def embed_query(self, text):
        self.queries.append(text)
        return self._vector(text)

    def fingerprint(self) -> str:
        return f"{self.name}-v1"


def _indexer(*spaces: DenseSpace) -> Indexer:
    return Indexer(None, None, DenseSpaces(spaces), Bm25SparseEncoder(), None)  # type: ignore[arg-type]


def _matcher(encoder) -> ProcedureMatcher:
    return ProcedureMatcher(_indexer(DenseSpace(VectorName.DENSE_ML, encoder)))


def _procedure(task: str, procedure_id: str, **kw) -> StoredProcedure:
    return StoredProcedure(
        procedure_id=procedure_id,
        tenant_id="acme",
        scope_key="k",
        pattern=task_pattern(task),
        steps=[{"ordinal": 0, "tool": "iam-reset_mfa"}],
        status="active",
        **kw,
    )


MFA = _procedure("reset multi-factor authentication for a locked-out user", "proc_mfa")
POS = _procedure("reboot the POS terminal of a store", "proc_pos")
STOCK = _procedure("order more stock for S12", "proc_stock")
PROCEDURES = [MFA, POS, STOCK]


@pytest.mark.parametrize(
    "task",
    [
        "MFA lockout again.",
        "Re-enrol two-factor for a user who is locked out.",
        "reset mfa for lockd out usr",
    ],
)
async def test_a_paraphrase_the_words_miss_is_matched_by_its_meaning(task: str) -> None:
    assert similarity(task_pattern(task), MFA.pattern) < PROCEDURE_MATCH, "the words miss it"
    assert rank_procedures(task, PROCEDURES, 3) == []
    assert await _matcher(_Meaning()).rank(task, PROCEDURES, 3) == [MFA]


@pytest.mark.parametrize(
    "task", ["Reset a store's POS terminal.", "Order more stock for S12.", "What time is it?"]
)
async def test_an_unrelated_task_is_not_matched_to_the_procedure(task: str) -> None:
    assert MFA not in await _matcher(_Meaning()).rank(task, PROCEDURES, 3)


async def test_a_word_match_is_the_fast_path_and_reads_no_model() -> None:
    """The words find the procedure: the ranking is the words' alone, and nothing is encoded
    (``k`` already filled), exactly as before the meaning was read."""
    encoder = _Meaning()
    task = "reset multi-factor authentication for a locked-out user"
    assert await _matcher(encoder).rank(task, PROCEDURES, 1) == [MFA]
    assert encoder.queries == [] and encoder.documents == []


async def test_a_word_match_outranks_a_meaning_match() -> None:
    worded = _procedure("reset mfa for a locked-out user", "proc_worded")
    task = "reset mfa for a locked-out user now"
    assert similarity(task_pattern(task), worded.pattern) >= PROCEDURE_MATCH
    assert similarity(task_pattern(task), MFA.pattern) < PROCEDURE_MATCH
    ranked = await _matcher(_Meaning()).rank(task, [MFA, worded, POS], 3)
    assert ranked == [worded, MFA]


async def test_without_a_calibrated_floor_only_the_words_count() -> None:
    """The stand-in encoder has no calibrated cosine (``HashEmbedding.procedure_cosine``),
    and a matcher with no encoder at all reads words alone."""
    assert HashEmbedding().procedure_cosine == 0.0
    encoder = _Meaning(procedure_cosine=0.0)
    assert await _matcher(encoder).rank("MFA lockout again.", PROCEDURES, 3) == []
    assert encoder.queries == [] and encoder.documents == []
    assert await ProcedureMatcher(None).rank("MFA lockout again.", PROCEDURES, 3) == []


def test_the_floor_belongs_to_the_encoder_that_measured_it() -> None:
    assert FROZEN_MODELS.dense_ml.procedure_cosine > 0, "the space every query reads"
    assert FROZEN_MODELS.dense.procedure_cosine > 0, "the single-encoder arm's"


async def test_a_pattern_is_encoded_once_and_the_task_vector_in_hand_is_reused() -> None:
    """The first request encodes each pattern the words miss, together; the next encodes
    none. The task's own vector comes from the request (the retrieval's), not the model."""
    encoder = _Meaning()
    matcher = _matcher(encoder)
    task = "MFA lockout again."
    vectors = QueryVectors(
        dense={VectorName.DENSE_ML: encoder._vector(task)}, sparse=None, script=Script.LATIN
    )
    assert await matcher.rank(task, PROCEDURES, 3, vectors=vectors) == [MFA]
    assert sorted(encoder.documents) == sorted(p.pattern for p in PROCEDURES)
    assert encoder.queries == [], "the task's vector was in hand"
    encoder.documents.clear()
    assert await matcher.rank("Re-enrol two-factor for a locked user.", PROCEDURES, 3) == [MFA]
    assert encoder.documents == [], "every pattern's vector is kept"
    assert encoder.queries == ["Re-enrol two-factor for a locked user."]


async def test_the_task_vectors_may_still_be_on_their_way() -> None:
    """A context build reads the procedures under its retrieval: the vectors arrive when the
    retrieval has encoded the query, and none are encoded twice."""
    encoder = _Meaning()
    matcher = _matcher(encoder)
    task = "MFA lockout again."
    arriving: asyncio.Future[QueryVectors | None] = asyncio.get_running_loop().create_future()
    ranking = asyncio.ensure_future(matcher.rank(task, PROCEDURES, 3, vectors=arriving))
    await asyncio.sleep(0)
    assert not ranking.done(), "it waits for the retrieval's vectors"
    arriving.set_result(
        QueryVectors(
            dense={VectorName.DENSE_ML: encoder._vector(task)}, sparse=None, script=Script.LATIN
        )
    )
    assert await ranking == [MFA] and encoder.queries == []


async def test_the_cache_is_bounded_and_keyed_by_the_encoder() -> None:
    encoder = _Meaning()
    matcher = ProcedureMatcher(_indexer(DenseSpace(VectorName.DENSE_ML, encoder)), capacity=2)
    await matcher.rank("MFA lockout again.", PROCEDURES, 3)
    assert len(matcher._patterns) == 2
    assert all(key.startswith("meaning-v1:") for key in matcher._patterns)


async def test_another_script_is_read_in_the_space_every_query_is_searched_in() -> None:
    """A Hindi task has no English-space vector: the match reads the multilingual space's,
    for the task and the pattern alike, never the English specialist."""
    english, multilingual = _Meaning(name="en"), _Meaning(name="ml")
    matcher = ProcedureMatcher(
        _indexer(
            DenseSpace(VectorName.DENSE_EN, english, query_scripts=frozenset({Script.LATIN})),
            DenseSpace(VectorName.DENSE_ML, multilingual),
        )
    )
    hindi = _procedure("लॉक हुए यूज़र का मल्टी-फ़ैक्टर ऑथेंटिकेशन रीसेट करना", "proc_hi")
    task = "फिर से MFA लॉकआउट, यूज़र लॉक है।"
    assert similarity(task_pattern(task), hindi.pattern) < PROCEDURE_MATCH
    assert await matcher.rank(task, [hindi, POS], 3) == [hindi]
    assert english.queries == [] and english.documents == []
    assert multilingual.documents == [hindi.pattern, POS.pattern]


class _Uow:
    def __init__(self, visible: list[StoredProcedure]) -> None:
        self.visible_procedures = visible
        self.procedures = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def visible(self, *_: object, **__: object) -> list[StoredProcedure]:
        return self.visible_procedures


async def test_the_context_skills_and_the_hints_plan_share_one_matcher() -> None:
    """``ToolHintsService.procedures`` is what the context's skills section and the hints'
    plan both read: a paraphrase finds the learned skill through either."""
    encoder = _Meaning()
    service = ToolHintsService(
        lambda: _Uow(PROCEDURES),  # type: ignore[arg-type,return-value]
        None,  # type: ignore[arg-type]
        None,
        _matcher(encoder),
    )
    ctx = MemoryExecutionContext(tenant_id="acme", user_id="u1")
    found = await service.procedures(ctx, "A user can't get past the MFA prompt", ["k"], k=3)
    assert found == [MFA]
    assert (
        await ToolHintsService(
            lambda: _Uow(PROCEDURES),  # type: ignore[arg-type,return-value]
            None,  # type: ignore[arg-type]
            None,
        ).procedures(ctx, "A user can't get past the MFA prompt", ["k"], k=3)
        == []
    ), "without a matcher, words alone"


class _Gathering:
    """The pinned sections, recording the query vectors the procedures read was handed."""

    def __init__(self) -> None:
        self.seen: list[QueryVectors | None] = []

    async def gather(self, ctx, query, visibility, *, procedures, vectors=None):
        assert isinstance(vectors, asyncio.Future)
        self.seen.append(await vectors)
        return Pinned()

    async def tools(self, ctx, query, request, visibility, *, memories, profile, vectors):
        return ToolHints()


@pytest.mark.parametrize("encodes", [True, False])
async def test_a_context_build_hands_the_skills_read_the_retrieval_s_own_vectors(
    encodes: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The skills section is read under the retrieval: it waits for the vectors the
    retrieval encoded (``on_encoded``), never encodes the query a second time, and a
    retrieval that reported none still lets it finish (with the result's own)."""
    # the standing rules' read starts on the same hand-over; this fixture has no store for it
    monkeypatch.setattr("memory_service.modules.context.builder.STANDING", False)
    builder = _builder()
    sections = _Gathering()
    builder.sections = sections  # type: ignore[assignment]
    encoded = QueryVectors(dense={VectorName.DENSE_ML: [1.0]}, sparse=None, script=Script.LATIN)
    original = builder.engine.retrieve

    async def retrieve(ctx, query, **kwargs):
        if encodes:
            kwargs["on_encoded"](QueryRouter().route(query, has_thread=False), encoded)
        result = await original(ctx, query, **kwargs)
        result.query_vectors = encoded if encodes else None
        return result

    builder.engine.retrieve = retrieve  # type: ignore[method-assign]
    await builder.build(CTX, "MFA lockout again.", tools=ToolsRequest(available=["iam-reset_mfa"]))
    assert sections.seen == [encoded if encodes else None]
    await builder.close()
