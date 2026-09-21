"""Proof that removing seven retrieval flags removes no capability.

Each flag names something a system must be able to do. The argument for deleting it is that
another mechanism — one that is on by default — already does it. That argument is worthless
unless the covering mechanism is exercised on a problem hard enough to need it, so every test
here runs with the candidate flags **off** and asserts the capability still works.

The corpus is deliberately adversarial: the subject is named once and referred to
anaphorically afterwards, the multi-hop answer is split across two documents, and a
distractor document repeats the vocabulary without the facts.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text

from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.jobs.registry import register_handlers
from tests.integration.conftest import TABLES

#: Indexing the distractor corpus dominates the runtime, so the per-test budget has to cover
#: it. These are evaluation tests, not unit tests.
pytestmark = [pytest.mark.integration, pytest.mark.timeout(900)]

#: Enough unrelated documents that retrieval has to discriminate rather than return the lot.
#: Three was not: every query returned every chunk and every assertion passed vacuously.
DISTRACTOR_DOCS = 150

#: The subject is named in the opening section and never again — every later reference is
#: "the city" / "the company". A chunk-local extractor cannot recover it.
ACME = """# Acme Industrial Report

## Overview
Acme Industrial is headquartered in Dortmund and was founded in 1961.

## Operations
The company operates four plants. The city hosts its largest facility, which employs
3,500 people and produces gearboxes for wind turbines.

## Logistics
Freight moves through the regional hub operated by Westfalen Logistik.
"""

TRANSIT = """# Westfalen Logistik Profile

## Mandate
Westfalen Logistik runs freight terminals across North Rhine-Westphalia.
Its chief executive is Maria Bergmann.
"""

#: Same vocabulary, none of the facts. Present so a match cannot be luck.
DISTRACTOR = """# Industrial Gearbox Market Review

## Summary
Gearbox producers employ thousands of people across several cities and rely on regional
logistics operators for freight. Wind turbine demand shapes plant utilisation.
"""


async def _ingest(container, uow_factory, ctx, name: str, body: str) -> str:
    async with uow_factory() as uow:
        ack = await container.services["ingestion"].accept_file(
            uow,
            ctx,
            filename=f"{name}.md",
            media_type="text/markdown",
            data=body.encode(),
            title=name,
        )
        await uow.commit()
    return ack.document_id


@pytest.fixture
async def corpus(container, uow_factory):
    """The adversarial documents, buried in a real corpus, with every candidate flag OFF.

    An earlier version of this fixture indexed three documents. Every query returned every
    chunk, because with three chunks there is nothing to discriminate against — the tests
    passed and proved nothing, which is the same flaw as a gate that reads 1.0 against its
    own fixtures. Retrieval capability only means something when retrieval has to choose,
    so the targets are hidden among several hundred unrelated documents and the assertions
    are about **rank**, not presence.
    """
    # These tests are the coverage argument for seven flags that have since been *removed*:
    # raptor, graphrag_global, graph_ppr, colbert, pageindex, minicoil and late_chunking. Each
    # named a capability something already-on provides, and each test below demonstrates the
    # capability surviving without it. The flags are gone, so there is nothing left to assert
    # off — but the proofs must keep running, or the removal stops being justified by anything
    # and becomes a claim in a document.
    cfg = container.settings.retrieval
    removed = {
        "raptor",
        "graphrag_global",
        "graph_ppr",
        "colbert",
        "pageindex",
        "minicoil",
        "late_chunking",
    }
    assert not (removed & set(type(cfg).model_fields)), (
        "a removed flag has come back; these tests prove its capability is covered without it"
    )

    dataset = Path(__file__).resolve().parents[2] / "benchmark" / "data" / "scifact.json"
    if not dataset.is_file():
        pytest.skip("distractor corpus missing — run `make bench-external-prepare`")
    distractors = json.loads(dataset.read_text())["corpus"][:DISTRACTOR_DOCS]

    async with container.database.engine.begin() as conn:
        await conn.execute(text("TRUNCATE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE"))
    register_handlers(container)
    ctx = MemoryExecutionContext(
        tenant_id="acme", user_id=f"u-{uuid.uuid4().hex[:6]}", workspace_id="ws1"
    )
    ids = {
        "acme": await _ingest(container, uow_factory, ctx, "acme-report", ACME),
        "transit": await _ingest(container, uow_factory, ctx, "westfalen-profile", TRANSIT),
        "distractor": await _ingest(container, uow_factory, ctx, "market-review", DISTRACTOR),
    }
    for doc in distractors:
        await _ingest(container, uow_factory, ctx, doc["id"], f"# {doc['title']}\n\n{doc['text']}")
    await container.tasks.drain()
    await container.tasks.drain()
    return ctx, ids


async def _texts(container, ctx, query: str, *, limit: int = 10) -> list[str]:
    result = await container.services["retrieval"].retrieve(
        ctx, query, kinds=("chunk",), limit=limit
    )
    return [c.text for c in result.candidates]


async def test_anaphora_across_nodes_is_retrievable_without_late_chunking(
    container, corpus
) -> None:
    """`late_chunking`'s retrieval-time claim, covered by ancestor-entity propagation.

    "The city hosts its largest facility" never names Dortmund. Late chunking would carry the
    referent in the vector; propagating ancestor entities into the contextual header carries
    it in the *text*, so BM25 benefits too.
    """
    ctx, _ = corpus
    # The query is the bare proper noun. It appears nowhere in the target chunk, nowhere in
    # its section path ("Acme Industrial Report > Operations") and nowhere in the file title.
    # Only document-salience propagation can put it in the indexed text. An earlier version
    # of this test used "Dortmund plant employees", which passed with the feature reverted
    # because "employs" lexically matches "employees" — it proved nothing.
    found = await _texts(container, ctx, "Dortmund", limit=5)
    assert any("3,500 people" in t for t in found[:5]), (
        f"the anaphoric chunk did not reach the top 5 among {DISTRACTOR_DOCS} distractors; "
        f"got {[t[:40] for t in found[:5]]}"
    )


async def test_multi_hop_across_documents_without_graph_ppr(container, corpus) -> None:
    """`graph_ppr`'s claim, covered by graph retrieval at hops=2.

    Acme -> Westfalen Logistik -> Maria Bergmann. No single chunk contains both ends.
    """
    ctx, _ = corpus
    result = await container.services["retrieval"].retrieve(
        ctx, "who leads the freight operator used by Acme Industrial?", limit=10
    )
    joined = " ".join(c.text for c in result.candidates)
    assert "Westfalen Logistik" in joined, "the first hop was not retrieved"
    assert "Maria Bergmann" in joined, "the second hop was not retrieved"


async def test_corpus_level_question_without_raptor_or_graphrag_global(container, corpus) -> None:
    """Their claim, covered by build_summaries(): section/document summaries are indexed."""
    ctx, _ = corpus
    result = await container.services["retrieval"].retrieve(
        ctx, "what is the Acme Industrial report about overall?", limit=10
    )
    kinds = {c.payload.get("kind") for c in result.candidates}
    assert "summary" in kinds, f"no summary record surfaced; kinds were {kinds}"


async def test_long_document_navigation_without_pageindex(container, corpus) -> None:
    """`pageindex`'s claim, covered by the hierarchy: every chunk carries its section path."""
    ctx, _ = corpus
    result = await container.services["retrieval"].retrieve(
        ctx, "Acme logistics freight hub", kinds=("chunk",), limit=10
    )
    paths = [c.payload.get("section_path") or "" for c in result.candidates]
    assert any("Logistics" in p for p in paths), f"section path missing; got {paths}"


async def test_precision_over_a_distractor_without_colbert(container, corpus) -> None:
    """`colbert`'s claim, covered by the cross-encoder reranker.

    The distractor shares the vocabulary and holds none of the facts; it must not outrank
    the document that answers the question.
    """
    ctx, ids = corpus
    result = await container.services["retrieval"].retrieve(
        ctx, "how many people work at the Acme gearbox plant?", kinds=("chunk",), limit=5
    )
    assert result.candidates, "nothing retrieved"
    top = result.candidates[0]
    assert top.payload.get("document_id") != ids["distractor"], (
        "the vocabulary-matching distractor outranked the answer"
    )


async def test_definitions_and_footnotes_still_expand(container, corpus) -> None:
    """The expansion kinds that are always on, and that answer-time referent resolution
    depends on — the other half of what late chunking is credited with."""
    cfg = container.settings.retrieval
    assert cfg.definition_expansion and cfg.parent_expansion and cfg.neighbor_expansion
    ctx, _ = corpus
    result = await container.services["retrieval"].retrieve(
        ctx, "the city largest facility", kinds=("chunk",), limit=8
    )
    joined = " ".join(c.text for c in result.candidates)
    assert "Dortmund" in joined or "Acme Industrial" in joined, (
        "expansion did not bring the referent into the returned context"
    )

