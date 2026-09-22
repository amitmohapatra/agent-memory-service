"""``chunk_context``: a bounded subset of chunks (parts of split nodes first, then tables) gets a
model-written situating context after the deterministic header; ``text`` never changes and
any gateway failure leaves the chunks exactly as ``chunk_nodes`` built them."""

from __future__ import annotations

from pathlib import Path

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.documents import DocumentNode
from memory_service.domain.enums import Representation
from memory_service.modules.ingestion.chunking import (
    chunk_nodes,
    situate_chunks,
    situated_candidates,
)
from memory_service.modules.ingestion.hierarchy import estimate_tokens
from memory_service.modules.ingestion.service import IngestionService
from tests.integration.conftest import container, uow_factory  # noqa: F401
from tests.support_llm import mocked_gateway

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "acme_fy26_annual_report.md"


def _node(text: str, representation=Representation.PARAGRAPH, ordinal: int = 0) -> DocumentNode:
    return DocumentNode(
        document_id="d",
        document_version_id="v",
        tenant_id="t",
        representation=representation,
        ordinal=ordinal,
        depth=1,
        text=text,
        text_hash="x",
        token_estimate=estimate_tokens(text),
        section_path="Doc > Results",
        title="Segments" if representation is Representation.TABLE else None,
    )


def _long(topic: str, n: int = 80) -> str:
    return " ".join(f"Sentence {i} of the {topic} narrative explains figure {i}." for i in range(n))


def _table(rows: int = 40) -> str:
    return "| Segment | Revenue |\n|---|---|\n" + "\n".join(
        f"| Segment {i} | {i * 10} |" for i in range(rows)
    )


def _fixture_nodes() -> list[DocumentNode]:
    return [
        _node(_long("restructuring", 40), ordinal=0),
        _node("A short paragraph that fits in one chunk.", ordinal=1),
        _node(_table(), Representation.TABLE, ordinal=2),
    ]


def _contexts(*pairs: tuple[int, str]) -> dict:
    return {"contexts": [{"index": i, "context": c} for i, c in pairs]}


def test_candidates_are_split_parts_first_then_tables() -> None:
    nodes = _fixture_nodes()
    chunks = chunk_nodes(nodes, document_title="Doc", max_tokens=100, overlap_tokens=10)
    split = [i for i, c in enumerate(chunks) if c.node_id == nodes[0].node_id]
    table = [i for i, c in enumerate(chunks) if c.node_id == nodes[2].node_id]
    assert len(split) > 2 and len(table) == 1
    assert situated_candidates(chunks, nodes, max_chunks=48) == [*split, *table]
    assert situated_candidates(chunks, nodes, max_chunks=2) == split[:2]


async def test_context_is_inserted_after_the_header_and_text_is_untouched() -> None:
    nodes = _fixture_nodes()
    chunks = chunk_nodes(nodes, document_title="Doc", max_tokens=100, overlap_tokens=10)
    reply = _contexts((0, "Opening of the restructuring narrative."), (1, "  Second   part. "))
    with mocked_gateway([reply]) as gw:
        assist = gw.assist(uses=["chunk_context"])
        out = await situate_chunks(assist, chunks, nodes, document_title="Doc")
        prompt = gw.prompts()[0]
    assert prompt["model"] == "test/fast"
    user = prompt["messages"][1]["content"]
    assert user.startswith("Document: Doc") and "### Chunk 0" in user
    assert "Surrounding text: " in user and "[…the chunk…]" in user
    assert out[0].text == chunks[0].text and out[0].text_hash == chunks[0].text_hash
    head = chunks[0].contextual_text[: -len(chunks[0].text)]
    assert head.startswith("Document: Doc\nSection: Doc > Results\n") and head.endswith("\n\n")
    assert out[0].contextual_text == (
        head + "Context: Opening of the restructuring narrative.\n\n" + chunks[0].text
    )
    assert "Context: Second part.\n\n" in out[1].contextual_text
    assert [c.contextual_text for c in out[2:]] == [c.contextual_text for c in chunks[2:]]


async def test_gateway_failure_keeps_deterministic_chunks() -> None:
    nodes = _fixture_nodes()
    chunks = chunk_nodes(nodes, document_title="Doc", max_tokens=100, overlap_tokens=10)
    with mocked_gateway(failing=True) as gw:
        out = await situate_chunks(
            gw.assist(uses=["chunk_context"]), chunks, nodes, document_title="Doc"
        )
        assert gw.route.call_count >= 1
    assert out == chunks


async def test_flag_off_never_calls_the_model() -> None:
    nodes = _fixture_nodes()
    chunks = chunk_nodes(nodes, document_title="Doc", max_tokens=100, overlap_tokens=10)
    with mocked_gateway([_contexts((0, "x"))]) as gw:
        out = await situate_chunks(
            gw.assist(uses=["summaries"]), chunks, nodes, document_title="Doc"
        )
        assert gw.route.call_count == 0
    assert out == chunks


async def test_bounds_cap_chunks_batches_and_windows() -> None:
    nodes = [_node(_long(f"topic {i}", 30), ordinal=i) for i in range(40)]
    nodes.append(_node(_table(), Representation.TABLE, ordinal=40))
    chunks = chunk_nodes(nodes, document_title="Doc", max_tokens=100, overlap_tokens=10)
    assert len(situated_candidates(chunks, nodes, max_chunks=10_000)) > 48
    with mocked_gateway([_contexts((0, "ctx"), (7, "ctx"), (9, "ignored"), (-1, "ignored"))]) as gw:
        out = await situate_chunks(
            gw.assist(uses=["chunk_context"]), chunks, nodes, document_title="Doc"
        )
        assert gw.route.call_count == 6
        for prompt in gw.prompts():
            user = prompt["messages"][1]["content"]
            assert user.count("### Chunk ") == 8
            for section in user.split("### Chunk ")[1:]:
                surrounding = section.split("Surrounding text: ", 1)[1].split("\nChunk text: ")[0]
                assert len(surrounding) <= 1500 + len("[…the chunk…]")
    changed = [i for i, c in enumerate(out) if c.contextual_text != chunks[i].contextual_text]
    assert len(changed) == 12
    assert all(c.node_id != nodes[-1].node_id for c in out if "Context: " in c.contextual_text)


async def test_parse_job_persists_situated_chunks(container, uow_factory) -> None:  # noqa: F811
    ctx = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
    native = container.services["ingestion"]
    async with uow_factory() as uow:
        ack = await native.accept_file(
            uow,
            ctx,
            filename="acme_fy26_annual_report.md",
            media_type="text/markdown",
            data=FIXTURE.read_bytes(),
            title="ACME FY26",
        )
        await uow.commit()
    reply = _contexts((0, "Situated by the model."), (1, "Situated by the model."))
    with mocked_gateway([reply]) as gw:
        ingestion = IngestionService(
            uow_factory,
            container.services["authz"],
            container.document_parser,
            container.blob,
            settings=container.tuning.documents,
            file_bucket=container.settings.blob.file_bucket,
            assist=gw.assist(uses=["chunk_context"]),
        )
        await ingestion.parse_document("acme", ack.document_id)
        assert gw.route.call_count == 1
    async with uow_factory() as uow:
        chunks = await uow.documents.list_chunks("acme", ack.document_id)
    situated = [c for c in chunks if "\nContext: Situated by the model.\n\n" in c.contextual_text]
    assert len(situated) == 2 and len(chunks) > 2
    table = next(c for c in situated if c.text.startswith("| Segment"))
    assert table.contextual_text.startswith("Document: ACME FY26 Annual Report\n")
    assert table.contextual_text.endswith("\n\n" + table.text)
