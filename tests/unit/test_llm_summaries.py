"""``summaries``: abstractive document/section summaries at index time and an abstractive
rolling conversation summary in the ContextBuilder, each bounded and falling back to the
extractive text whenever the model cannot help."""

from __future__ import annotations

from pathlib import Path

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.conversation import Message
from memory_service.domain.documents import Chunk, DocumentNode
from memory_service.domain.enums import MessageKind, MessageRole, Representation
from memory_service.domain.ids import content_hash, new_id
from memory_service.modules.context.builder import abstractive_rolling_summary, rolling_summary
from memory_service.modules.context.summaries import (
    SOURCE_CHARS,
    abstractive_summaries,
    build_summaries,
)
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.rag.indexer import Indexer
from tests.integration.conftest import container, uow_factory  # noqa: F401
from tests.support_llm import mocked_gateway

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "acme_fy26_annual_report.md"
TITLE = "ACME FY26"


def _node(
    representation: Representation,
    *,
    parent: str | None = None,
    title: str | None = None,
    ordinal: int = 0,
) -> DocumentNode:
    return DocumentNode(
        document_id="d",
        document_version_id="v",
        tenant_id="t",
        representation=representation,
        parent_id=parent,
        ordinal=ordinal,
        depth=0 if parent is None else 1,
        title=title,
    )


def _chunk(node: DocumentNode, text: str) -> Chunk:
    return Chunk(
        node_id=node.node_id,
        document_id="d",
        document_version_id="v",
        tenant_id="t",
        ordinal=0,
        text=text,
        text_hash=content_hash(text),
        contextual_text=text,
    )


def _tree(sections: int = 1) -> tuple[list[DocumentNode], list[Chunk]]:
    doc = _node(Representation.DOCUMENT)
    nodes, chunks = [doc], []
    for i in range(sections):
        section = _node(Representation.SECTION, parent=doc.node_id, title=f"S{i}", ordinal=i)
        para = _node(Representation.PARAGRAPH, parent=section.node_id, ordinal=0)
        nodes += [section, para]
        chunks.append(
            _chunk(
                para,
                f"Section {i} reports that revenue fell to EUR {100 + i} million. "
                f"Adjusted EBITDA rose to EUR {90 + i} million on cost savings. "
                "Management expects the trend to continue.",
            )
        )
    return nodes, chunks


def _msg(content: str, role=MessageRole.USER, kind=MessageKind.VISIBLE) -> Message:
    return Message(
        message_id=new_id("message"),
        tenant_id="t",
        thread_id="thr",
        session_id="s",
        turn_id="u",
        sequence=1,
        role=role,
        kind=kind,
        content=content,
        content_hash=content_hash(content),
        author_principal="user:u1",
    )


async def test_document_summaries_are_rewritten_from_extractive_and_source() -> None:
    nodes, chunks = _tree()
    doc, section = nodes[0], nodes[1]
    extractive = build_summaries(nodes, chunks, title=TITLE)
    with mocked_gateway(['{"summary": " Revenue fell while EBITDA rose. "}']) as gw:
        out = await abstractive_summaries(
            gw.assist(uses=["summaries"]), nodes, chunks, extractive, title=TITLE
        )
        prompts = gw.prompts()
    assert len(prompts) == 2 and all(p["model"] == "test/strong" for p in prompts)
    first = prompts[0]["messages"]
    assert "at most 900 characters" in first[0]["content"]
    assert first[1]["content"].startswith(f"Document: {TITLE}\nSection: {TITLE}\n")
    assert "Extractive summary:\n" in first[1]["content"]
    assert "Source text:\n" + chunks[0].text in first[1]["content"]
    assert "at most 500 characters" in prompts[1]["messages"][0]["content"]
    assert out[doc.node_id] == f"{TITLE}: Revenue fell while EBITDA rose."
    assert out[section.node_id] == "S0: Revenue fell while EBITDA rose."
    assert set(out) == set(extractive)


async def test_document_summaries_fall_back_on_gateway_failure() -> None:
    nodes, chunks = _tree()
    extractive = build_summaries(nodes, chunks, title=TITLE)
    with mocked_gateway(failing=True) as gw:
        out = await abstractive_summaries(
            gw.assist(uses=["summaries"]), nodes, chunks, extractive, title=TITLE
        )
        assert gw.route.call_count >= 1
    assert out == extractive


async def test_document_summaries_flag_off_never_calls_the_model() -> None:
    nodes, chunks = _tree()
    extractive = build_summaries(nodes, chunks, title=TITLE)
    with mocked_gateway(['{"summary": "nope"}']) as gw:
        out = await abstractive_summaries(
            gw.assist(uses=["query_expansion"]), nodes, chunks, extractive, title=TITLE
        )
        assert gw.route.call_count == 0
    assert out == extractive


async def test_document_summaries_bounds_nodes_source_and_length() -> None:
    nodes, chunks = _tree(sections=30)
    chunks[3] = chunks[3].model_copy(update={"text": " ".join([chunks[3].text] * 60)})
    extractive = build_summaries(nodes, chunks, title=TITLE)
    assert len(extractive) == 31
    too_long = {"summary": "x" * 751}
    empty = {"summary": "   "}
    good = {"summary": "Concise."}
    with mocked_gateway([good, too_long, empty, good]) as gw:
        out = await abstractive_summaries(
            gw.assist(uses=["summaries"]), nodes, chunks, extractive, title=TITLE
        )
        prompts = gw.prompts()
    assert len(prompts) == 24
    # largest source first: the document, then the padded section
    assert prompts[0]["messages"][1]["content"].startswith(f"Document: {TITLE}\nSection: {TITLE}")
    assert prompts[1]["messages"][1]["content"].startswith(f"Document: {TITLE}\nSection: S3")
    source = prompts[0]["messages"][1]["content"].split("Source text:\n", 1)[1]
    assert len(source) <= SOURCE_CHARS
    rewritten = {nid for nid in out if out[nid] != extractive[nid]}
    assert nodes[0].node_id in rewritten and len(rewritten) == 22
    assert out[nodes[0].node_id] == f"{TITLE}: Concise."
    doc_id, s3 = nodes[0].node_id, nodes[7].node_id
    assert out[s3] == extractive[s3]  # 751 chars > 1.5 x 500
    assert sum(1 for nid in extractive if out[nid] == extractive[nid]) == 9
    assert doc_id in out


async def test_rolling_summary_is_rewritten_or_kept() -> None:
    older = [
        _msg("Let's review the FY26 numbers. There is a lot to cover."),
        _msg("Sure, starting with revenue.", role=MessageRole.ASSISTANT),
        _msg("internal reasoning", kind=MessageKind.INTERNAL),
    ]
    extractive = rolling_summary(older)
    with mocked_gateway(
        ['{"summary": "The user asked to review FY26, starting with revenue."}']
    ) as gw:
        out = await abstractive_rolling_summary(gw.assist(uses=["summaries"]), older, extractive)
        user = gw.prompts()[0]["messages"][1]["content"]
    assert out == "The user asked to review FY26, starting with revenue."
    assert user.startswith(f"Deterministic digest:\n{extractive}\n\nEarlier conversation:\n")
    assert "user: Let's review the FY26 numbers. There is a lot to cover." in user
    assert "internal reasoning" not in user
    with mocked_gateway(failing=True) as gw:
        assert (
            await abstractive_rolling_summary(gw.assist(uses=["summaries"]), older, extractive)
            == extractive
        )
        assert gw.route.call_count >= 1
    with mocked_gateway(['{"summary": "nope"}']) as gw:
        assert (
            await abstractive_rolling_summary(gw.assist(uses=["reflection"]), older, extractive)
            == extractive
        )
        assert gw.route.call_count == 0
    huge = [_msg(f"Turn {i}: " + "detail " * 200) for i in range(20)]
    with mocked_gateway(['{"summary": "' + "y" * 901 + '"}']) as gw:
        digest = rolling_summary(huge)
        assert (
            await abstractive_rolling_summary(gw.assist(uses=["summaries"]), huge, digest) == digest
        )
        source = gw.prompts()[0]["messages"][1]["content"].split("Earlier conversation:\n", 1)[1]
        assert len(source) == SOURCE_CHARS and source.rstrip().endswith("detail")


async def test_indexer_and_builder_use_the_model(container, uow_factory) -> None:  # noqa: F811
    register_handlers(container)
    ctx = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
    async with uow_factory() as uow:
        ack = await container.services["ingestion"].accept_file(
            uow,
            ctx,
            filename="acme_fy26_annual_report.md",
            media_type="text/markdown",
            data=FIXTURE.read_bytes(),
            title=TITLE,
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()
    async with uow_factory() as uow:
        nodes = await uow.documents.list_nodes("acme", ack.document_id)
        before = await uow.documents.node_summaries("acme", [n.node_id for n in nodes])
    assert before and all(":" in s for s in before.values())
    with mocked_gateway(['{"summary": "ACME grew EBITDA while revenue fell."}']) as gw:
        indexer = Indexer(
            uow_factory,
            container.search,
            container.embedding,
            container.sparse,
            None,
            assist=gw.assist(uses=["summaries"]),
        )
        await indexer.index_document("acme", ack.document_id, force=True)
        assert 0 < gw.route.call_count <= 24
    async with uow_factory() as uow:
        after = await uow.documents.node_summaries("acme", list(before))
    doc_node = next(n for n in nodes if n.representation is Representation.DOCUMENT)
    assert after[doc_node.node_id] == f"{TITLE}: ACME grew EBITDA while revenue fell."
    assert set(after) == set(before)

    thread = new_id("thread")
    tctx = ctx.model_copy(
        update={"thread_id": thread, "session_id": new_id("session"), "turn_id": new_id("turn")}
    )
    conversation = container.services["conversation"]
    async with uow_factory() as uow:
        for i in range(8):
            await conversation.append_message(
                uow, tctx, role=MessageRole.USER, content=f"Turn {i}: " + "detail " * 60
            )
        await uow.commit()
    builder = container.services["context_builder"]
    builder.cfg = container.tuning.context.model_copy(update={"conversation_token_budget": 200})
    with mocked_gateway(['{"summary": "Eight turns of detail about the review."}']) as gw:
        builder.assist = gw.assist(uses=["summaries"])
        bundle = await builder.build(tctx, "what did I say earlier in this thread?")
        assert gw.route.call_count == 1
    assert bundle.conversation.summary == "Eight turns of detail about the review."
    assert len(bundle.conversation.message_ids) < 8
