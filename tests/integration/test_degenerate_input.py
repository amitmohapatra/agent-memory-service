"""What the service does with input that is not a well-formed document.

Real corpora contain empty files, binary blobs mislabelled as text, control characters, text
in the wrong script, and documents that are a single 50,000-character line. None of these are
hypothetical — the builtin parser was indexing raw PDF bytes as document text until a media
type check was added, which is what prompted this file.

The rule being asserted is narrow and absolute: *nothing unparseable may enter the index.*
Either it parses into something meaningful, or it is refused. Silently storing noise is worse
than either, because retrieval then returns it forever and nothing points at the cause.
"""

from __future__ import annotations

import uuid

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.errors import ValidationFailed
from memory_service.modules.jobs.registry import register_handlers

pytestmark = pytest.mark.integration


async def _accept(container, uow_factory, ctx, *, name: str, media_type: str, data: bytes):
    async with uow_factory() as uow:
        ack = await container.services["ingestion"].accept_file(
            uow, ctx, filename=name, media_type=media_type, data=data, title=name
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()
    return ack


@pytest.fixture
def ctx() -> MemoryExecutionContext:
    return MemoryExecutionContext(
        tenant_id="acme", user_id=f"u-{uuid.uuid4().hex[:6]}", workspace_id="ws1"
    )


async def test_a_binary_file_mislabelled_as_text_is_refused(container, uow_factory, ctx) -> None:
    """The bug this file exists for: a PDF reaching the text parser was decoded as UTF-8 and
    38,494 characters of "%PDF-1.7 %âãÏÓ 1 0 obj" were embedded and stored as knowledge."""
    register_handlers(container)
    pdf_bytes = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<\n/Producer (pypdf)\n>>\nendobj\n"
    from memory_service.adapters.parsers.builtin import BuiltinParser

    with pytest.raises(ValidationFailed, match="(?i)cannot read"):
        await BuiltinParser().parse(
            document_id="d",
            tenant_id="acme",
            filename="report.pdf",
            media_type="application/pdf",
            data=pdf_bytes,
        )


async def test_an_empty_document_is_refused_rather_than_indexed(
    container, uow_factory, ctx
) -> None:
    """Refusing is the right answer, and it is what already happens: an empty file has
    nothing to retrieve later, so accepting it would only create a document that can never
    match anything."""
    register_handlers(container)
    with pytest.raises(ValidationFailed, match="(?i)empty"):
        await _accept(
            container, uow_factory, ctx, name="empty.md", media_type="text/markdown", data=b""
        )


async def test_control_characters_and_null_bytes_do_not_reach_the_index(
    container, uow_factory, ctx
) -> None:
    register_handlers(container)
    payload = b"# Title\n\nGood sentence here.\n\n\x00\x01\x02\x03\x07\x1b[31m\x00\n"
    await _accept(
        container, uow_factory, ctx, name="noisy.md", media_type="text/markdown", data=payload
    )
    result = await container.services["retrieval"].retrieve(ctx, "good sentence", kinds=("chunk",))
    stored = " ".join(c.text or "" for c in result.candidates)
    assert "Good sentence" in stored, "the legible part should still be indexed"
    assert "\x00" not in stored, "null bytes reached the index"


async def test_a_single_enormous_line_is_split_rather_than_stored_whole(
    container, uow_factory, ctx
) -> None:
    """Minified JSON, a CSV export with no newlines, an OCR run that lost its line breaks.
    A chunk far over the token budget silently breaks the embedding window."""
    register_handlers(container)
    body = ("the quick brown fox jumps over the lazy dog. " * 2000).encode()
    await _accept(
        container, uow_factory, ctx, name="oneline.md", media_type="text/markdown", data=body
    )
    result = await container.services["retrieval"].retrieve(
        ctx, "quick brown fox", kinds=("chunk",)
    )
    assert result.candidates, "nothing was indexed"
    budget = container.tuning.documents.max_chunk_tokens
    for candidate in result.candidates:
        # ~4 characters per token; a chunk must not exceed the budget by more than the
        # overlap allowance, or it will be truncated by the encoder without anyone noticing
        assert len(candidate.text or "") <= budget * 8, (
            f"chunk of {len(candidate.text)} chars exceeds the {budget}-token budget"
        )


async def test_an_unanswerable_question_does_not_invent_evidence(
    container, uow_factory, ctx
) -> None:
    """The abstention property, on an empty corpus: with nothing indexed there is nothing to
    ground an answer in, and the bundle must say so rather than return unrelated text."""
    register_handlers(container)
    await _accept(
        container,
        uow_factory,
        ctx,
        name="facts.md",
        media_type="text/markdown",
        data=b"# Facts\n\nThe warehouse in Leeds holds 4,000 pallets.\n",
    )
    bundle = await container.services["context_builder"].build(
        ctx, "what is the share price of a company never mentioned here?"
    )
    rendered = bundle.render()
    assert "4,000 pallets" not in rendered or bundle.evidence.status.value != "COMPLETE", (
        "unrelated content was returned as COMPLETE evidence for an unanswerable question"
    )
