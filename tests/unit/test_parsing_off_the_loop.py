"""CPU-bound parsing and document fact extraction run off the event loop (ADR 0031): a large
document must not stop the loop's other work - in the worker, Procrastinate's heartbeat and
every other job - for as long as it takes to parse."""

from __future__ import annotations

import asyncio
import threading

import pytest

from memory_service.adapters.parsers import builtin
from memory_service.adapters.parsers.builtin import BuiltinParser

pytestmark = pytest.mark.unit


async def test_the_builtin_parse_runs_in_a_thread_and_the_loop_keeps_ticking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    original = builtin.markdown_blocks

    def recording(text: str):  # type: ignore[no-untyped-def]
        seen.append(threading.current_thread().name)
        return original(text)

    monkeypatch.setattr(builtin, "markdown_blocks", recording)
    text = "\n\n".join(
        f"# Section {i}\n\nRevenue was EUR {i} million in FY26. " * 3 for i in range(3000)
    )
    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.005)

    beat = asyncio.create_task(heartbeat())
    try:
        parsed = await BuiltinParser().parse(
            document_id="doc_1",
            tenant_id="acme",
            filename="big.md",
            media_type="text/markdown",
            data=text.encode(),
        )
    finally:
        beat.cancel()
    assert parsed.nodes
    assert seen and seen[0] != threading.main_thread().name
    assert ticks > 3, "the loop did not run while the document was parsed"


async def test_document_fact_extraction_runs_in_a_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memory_service.modules.graph import native

    seen: list[str] = []
    original = native.DocumentIE.run

    def recording(self):  # type: ignore[no-untyped-def]
        seen.append(threading.current_thread().name)
        return original(self)

    monkeypatch.setattr(native.DocumentIE, "run", recording)
    parsed = await BuiltinParser().parse(
        document_id="doc_1",
        tenant_id="acme",
        filename="r.md",
        media_type="text/markdown",
        data=b"# Report\n\nAcme Corp acquired Westfalen GmbH in 2025.\n",
    )
    from memory_service.domain.context import MemoryExecutionContext
    from memory_service.modules.ingestion.chunking import chunk_nodes

    chunks = chunk_nodes(parsed.nodes, document_title="Report")
    await native.NativeGraphEnrichment().enrich_document(
        parsed.version,
        parsed.nodes,
        chunks,
        MemoryExecutionContext(tenant_id="acme"),
        document_title="Report",
    )
    assert seen and seen[0] != threading.main_thread().name


async def test_a_document_with_two_top_level_headings_parses() -> None:
    """The title heading put the root at level 1, and the next H1 popped it: IndexError."""
    from memory_service.domain.enums import Representation

    parsed = await BuiltinParser().parse(
        document_id="doc_1",
        tenant_id="acme",
        filename="a.md",
        media_type="text/markdown",
        data=b"# A\n\nfirst\n\n# B\n\nsecond\n",
    )
    sections = [n.title for n in parsed.nodes if n.representation is Representation.SECTION]
    assert sections == ["B"]
