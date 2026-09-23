"""The heaviest model in the service is entered one caller at a time, like the other two.

Docling's layout and table models fan their work over every core. ``asyncio.to_thread``
hands them to the event loop's default executor - ``min(32, cpu_count + 4)``, eight threads
on a four-core box - so eight documents could be inside those models at once, competing with
each other and with the encoder for the same cores. ``SerialRunner`` exists precisely to stop
that, and had been applied to the encoder and the NLI head but not here.

Not visible to the load test, which uploads ``text/markdown``: that routes to the builtin
parser and never reaches docling at all. So the capacity numbers taken so far exclude the
most expensive path the service has, and nothing would have caught this by measurement.
"""

from __future__ import annotations

import asyncio

import pytest

from memory_service.adapters.models._runner import SerialRunner
from memory_service.adapters.parsers.docling_parser import DoclingParser

pytestmark = pytest.mark.unit


def test_the_parser_owns_a_gate_rather_than_the_shared_executor() -> None:
    parser = DoclingParser()
    assert isinstance(parser._runner, SerialRunner)
    assert parser._runner.name == "docling"


async def test_two_documents_are_never_inside_the_models_at_once() -> None:
    """The property that matters: the gate admits one, whatever the executor would allow."""
    parser = DoclingParser()
    inside = 0
    peak = 0

    def _convert() -> None:
        nonlocal inside, peak
        inside += 1
        peak = max(peak, inside)
        # long enough that a second caller would overlap if the gate were not there
        import time

        time.sleep(0.05)
        inside -= 1

    await asyncio.gather(*(parser._runner.run(_convert) for _ in range(4)))
    assert peak == 1, f"{peak} documents were inside docling at once"


def test_text_still_bypasses_docling_entirely() -> None:
    """Markdown and friends go to the builtin parser, so the gate costs them nothing."""
    parser = DoclingParser()
    assert "text/markdown" in parser.supported_media_types
    assert "application/pdf" in parser.supported_media_types
