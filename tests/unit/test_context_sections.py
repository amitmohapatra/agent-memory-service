"""The pinned sections of a pushed context: budgeting, rendering, and the delta."""

from __future__ import annotations

import orjson

from memory_service.domain.context_bundle import (
    ContextBundle,
    ContextItem,
    ConversationWindow,
    EvidenceReport,
    ProcedureView,
    ProfileBlockView,
    ThreadSummaryView,
)
from memory_service.domain.enums import EvidenceStatus, QueryType, Representation
from memory_service.domain.tools import MissingArgument, Prefill, ToolCandidate, ToolHints
from memory_service.modules.context.builder import _delta, bundle_to_api, served
from memory_service.modules.context.sections import Pinned, ToolsRequest, within_budget


def _pinned() -> Pinned:
    return Pinned(
        profile=[ProfileBlockView(block="user", text="name: Ann", version=2)],
        thread_summary=ThreadSummaryView(
            text="Ann ordered paper.", covers_to_sequence=20, version=1
        ),
        procedures=[
            ProcedureView(
                id="prc_1",
                title="Order paper",
                steps=[
                    {"ordinal": 0, "tool": "erp-get_stock"},
                    {"ordinal": 1, "tool": "erp-create_po"},
                ],
                success_rate=0.9,
                support=10,
            )
        ],
        tools=ToolHints(
            candidates=[ToolCandidate(name="erp-create_po", score=1.0, why="step 2")],
            next="erp-create_po",
            prefill={"supplier": Prefill(tool="erp-create_po", value="Acme", source="graph")},
            missing=[MissingArgument(tool="erp-create_po", arg="qty", question="How many?")],
        ),
    )


def test_the_pinned_sections_fit_half_the_budget_in_priority_order() -> None:
    kept, used = within_budget(_pinned(), 2000)
    assert kept.profile and kept.thread_summary and kept.procedures and kept.tools
    assert 0 < used <= 1000
    tight, used_tight = within_budget(_pinned(), 16)
    assert tight.profile and not tight.procedures and used_tight <= 8


def test_a_tools_request_is_part_of_the_cache_identity() -> None:
    assert ToolsRequest().fingerprint() == "tools:8:*"
    assert ToolsRequest(available=["b", "a"], k=3).fingerprint() == "tools:3:a,b"


def _bundle(**extra) -> ContextBundle:
    return ContextBundle(
        query="order paper",
        query_type=QueryType.GENERAL_SEMANTIC,
        conversation=ConversationWindow(thread_id="thr_1", rendered="USER: hi"),
        memories=[
            ContextItem(
                item_id="mem_1",
                representation=Representation.MEMORY,
                text="Acme supplies paper",
                citation="memory_id:mem_1",
            )
        ],
        evidence=EvidenceReport(status=EvidenceStatus.COMPLETE),
        token_budget=2000,
        token_estimate=100,
        **extra,
    )


def test_the_prompt_starts_from_the_pinned_sections() -> None:
    pinned = _pinned()
    rendered = _bundle(
        profile=pinned.profile,
        thread_summary=pinned.thread_summary,
        procedures=pinned.procedures,
        tools=pinned.tools,
    ).render()
    order = ["## Profile", "## Conversation summary", "## Procedures", "## Tools", "## Recent"]
    positions = [rendered.index(heading) for heading in order]
    assert positions == sorted(positions)
    assert "erp-get_stock -> erp-create_po (worked 90% of 10 runs)" in rendered
    assert "supplier = 'Acme' (graph)" in rendered and "missing qty: How many?" in rendered


def test_a_delta_lists_only_what_changed_since_the_record() -> None:
    before = _bundle()
    payload = orjson.dumps(bundle_to_api(before))
    same = orjson.loads(_delta(payload, orjson.dumps(served(before))))
    assert same["delta"] is True and same["memories"] == []
    changed = _bundle().model_copy(
        update={"memories": [before.memories[0].model_copy(update={"text": "Globex now"})]}
    )
    moved = orjson.loads(_delta(orjson.dumps(bundle_to_api(changed)), orjson.dumps(served(before))))
    assert [m["text"] for m in moved["memories"]] == ["Globex now"]
    assert orjson.loads(_delta(payload, None))["delta"] is False, "no record: the whole bundle"
