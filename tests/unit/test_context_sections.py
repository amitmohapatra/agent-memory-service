"""The pinned sections of a pushed context: budgeting and rendering."""

from __future__ import annotations

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
from memory_service.modules.context.sections import (
    SUMMARY_MIN_TOKENS,
    Pinned,
    ToolsRequest,
    within_budget,
)


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
            prefill={
                "erp-create_po.supplier": Prefill(
                    tool="erp-create_po", value="Acme", source="graph"
                )
            },
            missing=[MissingArgument(tool="erp-create_po", arg="qty", question="How many?")],
        ),
    )


def test_the_pinned_sections_fit_half_the_budget_in_priority_order() -> None:
    kept, used = within_budget(_pinned(), 2000)
    assert kept.profile and kept.thread_summary and kept.procedures and kept.tools
    assert 0 < used <= 1000
    tight, used_tight = within_budget(_pinned(), 30)
    assert tight.profile and not tight.procedures and used_tight <= 15


def test_each_section_costs_what_it_renders_to() -> None:
    """The tools section renders a few candidates, their arguments and what is missing, not
    all twenty: costing the whole hints object dropped it from budgets it fit."""
    from memory_service.domain.context_bundle import tools_section
    from memory_service.modules.ingestion.hierarchy import estimate_tokens

    pinned = _pinned()
    many = pinned.tools.model_copy(  # type: ignore[union-attr]
        update={
            "candidates": [
                ToolCandidate(name=f"tool_{i}", score=0.5, why="matches the task " * 5)
                for i in range(20)
            ]
        }
    )
    cost = estimate_tokens(tools_section(many) or "")
    only_tools = Pinned(tools=many)
    kept, used = within_budget(only_tools, cost * 2)
    assert kept.tools is not None and kept.tools.next == "erp-create_po" and used == cost


def test_a_summary_over_its_share_is_truncated_not_dropped() -> None:
    long = ThreadSummaryView(text="x " * 4000, covers_to_sequence=90, version=3)
    kept, used = within_budget(Pinned(thread_summary=long), 400)
    assert kept.thread_summary is not None
    assert kept.thread_summary.text.startswith("...")
    assert len(kept.thread_summary.text) < len(long.text) and used <= 200
    gone, _ = within_budget(Pinned(thread_summary=long), SUMMARY_MIN_TOKENS)
    assert gone.thread_summary is None, "below the minimum a summary says too little to keep"


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
    assert (
        "- erp-create_po (confidence 0.00, next step): supplier = 'Acme'; "
        "missing qty: How many?" in rendered
    ), rendered
    assert "step 2" not in rendered, "the why is for the full form, not the prompt"
    assert "[m1] Acme supplies paper" in rendered
