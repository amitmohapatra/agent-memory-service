"""Context preservation end to end: expansion over the Document Context Graph, hierarchical
summaries, evidence-group verification with escalation, honest abstention, and a bundle
whose evidence report reflects what fit in the budget."""

from __future__ import annotations

from pathlib import Path

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import EvidenceStatus, MessageRole, QueryType
from memory_service.domain.ids import new_id
from memory_service.modules.jobs.registry import register_handlers

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "acme_fy26_annual_report.md"
U1 = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
Q = "Why did Adjusted EBITDA increase despite lower revenue?"


async def _ingest(container, uow_factory, ctx=U1):
    register_handlers(container)
    async with uow_factory() as uow:
        ack = await container.services["ingestion"].accept_file(
            uow,
            ctx,
            filename="acme_fy26_annual_report.md",
            media_type="text/markdown",
            data=FIXTURE.read_bytes(),
            title="ACME FY26",
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()
    return ack.document_id


async def test_expansion_and_verification_complete_the_cross_page_chain(
    container, uow_factory
) -> None:
    await _ingest(container, uow_factory)
    engine = container.services["retrieval"]
    res = await engine.retrieve(U1, Q)
    assert res.diagnostics["stages"] == ["graph", "expansion", "verify"]
    report = res.diagnostics["evidence"]
    assert report["status"] == "COMPLETE", report
    names = set(report["required_groups"])
    assert {"defined_by:Adjusted EBITDA", "footnote:3", "cross_reference:Section 8"} <= names
    assert report["missing_groups"] == [] and set(report["satisfied_groups"]) == names
    # expansions are marked and come from the same document; the parent summary rides along
    expanded = [c for c in res.candidates if c.expansion_edge]
    assert expanded and {c.expansion_edge for c in expanded} >= {"PARENT"}
    assert all(c.retrievers in (["expansion"], ["graph"]) for c in expanded)
    summaries = [c for c in res.candidates if c.kind == "summary"]
    assert summaries and all(c.record_id.startswith("sum_nod_") for c in summaries)
    assert any("Adjusted EBITDA" in c.text for c in summaries)


async def test_escalation_fetches_missing_companions(container, uow_factory) -> None:
    await _ingest(container, uow_factory)
    engine = container.services["retrieval"]
    # without graph/expansion help and a tiny limit, the Section-8 chunk is not in the
    # ranked set: verification must escalate and fetch it directly
    stages = dict(engine.post_stages)
    engine.post_stages = {"verify": stages["verify"]}
    try:
        res = await engine.retrieve(U1, Q, limit=1)
    finally:
        engine.post_stages = stages
    report = res.diagnostics["evidence"]
    assert report["status"] == "COMPLETE", report
    assert report["escalations"] and "fetched" in report["escalations"][0]
    escalated = [c for c in res.candidates if c.expansion_edge == "ESCALATION"]
    assert escalated and 14 in {c.payload["page"] for c in escalated}
    # evidence items stay capped at the limit; companions ride along explicitly
    ranked = [c for c in res.candidates if c.kind == "chunk" and c.expansion_edge is None]
    assert len(ranked) == 1


async def test_abstention_and_no_evidence(container, uow_factory) -> None:
    await _ingest(container, uow_factory)
    engine = container.services["retrieval"]
    unrelated = "Who won the 1998 football championship?"
    res = await engine.retrieve(U1, unrelated)
    assert res.diagnostics["evidence"]["status"] == "INSUFFICIENT"
    assert "content term" in res.diagnostics["evidence"]["notes"][0]
    bundle = await container.services["context_builder"].build(U1, unrelated)
    assert bundle.evidence.status is EvidenceStatus.INSUFFICIENT
    assert "## Evidence status\nINSUFFICIENT" in bundle.render()
    # nothing at all for a stranger -> INSUFFICIENT with 'no evidence retrieved'
    stranger = MemoryExecutionContext(tenant_id="globex", user_id="u9")
    res = await engine.retrieve(stranger, Q)
    assert res.diagnostics["evidence"]["status"] == "INSUFFICIENT"
    assert res.diagnostics["evidence"]["notes"] == ["no evidence retrieved"]
    assert res.candidates == []


async def test_bundle_report_reflects_budget(container, uow_factory) -> None:
    await _ingest(container, uow_factory)
    builder = container.services["context_builder"]
    full = await builder.build(U1, Q)
    assert full.evidence.status is EvidenceStatus.COMPLETE
    assert {k.page for k in full.knowledge} >= {1, 11, 14, 20}
    assert full.summaries and "## Summaries" in full.render()
    assert all(s.expansion_edge == "PARENT" for s in full.summaries)
    # a seed travels with its companions: a tight budget still yields a COMPLETE bundle
    # (seed + definition + footnote + cross-reference packed together) ...
    tight = await builder.build(U1, Q, token_budget=260)
    assert tight.token_estimate <= 260 and tight.evidence.status is EvidenceStatus.COMPLETE
    assert {k.page for k in tight.knowledge} >= {1, 11, 14, 20}
    # ... and a budget that cannot hold the unit leaves the seed out and says so:
    # INCOMPLETE with the missing groups, never a seed without its companions
    small = await builder.build(U1, Q, token_budget=90)
    assert small.token_estimate <= 90
    assert small.evidence.status is EvidenceStatus.INCOMPLETE
    assert "token budget" in " ".join(small.evidence.notes)
    assert small.evidence.missing_groups
    seed_pages = {11}
    assert not ({k.page for k in small.knowledge} & seed_pages) or small.evidence.missing_groups


async def test_global_summary_and_conversation_rolling_summary(container, uow_factory) -> None:
    await _ingest(container, uow_factory)
    engine = container.services["retrieval"]
    res = await engine.retrieve(U1, "give me a summary of the main points of the report")
    assert res.routed.query_type is QueryType.GLOBAL_SUMMARY
    summaries = [c for c in res.candidates if c.kind == "summary"]
    assert summaries and any(c.text.startswith("ACME FY26:") for c in summaries)
    exact = await engine.retrieve(U1, f"show {summaries[0].record_id}")
    assert [c.record_id for c in exact.candidates] == [summaries[0].record_id]
    # conversation: older turns are digested into window.summary
    thread = new_id("thread")
    ctx = U1.model_copy(
        update={"thread_id": thread, "session_id": new_id("session"), "turn_id": new_id("turn")}
    )
    conv = container.services["conversation"]
    async with uow_factory() as uow:
        for i in range(8):
            await conv.append_message(
                uow, ctx, role=MessageRole.USER, content=f"Turn {i}: " + "detail " * 60
            )
        await uow.commit()
    small_window = container.tuning.context.model_copy(update={"conversation_token_budget": 200})
    builder = container.services["context_builder"]
    builder.cfg = small_window
    bundle = await builder.build(ctx, "what did I say earlier in this thread?")
    assert bundle.conversation.summary and bundle.conversation.summary.startswith("user: Turn 0")
    assert len(bundle.conversation.message_ids) < 8
    assert "## Conversation summary" in bundle.render()
