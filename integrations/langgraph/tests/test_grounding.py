"""``require_evidence`` + ``verify_answer``: the node's answer is verified claim by claim
against the recalled bundle before it is recorded; the report lands in the state and an
answer above ``max_hallucination_rate`` is recorded as not evidence-complete."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from tests.e2e.conftest import sdk_client

from universal_memory import GroundingReport
from universal_memory_langgraph import LangGraphMemory

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "acme_fy26_annual_report.md"
H = {"X-API-Key": "test-key", "X-Memory-Tenant": "acme", "X-Memory-User": "u1"}
Q = "Why did Adjusted EBITDA increase despite lower revenue?"
GOOD = "Adjusted EBITDA increased to EUR 98 million from EUR 81 million, despite lower revenue."
BAD = "Adjusted EBITDA increased to EUR 150 million from EUR 81 million."


class ChatState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    memory: Any


def _last_human(state: ChatState) -> str | None:
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", None) == "human":
            return m.content
    return None


def _upload(client) -> None:
    scope = {"thread_id": "thr_docs", "session_id": "ses_docs", "turn_id": "trn_docs"}
    r = client.post(
        "/v1/files",
        headers=H,
        files={"file": ("acme_fy26_annual_report.md", FIXTURE.read_bytes(), "text/markdown")},
        data={"scope": json.dumps(scope), "title": "ACME FY26", "visibility": "USER"},
    )
    assert r.status_code == 202, r.text


def _graph(memory: LangGraphMemory, answer_text: str, **wrap: Any):
    async def answer(state: ChatState) -> dict[str, Any]:
        assert state["memory"] is not None
        return {"messages": [("assistant", answer_text)]}

    return (
        StateGraph(ChatState)
        .add_node("answer", memory.wrap(answer, recall=_last_human, **wrap))
        .add_edge(START, "answer")
        .add_edge("answer", END)
        .compile(checkpointer=InMemorySaver())
    )


async def test_verify_answer_gates_recording_and_exposes_the_report(app, client) -> None:
    _upload(client)
    memory = LangGraphMemory(sdk_client(app), tenant_id="acme", user_id="u1")

    # a grounded answer passes the (critical, 0.0) gate
    cfg = {"configurable": {"thread_id": "grounded-1"}}
    out = await _graph(memory, GOOD, require_evidence=True, verify_answer=True).ainvoke(
        {"messages": [("user", Q)]}, cfg
    )
    last = memory.last
    assert last is not None and isinstance(last.grounding, GroundingReport)
    assert last.grounding.grounded and last.evidence_complete is True
    assert out["memory"]["grounding"].claims[0].verdict == "supported"
    assert out["memory"]["bundle"].bundle_id == last.bundle.bundle_id  # type: ignore[union-attr]
    history = await memory.context(cfg).chat.history()
    assert [m.role for m in history] == ["USER", "ASSISTANT"]
    recorded = await memory.context(cfg).chat.message(history[1].message_id)
    meta = recorded.model_dump().get("custom_metadata") or {}
    assert meta["grounding"]["evidence_complete"] is True and meta["grounding"]["claims"] == 1

    # a fabricated figure is contradicted: recorded, but not as evidence-complete
    cfg2 = {"configurable": {"thread_id": "grounded-2"}}
    out = await _graph(memory, f"{GOOD} {BAD}", require_evidence=True, verify_answer=True).ainvoke(
        {"messages": [("user", Q)]}, cfg2
    )
    last = memory.last
    assert last is not None and last.grounding is not None
    assert last.grounding.per_claim_hallucination_rate == 0.5 and last.evidence_complete is False
    report = out["memory"]["grounding"]
    assert [c.verdict for c in report.claims] == ["supported", "contradicted"]
    assert report.representative is False
    history = await memory.context(cfg2).chat.history()
    recorded = await memory.context(cfg2).chat.message(history[1].message_id)
    meta = recorded.model_dump().get("custom_metadata") or {}
    assert meta["grounding"]["evidence_complete"] is False
    assert meta["grounding"]["hallucination_rate"] == 0.5 and meta["grounding"]["contradicted"] == 1

    # the gate is configurable per node (and per adapter)
    cfg3 = {"configurable": {"thread_id": "grounded-3"}}
    await _graph(memory, f"{GOOD} {BAD}", verify_answer=True, max_hallucination_rate=0.5).ainvoke(
        {"messages": [("user", Q)]}, cfg3
    )
    assert memory.last is not None and memory.last.evidence_complete is True

    # without recall the answer is verified against a fresh retrieval for the answer itself
    lenient = LangGraphMemory(
        sdk_client(app), tenant_id="acme", user_id="u1", max_hallucination_rate=1.0
    )

    async def free(state: ChatState) -> dict[str, Any]:
        return {"messages": [("assistant", BAD)]}

    graph = (
        StateGraph(ChatState)
        .add_node("free", lenient.wrap(free, verify_answer=True))
        .add_edge(START, "free")
        .add_edge("free", END)
        .compile()
    )
    await graph.ainvoke({"messages": [("user", Q)]}, {"configurable": {"thread_id": "g-4"}})
    assert lenient.last is not None and lenient.last.bundle is None
    assert lenient.last.grounding is not None and lenient.last.grounding.contradicted == 1
    assert lenient.last.evidence_complete is True

    # no assistant message: nothing to verify, nothing attached
    async def silent(state: ChatState) -> dict[str, Any]:
        return {}

    graph = (
        StateGraph(ChatState)
        .add_node("silent", memory.wrap(silent, recall=_last_human, verify_answer=True))
        .add_edge(START, "silent")
        .add_edge("silent", END)
        .compile()
    )
    out = await graph.ainvoke({"messages": [("user", Q)]}, {"configurable": {"thread_id": "g-5"}})
    assert memory.last is not None and memory.last.grounding is None
    assert "memory" not in out or out.get("memory") is None
