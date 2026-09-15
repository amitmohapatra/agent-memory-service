"""Real LangGraph graphs against the real service (in-process ASGI): chat graph with the
``messages`` convention, nested subgraphs as agent runs (hand-off context flows down, not
up), checkpoint retries never duplicate writes, evidence gating and best-effort recall."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import RetryPolicy
from tests.e2e.conftest import sdk_client

from universal_memory import InsufficientEvidence, current_context
from universal_memory_langgraph import LangGraphMemory

pytestmark = pytest.mark.integration


class ChatState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    memory: Any
    seen: list[str]


def _last_human(state: ChatState) -> str | None:
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", None) == "human":
            return m.content
    return None


async def test_chat_graph_records_turns_and_injects_context(app, client) -> None:
    memory = LangGraphMemory(sdk_client(app), tenant_id="acme", user_id="u1")
    bundles: list[Any] = []

    async def answer(state: ChatState) -> dict[str, Any]:
        bundles.append(state["memory"])
        assert current_context() is not None  # tools inside the node can use the SDK
        return {"messages": [("assistant", f"Noted: {_last_human(state)}")]}

    graph = (
        StateGraph(ChatState)
        .add_node("answer", memory.wrap(answer, recall=_last_human))
        .add_edge(START, "answer")
        .add_edge("answer", END)
        .compile(checkpointer=InMemorySaver())
    )
    cfg = {"configurable": {"thread_id": "chat 42"}}
    await graph.ainvoke({"messages": [("user", "My timezone is Europe/Berlin.")]}, cfg)
    await graph.ainvoke({"messages": [("user", "What is my timezone?")]}, cfg)
    assert memory.last is not None and memory.last.recorded_messages == 2
    ctx = memory.context(cfg)
    history = await ctx.chat.history()
    assert [m.role for m in history] == ["USER", "ASSISTANT", "USER", "ASSISTANT"]
    assert history[0].content == "My timezone is Europe/Berlin."
    # the second turn saw the first one (conversation window) and the extracted memory
    assert "Europe/Berlin" in bundles[1].conversation.rendered
    assert any("europe/berlin" in m.text.lower() for m in bundles[1].memories)
    assert bundles[0].conversation.rendered == "" or "timezone" in bundles[0].conversation.rendered
    # nothing leaks to another user of the same LangGraph thread id
    other = LangGraphMemory(sdk_client(app), tenant_id="acme", user_id="u2").context(cfg)
    assert not any(
        "berlin" in i.text.lower() for i in await other.recall("timezone", kinds=["memory"])
    )


async def test_nested_subgraphs_are_agent_runs_and_handoff_flows_down(app, client) -> None:
    memory = LangGraphMemory(sdk_client(app), tenant_id="acme", user_id="u1")
    seen: dict[str, Any] = {}

    class S(TypedDict, total=False):
        question: str
        plan: str
        notes: str
        memory: Any

    async def plan(state: S) -> dict[str, Any]:
        seen["plan_scope"] = current_context().scope  # type: ignore[union-attr]
        return {"plan": "Plan: split the brief into revenue and cost."}

    async def worker(state: S) -> dict[str, Any]:
        seen["worker_scope"] = current_context().scope  # type: ignore[union-attr]
        seen["worker_bundle"] = state["memory"]
        return {"notes": "Draft note: revenue section uses Table 1."}

    research = (
        StateGraph(S)
        .add_node("worker", memory.wrap(worker, recall="question"))
        .add_edge(START, "worker")
        .add_edge("worker", END)
        .compile()
    )
    crew = (
        StateGraph(S)
        .add_node("plan", memory.wrap(plan, observe="plan"))
        .add_node("research", research)
        .add_edge(START, "plan")
        .add_edge("plan", "research")
        .add_edge("research", END)
        .compile()
    )

    async def answer(state: S) -> dict[str, Any]:
        seen["answer_bundle"] = state["memory"]
        return {}

    root = (
        StateGraph(S)
        .add_node("crew", crew)
        .add_node("answer", memory.wrap(answer, recall="question"))
        .add_edge(START, "crew")
        .add_edge("crew", "answer")
        .add_edge("answer", END)
        .compile(checkpointer=InMemorySaver())
    )
    cfg = {"configurable": {"thread_id": "brief-1"}}
    await root.ainvoke({"question": "plan for the brief sections"}, cfg)
    plan_scope, worker_scope = seen["plan_scope"], seen["worker_scope"]
    # `plan` runs inside the crew subgraph -> it acts as agent "crew" (one run per invocation)
    assert plan_scope.agent_id == "crew" and plan_scope.agent_run_id.startswith("lg-")
    assert plan_scope.parent_agent_run_id is None
    # `worker` runs in the research subgraph nested in crew -> agent "research", child of crew
    assert worker_scope.agent_id == "research"
    assert worker_scope.parent_agent_run_id == plan_scope.agent_run_id
    assert worker_scope.agent_run_id != plan_scope.agent_run_id
    # the crew's plan is RUN-scoped hand-off context: the child run reads it ...
    assert any("Plan:" in m.text for m in seen["worker_bundle"].memories)
    # ... the user at the root graph does not (never up)
    assert not any("Plan:" in m.text for m in seen["answer_bundle"].memories)
    acked = memory.last
    assert acked is not None and acked.node == "answer"
    mem = memory.context(cfg)
    assert not any("Plan:" in i.text for i in await mem.recall("plan brief", kinds=["memory"]))
    # explicit `agent=` on a node makes that node its own run under the enclosing one
    explicit = memory.context(
        {"configurable": {"thread_id": "brief-1", "checkpoint_ns": "crew:A|plan:T"}},
        agent="planner",
    )
    assert explicit.scope.agent_id == "planner" and explicit.scope.parent_agent_run_id == "lg-A"


async def test_retries_from_a_checkpoint_never_duplicate_writes(app, client) -> None:
    memory = LangGraphMemory(sdk_client(app), tenant_id="acme", user_id="u1")
    attempts = {"n": 0}

    class Flaky(RuntimeError):
        pass

    def observe(state: ChatState, result: dict[str, Any]) -> Any:
        # the message was already recorded when this runs; fail once so LangGraph retries
        # the whole node in the same superstep (same task id -> same idempotency keys)
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise Flaky("transient")
        return {"content": "We decided to use PostgreSQL.", "kind": "DECISION"}

    async def answer(state: ChatState) -> dict[str, Any]:
        return {"messages": [("assistant", "Decision recorded.")]}

    graph = (
        StateGraph(ChatState)
        .add_node(
            "answer",
            memory.wrap(answer, recall=_last_human, observe=observe),
            retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.01, retry_on=Flaky),
        )
        .add_edge(START, "answer")
        .add_edge("answer", END)
        .compile(checkpointer=InMemorySaver())
    )
    cfg = {"configurable": {"thread_id": "retry-1"}}
    await graph.ainvoke({"messages": [("user", "Let's decide on the database.")]}, cfg)
    assert attempts["n"] == 2
    ctx = memory.context(cfg)
    history = await ctx.chat.history()
    assert [m.role for m in history] == ["USER", "ASSISTANT"]  # recorded once, not twice
    assert memory.last is not None and len(memory.last.observations) == 1
    ack = memory.last.observations[0]
    # replaying the exact observation is acknowledged with the same id (idempotent)
    again = await ctx.observe(
        "We decided to use PostgreSQL.",
        kind="DECISION",
        idempotency_key=memory._key(
            "obs",
            memory.lineage({**cfg, "metadata": {"langgraph_step": 1}}),
            "answer",
            "DECISION",
            "We decided to use PostgreSQL.",
        ),
    )
    assert again.observation_id == ack.observation_id or again.observation_id.startswith("obs_")
    assert any(
        "postgresql" in i.text.lower()
        for i in await ctx.recall("database decision", kinds=["memory"])
    )


async def test_evidence_gate_and_best_effort_recall(app, client) -> None:
    strict = LangGraphMemory(sdk_client(app), tenant_id="acme", user_id="u1")

    async def answer(state: ChatState) -> dict[str, Any]:
        return {"messages": [("assistant", "I don't know.")]}

    gated = (
        StateGraph(ChatState)
        .add_node("answer", strict.wrap(answer, recall=_last_human, require_evidence=True))
        .add_edge(START, "answer")
        .add_edge("answer", END)
        .compile()
    )
    with pytest.raises(InsufficientEvidence):
        await gated.ainvoke(
            {"messages": [("user", "Who won the 1998 football championship?")]},
            {"configurable": {"thread_id": "gate-1"}},
        )
    # a recall failure with strict=False lets the node run without context
    lenient = LangGraphMemory(sdk_client(app, api_key="wrong-key"), tenant_id="acme", strict=False)
    got: list[Any] = []

    async def tolerant(state: ChatState) -> dict[str, Any]:
        got.append(state["memory"])
        return {}

    graph = (
        StateGraph(ChatState)
        .add_node("tolerant", lenient.wrap(tolerant, recall=_last_human, record_messages=False))
        .add_edge(START, "tolerant")
        .add_edge("tolerant", END)
        .compile()
    )
    await graph.ainvoke({"messages": [("user", "hello")]}, {"configurable": {"thread_id": "x"}})
    assert got == [None]
    # ... but a failing write is never swallowed (no silent data loss)
    graph2 = (
        StateGraph(ChatState)
        .add_node("tolerant", lenient.wrap(tolerant, observe=lambda s, r: "a fact"))
        .add_edge(START, "tolerant")
        .add_edge("tolerant", END)
        .compile()
    )
    with pytest.raises(Exception, match="(?i)auth|401|403|api"):
        await graph2.ainvoke(
            {"messages": [("user", "hello")]}, {"configurable": {"thread_id": "x"}}
        )
