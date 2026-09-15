"""A LangGraph research crew backed by the Memory Service — no LLM required.

    ./examples/run_server.sh &                       # http://localhost:8080
    uv run python examples/langgraph_crew/app.py     # MEMORY_URL / MEMORY_API_KEY override

Topology (every node is wrapped by ``LangGraphMemory`` so memory happens around it):

    root:   intake -> crew -> answer
    crew:   plan -> research -> review          (a subgraph = agent "crew", one run per turn)
    research: gather -> summarise               (nested subgraph = agent "research", a child run)

What the adapter does for us: ``thread_id`` becomes the memory thread; the user's turn and
the assistant's reply are recorded; each node gets a ``ContextBundle`` under
``state["memory"]``; the crew's plan is an agent RUN memory that the research run (its
child) can read and the user cannot; research findings are shared with the agent group
on purpose; the final answer is composed only from evidence the service verified.

The "reasoning" here is deterministic (it composes sentences from graph facts and
evidence), so the example runs anywhere. To use an LLM, call it inside ``answer`` with
``state["memory"].rendered`` as the prompt context — nothing else changes.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from universal_memory import MemoryClient, current_context
from universal_memory_langgraph import LangGraphMemory

URL = os.environ.get("MEMORY_URL", "http://localhost:8080")
API_KEY = os.environ.get("MEMORY_API_KEY", "dev-key")
FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
RUN = uuid.uuid4().hex[:8]
TENANT = f"crew-{RUN}"


class State(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    question: str
    memory: Any  # ContextBundle injected by the adapter
    plan: str
    findings: list[str]
    review: str
    answer: str
    trace: Annotated[list[str], lambda a, b: a + [x for x in b if x not in a]]


def last_human(state: State) -> str | None:
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", None) == "human":
            return m.content
    return None


# --------------------------------------------------------------------------- root nodes


async def intake(state: State) -> dict[str, Any]:
    """Normalise the user's turn into a question. The adapter already recorded the turn."""
    return {"question": last_human(state) or "", "trace": ["intake"]}


async def answer(state: State) -> dict[str, Any]:
    """Compose the reply from the verified bundle + the crew's review. With an LLM you would
    prompt it with ``bundle.rendered`` here; the evidence report tells you whether you may."""
    bundle = state["memory"]
    if bundle is None or bundle.evidence.status == "INSUFFICIENT":
        text = "I don't have enough evidence in your documents to answer that."
    else:
        wanted = ("has_value", "driven_by", "excludes", "would_have_value")
        facts = [f.text for f in bundle.graph_facts if f.predicate in wanted][:3]
        prefs = [m.text for m in bundle.memories if "concise" in m.text.lower()]
        lines = [state.get("review") or ""]
        if facts and not prefs:
            lines.append("Key facts: " + " | ".join(facts))
        pages = sorted({k.page for k in bundle.knowledge if k.page})
        lines.append(f"(evidence: {bundle.evidence.status.lower()}, pages {pages})")
        text = " ".join(line for line in lines if line)
    return {"answer": text, "messages": [("assistant", text)], "trace": ["answer"]}


# --------------------------------------------------------------------------- crew nodes


async def plan(state: State) -> dict[str, Any]:
    """The crew's plan is hand-off context: RUN-scoped, readable by the research child run."""
    q = state["question"]
    steps = (
        ["definitions", "figures", "drivers", "footnotes"] if "why" in q.lower() else ["figures"]
    )
    return {"plan": f"Plan: answer '{q}' by checking {', '.join(steps)}.", "trace": ["plan"]}


async def gather(state: State) -> dict[str, Any]:
    """Runs as agent 'research' (child of the crew run): it reads the crew's plan from
    memory, then uses the SDK directly (via current_context) as a 'tool'."""
    ctx = current_context()
    assert ctx is not None
    handoff = [m.text for m in state["memory"].memories if m.text.startswith("Plan:")]
    facts = await ctx.graph.query(query=state["question"], hops=1)
    findings = [f"{f.subject} {f.predicate.replace('_', ' ')} {f.object}" for f in facts.facts[:4]]
    findings += [f"[handoff] {h}" for h in handoff]
    return {"findings": findings, "trace": ["gather"]}


async def summarise(state: State) -> dict[str, Any]:
    """Share the research findings with the whole agent group (explicit, not implicit)."""
    return {"trace": ["summarise"]}


async def review(state: State) -> dict[str, Any]:
    """The crew reads the shared findings back from memory and checks for conflicts."""
    bundle = state["memory"]
    notes = list(bundle.evidence.notes) if bundle else []
    conflict = any("conflicting" in n for n in notes)
    shared = [m.text for m in bundle.memories if m.text.startswith("Finding:")] if bundle else []
    verdict = f"Reviewed {len(shared)} shared finding(s)"
    if conflict:
        verdict += "; conflicting memories flagged for a human"
    return {"review": verdict + ".", "trace": ["review"]}


def build_graph(memory: LangGraphMemory):
    research = (
        StateGraph(State)
        .add_node("gather", memory.wrap(gather, recall="question", record_messages=False))
        .add_node(
            "summarise",
            memory.wrap(
                summarise,
                record_messages=False,
                observe=lambda s, r: [
                    {
                        "content": f"Finding: {f}",
                        "hints": {"memory_type": "SHARED", "visibility": "AGENT_GROUP"},
                    }
                    for f in s.get("findings", [])
                    if not f.startswith("[handoff]")
                ],
            ),
        )
        .add_edge(START, "gather")
        .add_edge("gather", "summarise")
        .add_edge("summarise", END)
        .compile()
    )
    crew = (
        StateGraph(State)
        .add_node("plan", memory.wrap(plan, observe="plan", record_messages=False))
        .add_node("research", research)
        .add_node("review", memory.wrap(review, recall="question", record_messages=False))
        .add_edge(START, "plan")
        .add_edge("plan", "research")
        .add_edge("research", "review")
        .add_edge("review", END)
        .compile()
    )
    return (
        StateGraph(State)
        .add_node("intake", memory.wrap(intake))  # records the human turn
        .add_node("crew", crew)
        .add_node("answer", memory.wrap(answer, recall="question"))  # records the reply
        .add_edge(START, "intake")
        .add_edge("intake", "crew")
        .add_edge("crew", "answer")
        .add_edge("answer", END)
        .compile(checkpointer=InMemorySaver())
    )


async def main() -> int:
    client = MemoryClient(URL, api_key=API_KEY)
    memory = LangGraphMemory(client, tenant_id=TENANT, user_id="amit", agent_group_id="crew")
    graph = build_graph(memory)
    cfg = {"configurable": {"thread_id": f"crew-thread-{RUN}"}}
    user = memory.context(cfg)

    print(f"# LangGraph research crew (tenant {TENANT})\n")
    print("Ingesting the FY26 report ...")
    handle = await user.files.add(
        FIXTURES / "acme_fy26_annual_report.md",
        title="ACME FY26 Annual Report",
        visibility="AGENT_GROUP",
    )
    doc = await user.files.wait_ready(handle.document_id)
    print(f"  document {doc.document_id}: {doc.status}\n")

    failures: list[str] = []

    def check(cond: bool, what: str) -> None:
        print(f"  [{'ok' if cond else 'FAIL'}] {what}")
        if not cond:
            failures.append(what)

    # ---- turn 1: a multi-hop question ------------------------------------------
    q1 = "Why did Adjusted EBITDA increase despite lower revenue?"
    print(f"Turn 1: {q1}")
    out = await graph.ainvoke({"messages": [("user", q1)]}, cfg)
    print(f"  crew trace: {' -> '.join(out['trace'])}")
    print(f"  answer: {out['answer'][:220]}...\n")
    history = await user.chat.history()
    check([m.role for m in history] == ["USER", "ASSISTANT"], "turn recorded as USER + ASSISTANT")
    check(
        out["findings"] and any("[handoff] Plan:" in f for f in out["findings"]),
        "research run read the crew's plan (hand-off flows down)",
    )
    check(
        any("Adjusted EBITDA" in f for f in out["findings"]),
        "research used the knowledge graph as a tool",
    )
    check("evidence: complete" in out["answer"], "answer composed from a COMPLETE evidence bundle")
    user_view = await user.recall("plan for the brief", kinds=["memory"])
    check(
        not any(m.text.startswith("Plan:") for m in user_view),
        "the user cannot see the crew's working notes (never up)",
    )
    auditor = user.agent("auditor")
    shared = await auditor.recall("Adjusted EBITDA findings", kinds=["memory"])
    check(
        any(m.text.startswith("Finding:") for m in shared),
        "shared findings are readable by another agent in the group",
    )

    # ---- turn 2: a preference, then a question that uses it ---------------------
    print("\nTurn 2: I prefer concise answers.")
    await graph.ainvoke({"messages": [("user", "I prefer concise answers.")]}, cfg)
    q3 = "What was ACME's total revenue in FY26?"
    print(f"Turn 3: {q3}")
    out3 = await graph.ainvoke({"messages": [("user", q3)]}, cfg)
    print(f"  answer: {out3['answer'][:220]}\n")
    mems = await user.memories()
    check(any(m.predicate == "prefers" for m in mems), "the preference became a USER memory")
    check("Key facts" not in out3["answer"], "the answer honoured the stored preference (concise)")
    history = await user.chat.history()
    check(len(history) == 6, f"three turns in the thread ({len(history)} messages)")
    # retries: re-running the same superstep records nothing twice (same task ids -> same keys)
    snapshot = graph.get_state(cfg)
    check(snapshot.next == (), "graph finished cleanly (checkpointed)")

    await client.aclose()
    print(f"\n{'all checks passed' if not failures else f'{len(failures)} check(s) failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
