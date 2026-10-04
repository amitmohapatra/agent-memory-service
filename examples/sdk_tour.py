"""A guided tour of every SDK method and every API route, run against a live server.

    uv run python examples/serve.py &   # http://localhost:8080, API key "dev-key"
    uv run python examples/sdk_tour.py  # MEMORY_URL / MEMORY_API_KEY override the defaults

Each step is a check with an assertion; the script prints a checklist and exits non-zero if
anything failed. It doubles as living documentation of what the service guarantees:
durable acknowledgements, idempotent replays, scope isolation, memory consolidation,
agent-run visibility, evidence-gated context, and the knowledge graph.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trellis.memory import (
    AuthorizationError,
    MemoryClient,
    NotFoundError,
    current_context,
)

URL = os.environ.get("MEMORY_URL", "http://localhost:8080")
API_KEY = os.environ.get("MEMORY_API_KEY", "dev-key")
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
RUN = uuid.uuid4().hex[:8]  # unique ids so the tour can be re-run against the same server


class Checklist:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def ok(self, name: str, detail: str = "") -> None:
        self.rows.append((name, True, detail))
        print(f"  [ok]   {name}{' — ' + detail if detail else ''}")

    def fail(self, name: str, detail: str) -> None:
        self.rows.append((name, False, detail))
        print(f"  [FAIL] {name} — {detail}")

    async def step(self, name: str, fn: Any) -> Any:
        try:
            result = await fn()
        except Exception as exc:  # noqa: BLE001 - reported in the checklist
            self.fail(name, f"{type(exc).__name__}: {exc}")
            return None
        detail = result if isinstance(result, str) else ""
        self.ok(name, detail)
        return result

    @property
    def failed(self) -> int:
        return sum(1 for _, ok, _ in self.rows if not ok)


async def tour() -> int:
    c = Checklist()
    memory = MemoryClient(URL, api_key=API_KEY)
    tenant = f"tour-{RUN}"
    user = memory.bind(
        tenant_id=tenant,
        user_id="amit",
        workspace_id="finance",
        thread_id=f"thr-{RUN}",
        session_id=f"ses-{RUN}",
        turn_id=f"trn-{RUN}-1",
    )

    # ---------------------------------------------------------------- ops
    print("\n## Operations")

    async def health() -> str:
        ready = await memory.health()
        alive = await memory.alive()
        version = await memory.version()
        assert ready["status"] == "ready" and alive["status"] == "ok"
        return f"v{version['version']} providers={version['providers']['embedding']}"

    await c.step("GET /health/ready, /health/live, /version", health)

    # ---------------------------------------------------------------- chat
    print("\n## Conversation (threads, messages, history, lineage)")

    async def create_thread() -> str:
        info = await user.history.update(title="FY26 brief", metadata={"channel": "tour"})
        again = await user.history.update(title="FY26 brief", metadata={"channel": "tour"})
        assert info.thread_id == user.scope.thread_id == again.thread_id
        return f"thread {info.thread_id} title={info.title!r}"

    await c.step("PATCH /v1/threads/{id} (create with a title, converges)", create_thread)

    async def messages() -> str:
        said = ("USER", "My timezone is Europe/Berlin and I prefer concise answers.")
        [a] = await user.history.add([said])
        [replay] = await user.history.add([said])
        assert replay.message_id == a.message_id, "identical message must be an idempotent replay"
        [b] = await user.history.add([("ASSISTANT", "Noted: Europe/Berlin, concise answers.")])
        planner = user.agent("planner")
        await planner.history.add(
            [
                {
                    "role": "AGENT",
                    "kind": "INTERNAL",
                    "content": "Thinking: split the brief into revenue and cost.",
                }
            ]
        )
        visible = await user.history()
        everything = await user.history(include_internal=True)
        assert [m.role for m in visible] == ["USER", "ASSISTANT"]
        assert len(everything) == 3 and everything[2].kind == "INTERNAL"
        one = await user.history.message(b.message_id)
        assert one.content.startswith("Noted") and one.sequence == 2
        thread = await user.history.thread()
        assert thread.title == "FY26 brief"
        return (
            f"{len(everything)} messages (1 internal), sequences {[m.sequence for m in everything]}"
        )

    await c.step(
        "POST /v1/messages, GET /v1/threads/{id}[/messages], GET /v1/messages/{id}", messages
    )

    async def isolation() -> str:
        stranger = memory.bind(tenant_id=tenant, user_id="mallory", thread_id=user.scope.thread_id)
        try:
            await stranger.history()
            raise AssertionError("another user read the thread")
        except AuthorizationError:
            pass
        other_tenant = memory.bind(
            tenant_id=f"{tenant}-other", user_id="amit", thread_id=user.scope.thread_id
        )
        try:
            await other_tenant.history.thread()
            raise AssertionError("another tenant saw the thread")
        except (AuthorizationError, NotFoundError):
            pass
        return "other user -> 403, other tenant -> not found"

    await c.step("scope isolation on threads", isolation)

    # ---------------------------------------------------------------- files
    print("\n## Files (ingestion, documents, jobs)")
    doc_id: dict[str, str] = {}

    async def ingest() -> str:
        salt = f"\n\n<!-- tour {RUN} -->\n".encode()
        handle = await user.advanced.documents.add(
            FIXTURES / "acme_fy26_annual_report.md", title="ACME FY26 Annual Report"
        )
        handle2 = await user.advanced.documents.add(
            (FIXTURES / "globex_fy26_annual_report.md").read_bytes() + salt,
            filename="globex_fy26_annual_report.md",
            media_type="text/markdown",
            title="GLOBEX FY26 Annual Report",
        )
        doc = await user.advanced.documents.wait_ready(handle.document_id)
        doc2 = await user.advanced.documents.wait_ready(handle2.document_id)
        assert doc.status == "READY" and doc2.status == "READY", (doc.status, doc2.status)
        assert doc.archive_status == "ARCHIVED"
        dup = await user.advanced.documents.add(
            FIXTURES / "acme_fy26_annual_report.md", title="duplicate upload"
        )
        assert dup.document_id == handle.document_id and dup.deduplicated
        doc_id["acme"], doc_id["globex"] = handle.document_id, handle2.document_id
        if handle.job_ids:
            job = await user.advanced.job(handle.job_ids[0])
            assert job.status in ("SUCCEEDED", "PENDING", "RUNNING")
        return f"acme={handle.document_id} globex={handle2.document_id} (same bytes -> dedup)"

    await c.step("POST /v1/documents, GET /v1/documents/{id}, GET /v1/jobs/{id}", ingest)

    # ---------------------------------------------------------------- retrieval
    print("\n## Retrieval (recall, context, evidence)")

    async def recall() -> str:
        items = await user.search("Why did Adjusted EBITDA increase despite lower revenue?")
        pages = {i.page for i in items if i.document_id == doc_id["acme"]}
        assert {1, 11, 14, 20} <= pages, pages
        assert all(i.id and i.kind for i in items)
        table = await user.search("Legacy Services revenue FY25 vs FY26", limit=5)
        assert any("| Legacy Services | 153 | 111 |" in i.text for i in table)
        return (
            f"{len(items)} items, pages {sorted(p for p in pages if p)} incl. definition, footnote"
        )

    await c.step("POST /v1/recall (hybrid + graph + expansion + verification)", recall)

    async def context() -> str:
        question = "Why did Adjusted EBITDA increase despite lower revenue?"
        prompt = await user.context(question)
        assert prompt.rendered and prompt.bundle_id and "## " in prompt.rendered
        bundle = await user.context(question, format="full")
        assert bundle.evidence_status == "COMPLETE", bundle.missing_evidence
        assert bundle.knowledge and bundle.graph_facts
        assert bundle.conversation is not None
        assert any("Europe/Berlin" in m.text for m in bundle.conversation.messages)
        cached = await user.context(question, format="full")
        assert cached == bundle, "a repeat is served from the cache, unchanged"
        unrelated = await user.context("Who won the 1998 football championship?", format="full")
        assert unrelated.insufficient, unrelated.evidence_status
        small = await user.context("What is Adjusted EBITDA?", token_budget=400)
        assert small.token_estimate <= 400 + 120
        return (
            f"evidence={bundle.evidence_status}, {len(bundle.knowledge)} chunks, "
            f"{len(bundle.graph_facts)} facts, cached on repeat, abstains on unrelated question"
        )

    await c.step("POST /v1/context (prompt, full bundle, cache, budget, evidence)", context)

    # ---------------------------------------------------------------- memory
    print("\n## Memory intelligence (events, remember, list, get, supersede, forget)")
    mem_ids: dict[str, str] = {}
    before_correction = datetime.now(UTC)

    async def observe() -> str:
        said = ("USER", "I work at ACME Corp and my favourite editor is neovim.")
        [ack] = await user.history.add([said])
        [replay] = await user.history.add([said])
        assert replay.message_id == ack.message_id, "idempotent replay"
        # an event: the service extracts the triple (decided) from it
        await user.history.add([("EVENT", "We decided to use PostgreSQL as the canonical store.")])
        await user.remember("Always answer in British English.", memory_type="PREFERENCE")
        # a statement: stored verbatim, now, and replaced by a new version on update
        stated = await user.remember("The brief is due on Friday.", memory_type="TASK")
        moved = await user.update(stated.memory_id, "The brief is due on Monday.", reason="moved")
        assert (await user.advanced.memories.get(stated.memory_id)).superseded_by == moved.memory_id
        await asyncio.sleep(0.5)
        mems = await user.advanced.memories.list()
        preds = {m.predicate: m for m in mems if m.predicate}
        assert {"timezone", "prefers", "works_at", "favourite_editor", "decided"} <= set(preds), (
            set(preds)
        )
        tz = preds["timezone"]
        assert tz.memory_type == "USER" and tz.visibility == "USER" and tz.object == "europe/berlin"
        assert tz.evidence and tz.evidence[0].source_type == "message"
        got = await user.advanced.memories.get(tz.memory_id)
        assert got.memory_id == tz.memory_id and got.confidence > 0
        mem_ids["timezone"] = tz.memory_id
        mem_ids["editor"] = preds["favourite_editor"].memory_id
        return f"{len(mems)} memories: {sorted(preds)}"

    await c.step("POST /v1/messages (events), GET /v1/memories, GET /v1/memories/{id}", observe)

    async def consolidate() -> str:
        nonlocal before_correction
        await asyncio.sleep(0.05)
        before_correction = datetime.now(UTC)
        await asyncio.sleep(0.05)
        await user.history.add([("USER", "Actually, my timezone is now America/New_York.")])
        # the same fact again -> reinforce
        await user.history.add([("USER", "My timezone is America/New_York.")])
        await asyncio.sleep(0.5)
        mems = await user.advanced.memories.list()
        tz = [m for m in mems if m.predicate == "timezone"]
        assert len(tz) == 1 and tz[0].object == "america/new_york", [m.content for m in tz]
        assert tz[0].reinforcement_count >= 2
        history = await user.advanced.memories.list(include_superseded=True)
        old = next(m for m in history if m.memory_id == mem_ids["timezone"])
        assert old.temporal_status == "SUPERSEDED" and old.superseded_by == tz[0].memory_id
        items = await user.search("what is my timezone", kinds=["memory"])
        # the superseded fact is never served; the turns themselves stay searchable as what
        # was said when (a temporal question needs the old one), so "Berlin" may appear in
        # a verbatim turn but not as the timezone fact
        assert any("New_York" in i.text for i in items)
        assert not any(i.id == mem_ids["timezone"] for i in items)
        return "Berlin -> New_York superseded (history kept), repeat reinforced, recall serves only CURRENT facts"

    await c.step("consolidation: supersede + reinforce + temporal history", consolidate)

    async def forget() -> str:
        await user.forget(mem_ids["editor"])
        await user.forget(mem_ids["editor"])  # idempotent
        mems = await user.advanced.memories.list()
        assert not any(m.memory_id == mem_ids["editor"] for m in mems)
        items = await user.search("favourite editor", kinds=["memory"])
        # the fact is gone; the verbatim turn it was read from is not "unsaid" (forget it
        # too to remove the sentence) - the same contract tests/e2e/test_memory_flow.py holds
        assert not any(i.id == mem_ids["editor"] for i in items)
        try:
            await user.advanced.memories.get(mem_ids["editor"])
            raise AssertionError("forgotten memory still readable")
        except (NotFoundError, AuthorizationError):
            pass
        return "DELETE /v1/memories/{id}: gone from list, recall and get"

    await c.step("DELETE /v1/memories/{id} (forget everywhere)", forget)

    # ---------------------------------------------------------------- agents
    print("\n## Multi-agent semantics (run lineage, sharing)")

    async def agents() -> str:
        crew = memory.bind(
            **{
                **user.scope.model_dump(exclude_none=True),
                "agent_group_id": "crew",
                "turn_id": f"trn-{RUN}-2",
            }
        )
        planner = crew.agent("planner")
        writer = planner.agent("writer")  # child run
        reviewer = planner.agent("reviewer")  # sibling of writer
        await planner.remember("Plan: split the brief into revenue and cost.", visibility="RUN")
        await asyncio.sleep(0.3)
        q = "plan for the brief sections"
        seen_by_child = await writer.search(q, kinds=["memory"])
        seen_by_user = await crew.search(q, kinds=["memory"])
        seen_by_stranger = await crew.agent("intern").search(q, kinds=["memory"])
        assert any("Plan:" in i.text for i in seen_by_child)
        assert not any("Plan:" in i.text for i in seen_by_user)
        assert not any("Plan:" in i.text for i in seen_by_stranger)
        planner_mems = await planner.advanced.memories.list()
        assert planner_mems and planner_mems[0].visibility == "RUN"
        # explicit sharing with the agent group: every agent of the crew reads it
        fact = "Revenue was EUR 412 million in FY26."
        await planner.remember(fact, memory_type="SHARED", visibility="AGENT_GROUP")
        await asyncio.sleep(0.3)
        auditor = crew.agent("auditor")
        for reader in (auditor, reviewer):
            shared = await reader.search("FY26 revenue", kinds=["memory"])
            assert any("412" in i.text for i in shared)
        revenue = next(i for i in shared if "412" in i.text)
        details = await auditor.advanced.memories.get(revenue.id)
        assert details.visibility == "AGENT_GROUP"
        return (
            "child run reads parent's RUN memory, user/sibling/stranger do not; an AGENT_GROUP "
            "memory reaches the crew"
        )

    await c.step("agent runs, AGENT_GROUP sharing", agents)

    # ---------------------------------------------------------------- graph
    print("\n## Knowledge graph")

    async def graph() -> str:
        kg = user.advanced.graph

        async def around(text: str, **options: Any) -> tuple[Any, list[Any]]:
            """The first entity ``text`` names, and the facts at and around it."""
            found = await kg.entities(text)
            assert found, f"no entity for {text!r}"
            profile = await kg.entity(found[0].entity_id, depth=options.pop("depth", 1), **options)
            hood = profile.neighborhood.facts if profile.neighborhood else []
            return found[0], [*profile.relations, *profile.history, *hood]

        metric, answer = await around("Adjusted EBITDA")
        assert metric.entity_type == "METRIC", metric
        facts = {(f.predicate, f.object): f for f in answer}
        value = facts[("has_value", "EUR 98 million")]
        assert value.attributes["period"] == "FY26", value.attributes
        assert value.attributes["change"] == "+21%", value.attributes
        assert ("would_have_value", "EUR 91 million") in facts, sorted(facts)
        assert ("excludes", "Litigation settlement") in facts, sorted(facts)
        assert value.evidence[0].page == 11, value.evidence
        alias = await kg.entities("ARR")
        assert alias[0].canonical_name == "recurring revenue", alias
        _, free = await around("Who approved the restructuring programme?")
        approved = [(f.predicate, f.object) for f in free if f.predicate == "approved_by"]
        assert ("approved_by", "The Board") in approved, approved
        _, deal = await around("How much did GLOBEX pay for Initech?", depth=2)
        prices = [(f.predicate, f.object) for f in deal if f.predicate == "consideration"]
        assert ("consideration", "USD 210 million") in prices, prices
        # memory-derived facts are temporal (valid time): the current view has only the
        # corrected timezone; a view dated before the correction returns the old value
        me = (await kg.entities("user:amit"))[0]
        now = await kg.entity(me.entity_id)
        tz = [(v.predicate, v.value) for v in now.current if v.predicate == "timezone"]
        assert tz == [("timezone", "america/new_york")], tz
        past = await kg.entity(me.entity_id, depth=1, as_of=before_correction)
        hood = past.neighborhood.facts if past.neighborhood else []
        old = [(f.object, f.status) for f in hood if f.predicate == "timezone"]
        assert old == [("europe/berlin", "SUPERSEDED")], old
        return (
            f"{len(answer)} facts around Adjusted EBITDA (value, change, exclusions, "
            f"counterfactual); ARR -> Recurring Revenue; 2-hop deal price; temporal as_of"
        )

    await c.step("GET /v1/graph/entities[/{id}] (free text, aliases, depth, as_of)", graph)

    # ---------------------------------------------------------------- context propagation
    print("\n## SDK ergonomics")

    async def ergonomics() -> str:
        async with user:
            assert current_context() is user
        child = user.agent("x")
        assert child.scope.parent_agent_run_id is None and child.scope.agent_run_id
        grand = child.agent("y")
        assert grand.scope.parent_agent_run_id == child.scope.agent_run_id
        return "contextvars propagation, agent() lineage"

    await c.step("current_context(), agent()", ergonomics)

    async def delete_thread() -> str:
        await user.history.delete()
        try:
            await user.history.thread()
            raise AssertionError("deleted thread still readable")
        except NotFoundError:
            pass
        return "DELETE /v1/threads/{id}: thread gone (archive retained per policy)"

    await c.step("DELETE /v1/threads/{id}", delete_thread)

    await memory.aclose()
    total = len(c.rows)
    print(f"\n{total - c.failed}/{total} checks passed")
    return 1 if c.failed else 0


if __name__ == "__main__":
    t0 = time.perf_counter()
    code = asyncio.run(tour())
    print(f"({time.perf_counter() - t0:.1f}s)")
    sys.exit(code)
