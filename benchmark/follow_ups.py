"""Follow-up retrieval: does a short question that points back ("and the second one?", "who
runs that?") retrieve what it points at, and does a short question on a new topic still
retrieve its own? The memory context is asked the way a harness asks it: the thread holds the
previous question and its answer, the query is the new question alone.

In-process, real weights (the frozen ONNX set under ``./models``), real PostgreSQL, qdrant
local mode in a fresh directory, a fresh tenant per run. A database of its own, migrated::

    createdb memory_bench_followups
    MEMORY__DATABASE__URL=postgresql+psycopg://memory:memory@localhost:5432/memory_bench_followups \\
        uv run alembic upgrade head
    BENCH_FOLLOWUPS_DB=postgresql+psycopg://memory:memory@localhost:5432/memory_bench_followups \\
        uv run python -m benchmark.follow_ups

Arms: ``off`` (the question alone, as before), ``shipped`` (a question of at most
``FOLLOW_UP_WORDS`` words whose own search is weak, best similarity under
``FOLLOW_UP_FLOOR``, is searched again with the thread's previous question), ``always`` (every
such short question is, weak or not: plain concatenation). A case is scored on
where its gold memory lands among the packed items (by relevance): packed at all, in the
top three, and the reciprocal rank (MRR).
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
import uuid
from typing import Any

from memory_service import __version__
from memory_service.application.container import Overrides, build_container
from memory_service.config.settings import Settings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType, MessageRole
from memory_service.modules.context import builder as context_builder

DB = os.environ.get(
    "BENCH_FOLLOWUPS_DB",
    "postgresql+psycopg://memory:memory@localhost:5432/memory_bench_followups",
)

#: What the user told the assistant over time: one store, every case's distractors.
MEMORIES = (
    "The Lisbon offsite is booked at Hotel Avenida from 12 to 14 May.",
    "Marta Silva is running the Lisbon offsite and approves its budget.",
    "Our packaging shortlist is Boxly, CartonCo and PakRight.",
    "CartonCo quoted a lead time of 21 days for the custom mailer boxes.",
    "Boxly quoted 9 days but only ships from Rotterdam.",
    "My dentist is Dr. Okafor at Riverside Dental.",
    "The next dental check-up is on 3 March at 4pm.",
    "The Q3 board deck is due to Priya by 20 September.",
    "Priya wants the board deck to open with churn, not revenue.",
    "Our AWS bill in August was 41,200 dollars, mostly from the analytics cluster.",
    "The analytics cluster runs on eight r6i.4xlarge nodes in eu-west-1.",
    "My daughter Leela starts piano lessons with Mr. Haas on Thursdays.",
    "Mr. Haas charges 45 euros per lesson and wants payment monthly.",
    "The Berlin warehouse lease ends on 31 January 2027.",
    "The landlord of the Berlin warehouse is Kessler Immobilien.",
    "Our tax advisor is Jonas Weber at Weber & Partner.",
    "The VAT return for Q2 was filed on 28 July.",
    "The Acme renewal is worth 380,000 dollars a year over three years.",
    "Acme's procurement contact is Dana Whitfield.",
    "The mobile app release 4.2 is blocked on the payments SDK upgrade.",
    "The payments SDK upgrade is owned by Tomás in the platform team.",
    "I am allergic to penicillin.",
    "My car insurance with Allianz renews on 1 April.",
    "The Allianz policy number is AZ-77120-K.",
    "We hired three engineers in Q2: Ana, Felix and Rui.",
    "Felix joined the data team and reports to Sven.",
    "The office Wi-Fi password rotates on the first Monday of each month.",
    "Our customer NPS for September was 46, up from 39 in August.",
    "Support tickets about late deliveries doubled in September.",
    "The marketing budget for Q4 is 120,000 euros, split evenly across three channels.",
)

#: (previous question, its answer, the follow-up, a phrase of the gold memory)
FOLLOW_UPS = (
    (
        "Where is the Lisbon offsite booked?",
        "Hotel Avenida, 12 to 14 May.",
        "Who is in charge of it?",
        "Marta Silva",
    ),
    (
        "Which suppliers are on our packaging shortlist?",
        "Boxly, CartonCo and PakRight.",
        "What lead time did the second one quote?",
        "21 days",
    ),
    (
        "Who is my dentist?",
        "Dr. Okafor at Riverside Dental.",
        "When am I seeing them next?",
        "3 March",
    ),
    (
        "When is the Q3 board deck due?",
        "20 September, to Priya.",
        "What should it lead with?",
        "open with churn",
    ),
    (
        "How much was our AWS bill in August?",
        "41,200 dollars, mostly the analytics cluster.",
        "What instances does that cluster use?",
        "r6i.4xlarge",
    ),
    (
        "Who teaches Leela piano?",
        "Mr. Haas, on Thursdays.",
        "How much does he charge?",
        "45 euros",
    ),
    (
        "When does the Berlin warehouse lease end?",
        "31 January 2027.",
        "Who is the landlord there?",
        "Kessler Immobilien",
    ),
    (
        "Who is our tax advisor?",
        "Jonas Weber at Weber & Partner.",
        "When did we file the last VAT return?",
        "28 July",
    ),
    (
        "How much is the Acme renewal worth?",
        "380,000 dollars a year over three years.",
        "Who should I talk to there?",
        "Dana Whitfield",
    ),
    (
        "What is blocking the 4.2 app release?",
        "The payments SDK upgrade.",
        "Who owns that?",
        "Tomás",
    ),
    (
        "When does my car insurance renew?",
        "1 April, with Allianz.",
        "What's the policy number?",
        "AZ-77120-K",
    ),
    (
        "Who did we hire in Q2?",
        "Ana, Felix and Rui.",
        "Who does the second one report to?",
        "reports to Sven",
    ),
    (
        "What was our NPS in September?",
        "46, up from 39 in August.",
        "Anything else that got worse that month?",
        "late deliveries",
    ),
    (
        "What's the Q4 marketing budget?",
        "120,000 euros across three channels.",
        "How is it split?",
        "split evenly",
    ),
    (
        "Which supplier quoted 9 days?",
        "Boxly.",
        "Where does it ship from?",
        "Rotterdam",
    ),
)

#: (an unrelated previous question, its answer, a short standalone question, its gold phrase):
#: the topic changed; the previous question must not crowd the new one out.
TOPIC_CHANGES = (
    (
        "Where is the Lisbon offsite booked?",
        "Hotel Avenida.",
        "Am I allergic to anything?",
        "penicillin",
    ),
    ("Who is my dentist?", "Dr. Okafor.", "Who is Acme's procurement contact?", "Dana Whitfield"),
    (
        "How much was our AWS bill in August?",
        "41,200 dollars.",
        "When does the Berlin lease end?",
        "31 January 2027",
    ),
    (
        "Who teaches Leela piano?",
        "Mr. Haas.",
        "What is blocking release 4.2?",
        "blocked on the payments SDK",
    ),
    ("What's the Q4 marketing budget?", "120,000 euros.", "Who is our tax advisor?", "Jonas Weber"),
    (
        "Who did we hire in Q2?",
        "Ana, Felix and Rui.",
        "When is my next dental check-up?",
        "3 March",
    ),
    ("When does my car insurance renew?", "1 April.", "What was September's NPS?", "46"),
    (
        "Who is running the Lisbon offsite?",
        "Marta Silva.",
        "When is the board deck due?",
        "20 September",
    ),
    (
        "What lead time did CartonCo quote?",
        "21 days.",
        "When does the Wi-Fi password rotate?",
        "first Monday",
    ),
    (
        "Who owns the payments SDK upgrade?",
        "Tomás.",
        "Who is the Berlin warehouse landlord?",
        "Kessler",
    ),
    ("When was the Q2 VAT return filed?", "28 July.", "How much does Mr. Haas charge?", "45 euros"),
    (
        "How much is the Acme renewal worth?",
        "380,000 dollars.",
        "Which suppliers are shortlisted for packaging?",
        "Boxly, CartonCo",
    ),
    ("Who does Felix report to?", "Sven.", "What's my Allianz policy number?", "AZ-77120-K"),
    (
        "What does Priya want the deck to open with?",
        "Churn.",
        "Which instances run the analytics cluster?",
        "r6i.4xlarge",
    ),
    ("Where does Boxly ship from?", "Rotterdam.", "Who did we hire in Q2?", "Ana, Felix and Rui"),
)

ARMS = ("off", "shipped", "always")

#: Distractors: a real user's store holds hundreds of statements like the ones above, about
#: other people, places, projects and dates (none of the cases' own).
_PEOPLE = ("Hannah", "Omar", "Lucía", "Kenji", "Grace", "Mateo", "Ingrid", "Ravi", "Chloe", "Yusuf")
_THINGS = (
    "the Madrid roadshow",
    "the Oslo partner summit",
    "the onboarding revamp",
    "the CRM migration",
    "the Q1 pricing review",
    "the Dublin hiring fair",
    "the security audit",
    "the website redesign",
    "the Singapore launch",
    "the annual sales kickoff",
    "the Zurich client dinner",
    "the data retention policy",
    "the support chatbot pilot",
    "the Toronto office move",
    "the loyalty programme",
    "the SOC 2 renewal",
    "the Prague workshop",
    "the invoice backlog",
    "the vendor consolidation",
    "the Austin meetup",
)
_FACTS = (
    "{who} is coordinating {what}.",
    "{what} is scheduled for {day} {month}.",
    "{who} asked for a status update on {what} every Friday.",
    "The budget for {what} is {n},000 euros.",
    "{what} was moved to {day} {month} at {who}'s request.",
    "{who} will present the results of {what} to the leadership team.",
)
_MONTHS = ("January", "February", "April", "June", "August", "October", "November", "December")
DISTRACTORS = tuple(
    template.format(
        who=_PEOPLE[(i + j) % len(_PEOPLE)],
        what=thing,
        day=(i * 7 + j * 3) % 27 + 1,
        month=_MONTHS[(i + j) % len(_MONTHS)],
        n=(i + 2) * (j + 3),
    )
    for i, thing in enumerate(_THINGS)
    for j, template in enumerate(_FACTS)
)


def settings() -> Settings:
    return Settings(
        _env_file=None,
        service={"environment": "test", "log_level": "WARNING", "log_json": False},
        authentication={"trusted_dev_api_keys": ["k"]},
        database={"url": DB},
        models={"llm": {"accept_agent_endpoints": False}},
    )


def ctx(tenant: str, thread: str | None = None) -> MemoryExecutionContext:
    return MemoryExecutionContext(
        tenant_id=tenant, user_id="u", workspace_id="ws", thread_id=thread
    )


async def seed(c: Any, tenant: str) -> None:
    async with c.services["uow_factory"]() as uow:
        for text in (*MEMORIES, *DISTRACTORS):
            await c.services["memory"].remember(
                uow,
                ctx(tenant),
                content=text,
                memory_type=MemoryType.SEMANTIC,
                lifetime=Lifetime.LONG_TERM,
            )
        await uow.commit()
    await c.tasks.drain()


async def turn(c: Any, tenant: str, asked: str, answer: str) -> str:
    """A thread holding one question and its answer, as a harness leaves it."""
    thread = f"thr_{uuid.uuid4().hex[:12]}"
    conversation = c.services["conversation"]
    async with c.services["uow_factory"]() as uow:
        await conversation.create_thread(uow, ctx(tenant, thread))
        for role, text in ((MessageRole.USER, asked), (MessageRole.ASSISTANT, answer)):
            await conversation.append_message(uow, ctx(tenant, thread), role=role, content=text)
        await uow.commit()
    await c.tasks.drain()
    return thread


def arm(name: str) -> None:
    """Set the builder's follow-up rule for ``name`` (module state, one arm at a time)."""
    floor = {"off": -1.0, "shipped": SHIPPED_FLOOR, "always": 2.0}[name]
    context_builder.FOLLOW_UP_FLOOR = floor  # type: ignore[misc]


SHIPPED_FLOOR = context_builder.FOLLOW_UP_FLOOR


async def main() -> dict[str, Any]:
    tenant = f"fu_{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as qdrant:
        c = await build_container(
            settings(),
            __version__,
            overrides=Overrides(
                search_local_path=qdrant,
                authorization="memory",
                tasks="inline",
                cache="disabled",  # no arm reads another's bundle
                blob="memory",
            ),
        )
        try:
            await seed(c, tenant)
            cases = [("follow_up", *f) for f in FOLLOW_UPS] + [
                ("topic_change", *t) for t in TOPIC_CHANGES
            ]
            threads = [await turn(c, tenant, asked, answer) for _, asked, answer, _, _ in cases]
            report: dict[str, Any] = {}
            builder = c.services["context_builder"]
            for name in ARMS:
                arm(name)
                hits: dict[str, list[bool]] = {"follow_up": [], "topic_change": []}
                top3: dict[str, list[bool]] = {"follow_up": [], "topic_change": []}
                rr: dict[str, list[float]] = {"follow_up": [], "topic_change": []}
                packed_n: list[int] = []
                tokens: list[int] = []
                ranks: list[int | None] = []
                sims: list[float] = []
                ms: list[float] = []
                misses: list[str] = []
                for (kind, _, _, question, gold), thread in zip(cases, threads, strict=True):
                    start = time.perf_counter()
                    bundle = await builder.build(ctx(tenant, thread), question, window=False)
                    ms.append((time.perf_counter() - start) * 1000)
                    [memory] = [m for m in MEMORIES if gold in m]  # its own, not a chat's echo
                    packed = sorted(
                        (*bundle.memories, *bundle.knowledge, *bundle.graph_facts),
                        key=lambda item: -item.relevance,
                    )
                    rank = next(
                        (i for i, item in enumerate(packed, 1) if memory in item.text), None
                    )
                    hit = rank is not None
                    hits[kind].append(hit)
                    top3[kind].append(rank is not None and rank <= 3)
                    rr[kind].append(1 / rank if rank else 0.0)
                    packed_n.append(len(packed))
                    tokens.append(bundle.token_estimate)
                    ranks.append(rank)
                    sims.append(round(max((i.relevance for i in packed), default=0.0), 3))
                    if not hit:
                        misses.append(f"{kind}: {question}")
                report[name] = (
                    {f"{kind} packed": f"{sum(v)}/{len(v)}" for kind, v in hits.items()}
                    | {f"{kind} top3": f"{sum(v)}/{len(v)}" for kind, v in top3.items()}
                    | {f"{kind} mrr": round(statistics.mean(v), 3) for kind, v in rr.items()}
                    | {
                        "items_packed_mean": round(statistics.mean(packed_n), 1),
                        "tokens_mean": round(statistics.mean(tokens)),
                        "ranks": ranks,
                        "top_relevance": sims,
                        "p50_ms": round(statistics.median(ms), 1),
                        "p95_ms": round(sorted(ms)[int(len(ms) * 0.95) - 1], 1),
                        "misses": misses,
                    }
                )
            return report
        finally:
            await c.close()


if __name__ == "__main__":
    # the service logs to stdout: the report goes to its own file
    out = sys.argv[1] if len(sys.argv) > 1 else "benchmark/results/follow_ups.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(asyncio.run(main()), f, indent=2)
