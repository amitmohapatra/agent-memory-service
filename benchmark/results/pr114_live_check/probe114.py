"""probe114.py: PR #114 live check without a benchmark. The memory service's own short model
calls (turn restatement, conflict adjudication, entity summary), with the service's own prompts,
schemas and answer sizes, sent to Haiku through EndpointLLM twice: thinking_room=0 (main's
request limit) and thinking_room=8000 (the PR). Counts calls that fail or return no text, and
the tokens the provider billed (read from each response's usage)."""

import asyncio
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import benchmark.longmemeval_retrieval as lmr
import httpx

from memory_service.adapters.models.llm import EndpointLLM, LLMCallFailed
from memory_service.config.constants import LLM
from memory_service.modules.graph import summaries as graph_summaries
from memory_service.modules.memory import native as memory_native
from memory_service.modules.memory import restatement
from memory_service.ports.models import LLMMessage

HAIKU = "claude-haiku-5-5"
USAGE = {"in": 0, "out": 0, "calls": 0}


class Counting(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.inner = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self.inner.handle_async_request(request)
        body = await response.aread()
        try:
            usage = json.loads(body).get("usage") or {}
            USAGE["in"] += int(usage.get("prompt_tokens") or 0)
            USAGE["out"] += int(usage.get("completion_tokens") or 0)
            USAGE["calls"] += 1
        except ValueError:
            pass
        headers = [
            (k, v)
            for k, v in response.headers.items()
            if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")
        ]
        return httpx.Response(response.status_code, headers=headers, content=body, request=request)


TURNS = [
    (
        "Mara",
        "Jon",
        "I finally signed up for the pottery class at the community centre, it starts next Tuesday!",
    ),
    ("Jon", "Mara", "That's great. I spent yesterday fixing the bike I bought from my neighbour."),
    (
        "Mara",
        "Jon",
        "My sister Lena moved to Lisbon last month for her new job at a design studio.",
    ),
    ("Jon", "Mara", "We adopted a second cat two weeks ago, a grey one we named Pebble."),
    ("Mara", "Jon", "I ran my first half marathon on Sunday in 2 hours and 4 minutes."),
    ("Jon", "Mara", "I'm thinking about switching from nursing to teaching biology next year."),
    ("Mara", "Jon", "Remember the lake trip? We're going back there with my parents in August."),
    ("Jon", "Mara", "My grandmother turned ninety yesterday, the whole family came over."),
    ("Mara", "Jon", "I started volunteering at the animal shelter on Saturday mornings."),
    ("Jon", "Mara", "The band I play bass in has a gig at the Blue Room next Friday."),
    (
        "Mara",
        "Jon",
        "I've been reading a lot of Ursula Le Guin lately, The Dispossessed is my favourite.",
    ),
    ("Jon", "Mara", "We painted the kids' room green last weekend, it took two days."),
    ("Mara", "Jon", "My doctor told me to cut back on coffee, so I'm down to one cup a day."),
    ("Jon", "Mara", "I got the promotion! I'll be leading the night shift team from March."),
    ("Mara", "Jon", "Lena is visiting us for Christmas, she's bringing her boyfriend Tomas."),
]
PAIRS = [
    ("Mara lives in Porto", "Mara moved to Lisbon in May"),
    ("Jon has one cat named Miso", "Jon has two cats, Miso and Pebble"),
    ("Mara drinks four cups of coffee a day", "Mara drinks one cup of coffee a day"),
    ("Jon works as a nurse", "Jon plans to teach biology next year"),
    ("Mara's favourite book is The Dispossessed", "Mara loves The Dispossessed by Le Guin"),
    ("Jon plays guitar in a band", "Jon plays bass in a band"),
    ("Mara volunteers at the shelter on Sundays", "Mara volunteers at the shelter on Saturdays"),
    ("Jon leads the day shift", "Jon leads the night shift from March"),
    ("Lena works at a bank", "Lena works at a design studio in Lisbon"),
    ("Mara's half marathon time was 2:10", "Mara ran a half marathon in 2 hours 4 minutes"),
]
FACTS = [
    "Mara (PERSON): sister of Lena; ran half marathon; volunteers at animal shelter; reads Ursula Le Guin",
    "Jon (PERSON): owns cat Pebble; plays bass in band; works as nurse; leads night shift",
    "Lena (PERSON): moved to Lisbon; works at design studio; dating Tomas",
    "Pebble (ANIMAL): adopted by Jon; grey cat",
    "Blue Room (PLACE): hosts gig of Jon's band",
    "pottery class (EVENT): starts Tuesday; at community centre",
    "Lisbon (LOCATION): home of Lena",
    "Tomas (PERSON): boyfriend of Lena; visiting at Christmas",
    "half marathon (EVENT): run by Mara on Sunday; time 2 hours 4 minutes",
    "The Dispossessed (WORK): favourite book of Mara; written by Ursula Le Guin",
]


async def arm(room: int) -> dict:
    settings = lmr._settings().llm
    tuning = replace(LLM, model=HAIKU, fast_model=HAIKU, thinking_room=room)
    llm = EndpointLLM(settings, http=Counting(), tuning=tuning)
    report: dict[str, dict[str, int]] = {}

    async def call(use: str, coro) -> None:
        row = report.setdefault(use, {"calls": 0, "failed": 0, "empty": 0})
        row["calls"] += 1
        try:
            out = await coro
        except LLMCallFailed:
            row["failed"] += 1
            return
        if not out or (hasattr(out, "text") and not out.text):
            row["empty"] += 1

    said = datetime(2026, 5, 4, 18, 0, tzinfo=UTC)
    before = None
    for i, (speaker, to, text) in enumerate(TURNS):
        payload = {
            "said_on": (said + timedelta(days=i)).date().isoformat(),
            "weekday": (said + timedelta(days=i)).strftime("%A"),
            "dates": {},
            "previous_turn": before,
            "turn": {"speaker": speaker, "addressee": to, "text": text},
        }
        before = {"speaker": speaker, "text": text}
        messages = [
            LLMMessage(role="system", content=restatement.SYSTEM),
            LLMMessage(role="user", content=json.dumps(payload)),
        ]
        await call(
            restatement.USE,
            llm.structured(
                messages, schema=restatement.SCHEMA, max_tokens=500, use=restatement.USE
            ),
        )
    for old, new in PAIRS:
        user = (
            f"NEW (by user:mara, observed 2026-05-10T10:00:00+00:00):\n{new}\ntriple: - / - / -\n\n"
            f"EXISTING (by user:mara, observed 2026-03-01T10:00:00+00:00):\n{old}\ntriple: - / - / -"
        )
        messages = [
            LLMMessage(role="system", content=memory_native._ADJUDICATION_SYSTEM),
            LLMMessage(role="user", content=user),
        ]
        await call(
            "conflict_adjudication",
            llm.structured(
                messages,
                schema=memory_native._ADJUDICATION_SCHEMA,
                max_tokens=50,
                use="conflict_adjudication",
            ),
        )
    for line in FACTS:
        messages = [
            LLMMessage(role="system", content=graph_summaries._SYSTEM),
            LLMMessage(role="user", content=line),
        ]
        await call("summaries", llm.complete(messages, max_tokens=256, use="summaries"))
    await llm.close()
    return report


async def main() -> dict:
    out = {}
    for room in (0, 8000):
        before = dict(USAGE)
        out[f"thinking_room={room}"] = await arm(room)
        tokens_in, tokens_out = USAGE["in"] - before["in"], USAGE["out"] - before["out"]
        out[f"thinking_room={room}"]["_usage"] = {
            "requests": USAGE["calls"] - before["calls"],
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "usd": round(tokens_in * 0.1e-6 + tokens_out * 0.5e-6, 4),
        }
    return out


result = json.dumps(asyncio.run(main()), indent=1)
Path(sys.argv[1]).write_text(result)
print(result)
