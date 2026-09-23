"""Network-level load test against a deployed Memory Service (Locust).

    uv run locust -f benchmark/load/locustfile.py --headless -u 20 -r 20 -t 300s \
        --host http://localhost:8080

Every user issues exactly one request per second (``constant_throughput(1.0)``), so the
offered rate is the user count and a run can be aimed at a target: twenty users is twenty
requests per second whatever the server's latency does. The first version used a random
think time of 0.2-1.0 s, which meant the rate drifted with the response time - the load
test slowed down exactly when the server did, and the measured throughput said as much
about Locust as about the service.

Environment: ``MEMORY_API_KEY`` (default ``dev``), ``MEMORY_TENANT`` (``acme``),
``MEMORY_USERS`` (comma-separated user ids, default ``u1,u2,u3``), ``MEMORY_LOAD_ARM``:

* ``cold`` (the default) gives every query a unique salt, so the bundle cache always
  misses and the measurement is of the work, not of the cache.
* ``warm`` draws from a small fixed set, so bundles are served from cache - the number an
  operator sees on a hot path, and useless as a budget.

Each simulated user keeps one thread and alternates chat turns, recall, context bundles
and, rarely, a file upload - the same operations the p95 budgets cover. Compare the Locust
percentiles with ``benchmark/results/performance.json``; the difference is the network and
server hop.
"""

from __future__ import annotations

import json
import os
import random
import uuid
from pathlib import Path

from locust import HttpUser, constant_throughput, task

API_KEY = os.environ.get("MEMORY_API_KEY", "dev")
TENANT = os.environ.get("MEMORY_TENANT", "acme")
USERS = os.environ.get("MEMORY_USERS", "u1,u2,u3").split(",")
ARM = os.environ.get("MEMORY_LOAD_ARM", "cold").strip().lower()
QUERIES = [
    "Why did Adjusted EBITDA increase despite lower revenue?",
    "What is the FY26 revenue?",
    "What did I say about my timezone?",
    "Summarize the restructuring programme",
    "Who is the CFO?",
]
FACTS = [
    "My timezone is Europe/Berlin.",
    "I prefer concise answers.",
    "We decided to use PostgreSQL as the canonical store.",
    "Revenue was EUR 412 million in FY26.",
]
DOC = b"# Load test note\n\nRevenue was EUR 412 million in FY26. Adjusted EBITDA rose 8%.\n"

#: Which document the upload task sends. ``text`` is markdown, which the builtin parser
#: handles; ``pdf`` is a real 39 KB Federal Reserve release, which is the only way to reach
#: docling's layout and table models - the most expensive thing the service loads.
#:
#: It defaults to ``text`` so this run stays comparable with every capacity number taken
#: before it. That default is also why those numbers describe traffic without documents in
#: it: markdown never enters docling, so no measurement so far has priced the heavy path.
DOCS = os.environ.get("MEMORY_LOAD_DOCS", "text").strip().lower()
PDF_PATH = Path(__file__).resolve().parents[2] / "tests/fixtures/fed_beige_book_2024_01.pdf"
PDF = PDF_PATH.read_bytes() if DOCS == "pdf" and PDF_PATH.is_file() else b""


def _query() -> str:
    """A question for one request. The cold arm salts it so no two requests can share a
    bundle-cache entry; the warm arm repeats, which is what a cache hit looks like."""
    question = random.choice(QUERIES)
    return f"{question} [{uuid.uuid4().hex[:8]}]" if ARM == "cold" else question


class MemoryUser(HttpUser):
    #: one request per second per user, so the offered rate is the user count
    wait_time = constant_throughput(1.0)

    def on_start(self) -> None:
        self.user_id = random.choice(USERS)
        self.headers = {
            "X-API-Key": API_KEY,
            "X-Memory-Tenant": TENANT,
            "X-Memory-User": self.user_id,
        }
        self.scope = {
            "thread_id": f"thr_load_{uuid.uuid4().hex}",
            "session_id": f"ses_load_{uuid.uuid4().hex}",
            "turn_id": f"trn_load_{uuid.uuid4().hex}",
        }
        self.turn = 0
        # The thread must exist before anything reads it: the history task used to run
        # first on a freshly minted thread id and score a 404 as a server failure.
        self.chat()

    @task(5)
    def chat(self) -> None:
        self.turn += 1
        self.client.post(
            "/v1/messages",
            headers={**self.headers, "Idempotency-Key": f"{self.scope['thread_id']}-{self.turn}"},
            json={
                "scope": self.scope,
                "role": "USER",
                "content": f"{random.choice(FACTS)} (turn {self.turn})",
            },
            name="POST /v1/messages",
        )

    @task(3)
    def recall(self) -> None:
        self.client.post(
            "/v1/recall",
            headers=self.headers,
            json={"scope": self.scope, "query": _query()},
            name="POST /v1/recall",
        )

    @task(3)
    def context_bundle(self) -> None:
        self.client.post(
            "/v1/context",
            headers=self.headers,
            json={"scope": self.scope, "query": _query()},
            name="POST /v1/context",
        )

    @task(1)
    def context_with_verification(self) -> None:
        """The same bundle, plus the grounding cascade over a short answer.

        This is the only task that touches the NLI model, and it is weighted like the
        upload on purpose: verification is opt-in per request, so a capacity number that
        assumed every request paid for it would size the box for traffic nobody sends -
        and one that never exercised it would miss the most expensive thing the service
        can be asked to do.
        """
        self.client.post(
            "/v1/context",
            headers=self.headers,
            json={
                "scope": self.scope,
                "query": _query(),
                "answer": "Revenue was EUR 412 million in FY26 and Adjusted EBITDA rose 8%.",
            },
            name="POST /v1/context (verified)",
        )

    @task(1)
    def upload(self) -> None:
        """One document per call, salted so content-hash dedup cannot skip the parse.

        A repeated upload is recognised as the duplicate it is and never reaches the parser,
        which would make this task measure the idempotency check rather than ingestion. The
        markdown salt is an HTML comment; the PDF salt is a ``%`` comment line after the
        trailer, which readers ignore.
        """
        if PDF:
            body = PDF + b"\n%" + uuid.uuid4().hex.encode() + b"\n"
            payload = ("note.pdf", body, "application/pdf")
        else:
            body = DOC + f"\n<!-- {uuid.uuid4().hex} -->\n".encode()
            payload = ("note.md", body, "text/markdown")
        self.client.post(
            "/v1/files",
            headers=self.headers,
            files={"file": payload},
            data={"scope": json.dumps(self.scope), "title": "load note"},
            name=f"POST /v1/files ({'pdf' if PDF else 'text'})",
        )

    @task(1)
    def history(self) -> None:
        self.client.get(
            f"/v1/threads/{self.scope['thread_id']}/messages",
            headers=self.headers,
            name="GET /v1/threads/{id}/messages",
        )
