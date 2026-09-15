"""Network-level load test against a deployed Memory Service (Locust).

    uv run locust -f benchmark/load/locustfile.py --headless -u 20 -r 5 -t 60s \
        --host http://localhost:8080

Environment: ``MEMORY_API_KEY`` (default ``dev``), ``MEMORY_TENANT`` (``acme``),
``MEMORY_USERS`` (comma-separated user ids, default ``u1,u2,u3``). Each simulated user
keeps one thread and alternates chat turns, recall, context bundles and, rarely, a file
upload — the same operations the p95 budgets cover. Compare the Locust percentiles with
``benchmark/results/performance.json``; the difference is the network + server hop.
"""

from __future__ import annotations

import json
import os
import random
import uuid

from locust import HttpUser, between, task

API_KEY = os.environ.get("MEMORY_API_KEY", "dev")
TENANT = os.environ.get("MEMORY_TENANT", "acme")
USERS = os.environ.get("MEMORY_USERS", "u1,u2,u3").split(",")
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


class MemoryUser(HttpUser):
    wait_time = between(0.2, 1.0)

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
            json={"scope": self.scope, "query": random.choice(QUERIES)},
            name="POST /v1/recall",
        )

    @task(3)
    def context(self) -> None:
        self.client.post(
            "/v1/context",
            headers=self.headers,
            json={"scope": self.scope, "query": random.choice(QUERIES)},
            name="POST /v1/context",
        )

    @task(1)
    def upload(self) -> None:
        salt = f"\n<!-- {uuid.uuid4().hex} -->\n".encode()
        self.client.post(
            "/v1/files",
            headers=self.headers,
            files={"file": ("note.md", DOC + salt, "text/markdown")},
            data={"scope": json.dumps(self.scope), "title": "load note"},
            name="POST /v1/files",
        )

    @task(1)
    def history(self) -> None:
        self.client.get(
            f"/v1/threads/{self.scope['thread_id']}/messages",
            headers=self.headers,
            name="GET /v1/threads/{id}/messages",
        )
