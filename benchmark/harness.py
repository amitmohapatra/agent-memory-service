"""Pieces shared by the HTTP-driven gate producers (``durability``, ``performance`` and
``deployed``): the acknowledged request stream, its verification, and the five budgeted
operations. The producers differ only in what sits behind the ``httpx`` client (in-process
ASGI or a TCP connection to a deployed instance) and in how they wait for background work.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from benchmark.retrieval import FIXTURES, _pct
from memory_service.config.settings import Settings
from memory_service.domain.ids import new_id
from memory_service.modules.evaluation.golden import GoldenSet

H = {"X-API-Key": "bench", "X-Memory-Tenant": "acme", "X-Memory-User": "u1"}
FACTS = [
    "My timezone is Europe/Berlin.",
    "I prefer concise answers.",
    "We decided to use PostgreSQL as the canonical store.",
    "Revenue was EUR 412 million in FY26.",
    "My favourite editor is neovim.",
    "I work at ACME Corp.",
]
BUDGET_KEYS = (
    "chat_accept_p95_ms",
    "cached_context_p95_ms",
    "recall_p95_ms",
    "context_bundle_p95_ms",
    "file_accept_p95_ms",
)
LATENCY_SERIES = ("chat", "cached", "recall", "context", "file")
FIXTURE_REPORT = FIXTURES / "acme_fy26_annual_report.md"

Settle = Callable[[], Awaitable[Any]]


def headers(api_key: str = "bench", tenant: str = "acme", user: str = "u1") -> dict[str, str]:
    return {"X-API-Key": api_key, "X-Memory-Tenant": tenant, "X-Memory-User": user}


def new_scope() -> dict[str, str]:
    return {
        "thread_id": new_id("thread"),
        "session_id": new_id("session"),
        "turn_id": new_id("turn"),
    }


def budgets_ms(settings: Settings) -> dict[str, float]:
    return {key: getattr(settings.budgets, key) for key in BUDGET_KEYS}


def stats(xs: list[float]) -> dict[str, Any]:
    return {
        "p50": _pct(xs, 50),
        "p95": _pct(xs, 95),
        "p99": _pct(xs, 99),
        "max": round(max(xs), 2) if xs else 0.0,
        "samples": len(xs),
    }


async def post_with_retry(
    client: httpx.AsyncClient, path: str, hdrs: dict[str, str], **kw: Any
) -> httpx.Response:
    for attempt in range(3):
        try:
            return await client.post(path, headers=hdrs, **kw)
        except httpx.TransportError:
            await asyncio.sleep(0.05 * (attempt + 1))
    return await client.post(path, headers=hdrs, **kw)


async def get_with_retry(
    client: httpx.AsyncClient, path: str, hdrs: dict[str, str], **kw: Any
) -> httpx.Response:
    for attempt in range(3):
        try:
            return await client.get(path, headers=hdrs, **kw)
        except httpx.TransportError:
            await asyncio.sleep(0.5 * (attempt + 1))
    return await client.get(path, headers=hdrs, **kw)


async def timed(
    client: httpx.AsyncClient, method: str, path: str, hdrs: dict[str, str], **kw: Any
) -> tuple[float, int]:
    t = time.perf_counter()
    r = await client.request(method, path, headers=hdrs, **kw)
    return (time.perf_counter() - t) * 1000, r.status_code


# --------------------------------------------------------------------- acknowledged stream


@dataclass
class Acked:
    """Everything the service acknowledged (202) while the stream ran, and what it refused."""

    messages: dict[str, dict[str, Any]] = field(default_factory=dict)
    observations: dict[str, dict[str, Any]] = field(default_factory=dict)
    uploads: dict[str, dict[str, Any]] = field(default_factory=dict)
    refused: dict[str, int] = field(
        default_factory=lambda: {"messages": 0, "observations": 0, "files": 0}
    )
    errors: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "messages": len(self.messages),
            "observations": len(self.observations),
            "files": len(self.uploads),
        }


async def drive_stream(
    client: httpx.AsyncClient,
    hdrs: dict[str, str],
    scopes: list[dict[str, str]],
    *,
    messages: int,
    files: int,
    fixture: bytes,
    rng: random.Random,
    on_step: Callable[[int, int], Awaitable[None]] | None = None,
    concurrency: int = 16,
) -> Acked:
    """Concurrent chat messages (an observation every third turn) and file uploads over
    the given client. ``on_step(sent, total)`` runs under a lock before each chat turn so a
    caller can drive a fault schedule off the stream position."""
    acked = Acked()
    total = len(scopes) * messages
    sem = asyncio.Semaphore(concurrency)
    sent = 0
    lock = asyncio.Lock()

    async def step() -> None:
        nonlocal sent
        async with lock:
            sent += 1
            if on_step is not None:
                await on_step(sent, total)

    async def chat(scope: dict[str, str], i: int) -> None:
        async with sem:
            await step()
            content = f"{rng.choice(FACTS)} (turn {i} of {scope['thread_id'][-6:]})"
            r = await post_with_retry(
                client,
                "/v1/messages",
                hdrs,
                json={"scope": scope, "role": "USER", "content": content},
            )
            if r.status_code == 202:
                body = r.json()
                acked.messages[body["message_id"]] = {
                    "thread_id": scope["thread_id"],
                    "content": content,
                    "job_ids": body.get("job_ids", []),
                }
            elif r.status_code in (503, 429):
                acked.refused["messages"] += 1
            else:
                acked.errors.append(f"messages {r.status_code}: {r.text[:200]}")
            if i % 3 == 0:
                r = await post_with_retry(
                    client,
                    "/v1/observations",
                    hdrs,
                    json={"scope": scope, "kind": "EVENT", "content": rng.choice(FACTS)},
                )
                if r.status_code == 202:
                    acked.observations[r.json()["observation_id"]] = {
                        "thread_id": scope["thread_id"]
                    }
                elif r.status_code in (503, 429):
                    acked.refused["observations"] += 1
                else:
                    acked.errors.append(f"observations {r.status_code}: {r.text[:200]}")

    async def upload(n: int) -> None:
        async with sem:
            scope = scopes[n % len(scopes)]
            salt = f"\n\n<!-- durability copy {n} -->\n".encode()
            r = await post_with_retry(
                client,
                "/v1/files",
                hdrs,
                files={"file": (f"report_{n}.md", fixture + salt, "text/markdown")},
                data={"scope": json.dumps(scope), "title": f"report {n}"},
            )
            if r.status_code == 202:
                acked.uploads[r.json()["document_id"]] = {"n": n}
            elif r.status_code in (503, 429):
                acked.refused["files"] += 1
            else:
                acked.errors.append(f"files {r.status_code}: {r.text[:200]}")

    tasks = [chat(s, i) for s in scopes for i in range(messages)]
    tasks += [upload(n) for n in range(files)]
    rng.shuffle(tasks)
    await asyncio.gather(*tasks)
    return acked


# ------------------------------------------------------------------------- verification


def check_message_listing(
    listing: dict[str, Any],
    status_code: int,
    thread_id: str,
    acked_messages: dict[str, dict[str, Any]],
) -> list[str]:
    """Every acknowledged message of the thread is in the listing with its content."""
    got = {m["message_id"]: m for m in listing.get("messages", [])}
    lost: list[str] = []
    for mid, info in acked_messages.items():
        if info["thread_id"] != thread_id:
            continue
        m = got.get(mid)
        if m is None:
            lost.append(f"message {mid} missing from listing ({status_code})")
        elif m["content"] != info["content"]:
            lost.append(f"message {mid} content mismatch: {m['content'][:40]!r}")
    return lost


def check_document(document_id: str, status_code: int, doc: dict[str, Any] | None) -> list[str]:
    if status_code != 200 or doc is None:
        return [f"document {document_id}: {status_code}"]
    if doc["status"] not in ("READY",) or doc["archive_status"] != "ARCHIVED":
        return [f"document {document_id}: {doc['status']}/{doc['archive_status']}"]
    return []


def duplicate_memories(memories: list[dict[str, Any]]) -> int:
    """Duplicate side effects: one memory per distinct fact per thread at most."""
    seen: dict[tuple[str, str], int] = {}
    for m in memories:
        key = (m.get("predicate") or "", m.get("object") or "")
        seen[key] = seen.get(key, 0) + 1
    return sum(v - 1 for v in seen.values() if v > 1)


async def verify_messages(
    client: httpx.AsyncClient,
    hdrs: dict[str, str],
    scopes: list[dict[str, str]],
    acked_messages: dict[str, dict[str, Any]],
) -> list[str]:
    lost: list[str] = []
    for scope in scopes:
        r = await get_with_retry(
            client, f"/v1/threads/{scope['thread_id']}/messages", hdrs, params={"limit": 500}
        )
        body = r.json() if r.status_code == 200 else {}
        lost += check_message_listing(body, r.status_code, scope["thread_id"], acked_messages)
    return lost


async def verify_documents(
    client: httpx.AsyncClient, hdrs: dict[str, str], acked_uploads: dict[str, dict[str, Any]]
) -> list[str]:
    lost: list[str] = []
    for doc_id in acked_uploads:
        r = await get_with_retry(client, f"/v1/documents/{doc_id}", hdrs)
        lost += check_document(doc_id, r.status_code, r.json() if r.status_code == 200 else None)
    return lost


async def count_duplicate_memories(
    client: httpx.AsyncClient, hdrs: dict[str, str], scopes: list[dict[str, str]]
) -> int:
    duplicates = 0
    for scope in scopes:
        r = await get_with_retry(
            client, "/v1/memories", hdrs, params={"thread_id": scope["thread_id"]}
        )
        duplicates += duplicate_memories(r.json().get("memories", []))
    return duplicates


# ------------------------------------------------------------------ budgeted operations


async def build_corpus(
    client: httpx.AsyncClient,
    hdrs: dict[str, str],
    scope: dict[str, str],
    golden: GoldenSet,
    copies: int,
) -> list[str]:
    """``copies`` salted copies of each golden document; returns the accepted document ids."""
    document_ids: list[str] = []
    for alias, filename in golden.documents.items():
        data = (FIXTURES / filename).read_bytes()
        for n in range(copies):
            salt = b"" if n == 0 else f"\n\n<!-- perf copy {n} -->\n".encode()
            r = await client.post(
                "/v1/files",
                headers=hdrs,
                files={"file": (filename, data + salt, "text/markdown")},
                data={"scope": json.dumps(scope), "title": f"{alias}-{n}"},
            )
            assert r.status_code == 202, r.text
            document_ids.append(r.json()["document_id"])
    return document_ids


async def measure_budgeted(
    client: httpx.AsyncClient,
    hdrs: dict[str, str],
    scope: dict[str, str],
    queries: list[str],
    fixture: bytes,
    requests: int,
    settle: Settle,
) -> tuple[dict[str, list[float]], dict[str, dict[int, int]]]:
    """The five budgeted operations, in the order and mix of ``performance.json``.
    ``settle`` lets background work finish (in-process drain or a wait on the deployed
    queue) after every ten chat turns, after the chat phase and at the end."""
    lat: dict[str, list[float]] = {k: [] for k in LATENCY_SERIES}
    statuses: dict[str, dict[int, int]] = {k: {} for k in lat}

    def record(series: str, ms: float, code: int) -> None:
        lat[series].append(ms)
        statuses[series][code] = statuses[series].get(code, 0) + 1

    # 1. chat accept
    for i in range(requests):
        ms, code = await timed(
            client,
            "POST",
            "/v1/messages",
            hdrs,
            json={
                "scope": scope,
                "role": "USER",
                "content": f"Turn {i}: {queries[i % len(queries)]}",
            },
        )
        record("chat", ms, code)
        if i % 10 == 9:
            await settle()
    await settle()
    await settle()
    # 2. recall
    for i in range(requests):
        ms, code = await timed(
            client,
            "POST",
            "/v1/recall",
            hdrs,
            json={"scope": scope, "query": queries[i % len(queries)]},
        )
        record("recall", ms, code)
    # 3. context bundle, uncached (a unique suffix defeats the bundle cache)
    for i in range(requests):
        ms, code = await timed(
            client,
            "POST",
            "/v1/context",
            hdrs,
            json={"scope": scope, "query": f"{queries[i % len(queries)]} (variant {i})"},
        )
        record("context", ms, code)
    # 4. cached context: the same query twice, the second one is measured
    for i in range(requests):
        body = {"scope": scope, "query": queries[i % len(queries)]}
        await client.post("/v1/context", headers=hdrs, json=body)
        ms, code = await timed(client, "POST", "/v1/context", hdrs, json=body)
        record("cached", ms, code)
    # 5. file accept (distinct bytes each time so nothing is deduplicated)
    for i in range(requests):
        salt = f"\n\n<!-- accept {i} -->\n".encode()
        ms, code = await timed(
            client,
            "POST",
            "/v1/files",
            hdrs,
            files={"file": (f"accept_{i}.md", fixture + salt, "text/markdown")},
            data={"scope": json.dumps(scope), "title": f"accept {i}"},
        )
        record("file", ms, code)
    await settle()
    return lat, statuses


def latency_report(
    lat: dict[str, list[float]],
    statuses: dict[str, dict[int, int]],
    budgets: dict[str, float],
) -> dict[str, Any]:
    """The ``performance.json`` shape: one p95 per budget key, the budgets, the detail
    per series and the verdict."""
    out: dict[str, Any] = {
        "chat_accept_p95_ms": _pct(lat["chat"], 95),
        "cached_context_p95_ms": _pct(lat["cached"], 95),
        "recall_p95_ms": _pct(lat["recall"], 95),
        "context_bundle_p95_ms": _pct(lat["context"], 95),
        "file_accept_p95_ms": _pct(lat["file"], 95),
        "budgets_ms": dict(budgets),
        "detail": {k: {**stats(v), "status_codes": statuses[k]} for k, v in lat.items()},
    }
    out["within_budget"] = all(out[k] <= v for k, v in out["budgets_ms"].items())
    return out


def print_budget_table(payload: dict[str, Any]) -> None:
    for key, budget in payload["budgets_ms"].items():
        flag = "ok" if payload[key] <= budget else "OVER"
        print(f"{key:24} {payload[key]:8.2f} ms  (budget {budget})  {flag}")
