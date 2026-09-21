"""One real turn through a running Memory Service, end to end.

Not a unit test: it speaks HTTP to whatever is listening, so it catches the things the suite
cannot — a container that starts but cannot reach Postgres, a migration that did not run, an
authorization store with no model loaded, a worker that is not draining the queue.

    BASE_URL=http://localhost:8080 API_KEY=dev-key python scripts/smoke.py
"""

from __future__ import annotations

import os
import sys
import time
import uuid

import httpx

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080")
API_KEY = os.environ.get("API_KEY", "dev-key")
TENANT = os.environ.get("SMOKE_TENANT", "smoke")
#: how long the asynchronous path (extract → consolidate → index) is given to land
CONSOLIDATION_BUDGET_SECONDS = float(os.environ.get("SMOKE_BUDGET_SECONDS", "60"))
POLL_SECONDS = 2.0


def main() -> int:
    run = uuid.uuid4().hex[:8]
    scope = {
        "tenant_id": TENANT,
        "user_id": f"smoke-{run}",
        "thread_id": f"smoke-thread-{run}",
        "session_id": f"smoke-session-{run}",
        "turn_id": f"smoke-turn-{run}",
        "agent_id": "smoke-agent",
    }
    fact = f"The smoke marker for run {run} is EGRET-{run.upper()}."
    headers = {"X-API-Key": API_KEY, "X-Memory-Tenant": TENANT, "X-Memory-User": scope["user_id"]}

    with httpx.Client(base_url=BASE_URL, headers=headers, timeout=60) as http:
        for probe in ("/health/live", "/health/ready"):
            response = http.get(probe)
            if response.status_code != 200:
                return _fail(f"{probe} returned {response.status_code}")

        written = http.post(
            "/v1/observations", json={"scope": scope, "kind": "MESSAGE", "content": fact}
        )
        if written.status_code >= 300:
            return _fail(
                f"POST /v1/observations returned {written.status_code}: {written.text[:200]}"
            )

        # the write is asynchronous by design, so poll rather than assume
        deadline = time.monotonic() + CONSOLIDATION_BUDGET_SECONDS
        while time.monotonic() < deadline:
            found = http.post("/v1/context", json={"query": f"smoke marker {run}", "scope": scope})
            if found.status_code >= 300:
                return _fail(f"POST /v1/context returned {found.status_code}: {found.text[:200]}")
            if run.upper() in found.text or run in found.text:
                elapsed = CONSOLIDATION_BUDGET_SECONDS - (deadline - time.monotonic())
                sys.stdout.write(f"wrote and read back a memory in {elapsed:.1f}s ({BASE_URL})\n")
                return 0
            time.sleep(POLL_SECONDS)

    return _fail(
        f"the observation never became retrievable within {CONSOLIDATION_BUDGET_SECONDS:.0f}s — "
        "the API answered, so look at the worker and the queue"
    )


def _fail(message: str) -> int:
    sys.stderr.write(f"smoke failed: {message}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
