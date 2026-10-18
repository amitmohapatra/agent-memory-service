"""Leave the service as the run found it: forget what a run wrote, through the service's own
API, when the run ends.

A full ``nollm`` run writes about 70,000 memories into the eval tenant, mostly from the scale
corpus. Repeated gate runs that leave their data behind fill the live stack's disk, as one
afternoon of repeats did. So every mode purges its own writes at the end, unless ``--keep``
or ``--reuse`` is given (a run you mean to re-ask later with ``--reuse`` is run with
``--keep``).

The purge uses the supported paths only, as a tenant administrator would:

* every memory of each thread the run wrote (listed with the thread bound, superseded ones
  included) and each user-level memory is forgotten (``DELETE /v1/memories/{id}``: a soft
  delete that removes the vector index points);
* every thread the run wrote is deleted (``DELETE /v1/threads/{id}``: a soft delete that
  removes the thread's searchable episode).

The service has no tenant purge: soft-deleted rows stay in PostgreSQL, and the graph's
entities and relations built from the writes stay too. The index points are what this
returns (Qdrant reclaims the space as it vacuums segments). Both gaps are recorded in the
capability eval docs.

    python -m benchmark.capabilities.cleanup <raw.json.gz>...   # purge a kept run's writes
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

#: parallel forget requests; small, so the purge never competes with a measurement
CONCURRENCY = 16


@dataclass
class Purged:
    memories: int = 0
    threads: int = 0
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "memories_forgotten": self.memories,
            "threads_deleted": self.threads,
            "errors": self.errors[:20],
            "error_count": len(self.errors),
            "seconds": round(self.seconds, 1),
        }


async def purge(
    client: Any, owned: Iterable[tuple[str, Sequence[str]]], *, concurrency: int = CONCURRENCY
) -> Purged:
    """Forget every memory of each ``(user, threads)`` and delete the threads. Idempotent: a
    second purge finds nothing left to forget."""
    out = Purged()
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()

    async def forget_all(ctx: Any) -> None:
        ids = [m.memory_id async for m in ctx.advanced.memories.iter(include_superseded=True)]

        async def one(memory_id: str) -> None:
            async with sem:
                try:
                    await ctx.forget(memory_id)
                    out.memories += 1
                except Exception as exc:  # noqa: BLE001 - recorded; the rest still go
                    out.errors.append(f"forget {memory_id}: {type(exc).__name__}: {exc}"[:200])

        await asyncio.gather(*(one(i) for i in ids))

    async def user(user_id: str, threads: Sequence[str]) -> None:
        for thread in dict.fromkeys(threads):
            ctx = client.bind(user_id=user_id, thread_id=thread)
            try:
                await forget_all(ctx)
                await ctx.history.delete()
                out.threads += 1
            except Exception as exc:  # noqa: BLE001 - a thread never written is not an error
                if "404" not in str(exc) and "not found" not in str(exc).lower():
                    out.errors.append(f"thread {thread}: {type(exc).__name__}: {exc}"[:200])
        try:
            await forget_all(client.bind(user_id=user_id))
        except Exception as exc:  # noqa: BLE001
            out.errors.append(f"user {user_id}: {type(exc).__name__}: {exc}"[:200])

    users = list(owned)
    for i in range(0, len(users), concurrency):
        await asyncio.gather(*(user(u, ts) for u, ts in users[i : i + concurrency]))
    out.seconds = time.perf_counter() - t0
    return out


def should_purge(settings: Any) -> bool:
    """Purge unless the run is kept for a later ``--reuse``, or is itself one."""
    return not (getattr(settings, "keep", False) or getattr(settings, "reuse", None))


async def _purge_kept(paths: Sequence[str]) -> int:
    """Purge the writes of runs that were kept: their raw records name the tenant, the run and
    the users and threads written (``meta.written``)."""
    import gzip
    import json
    import os

    from trellis.memory import MemoryClient

    url = os.environ.get("MEMORY_URL", "http://localhost:8091")
    admin_key = os.environ.get("CAPEVAL_ADMIN_KEY") or os.environ.get("TRELLIS_ADMIN_KEY")
    if not admin_key:
        print("CAPEVAL_ADMIN_KEY is needed", file=sys.stderr)
        return 2
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            meta = json.load(f)["meta"]
        owned = meta.get("written") or {}
        if not owned:
            print(f"{path}: no users and threads recorded (a run before cleanup existed)")
            continue
        async with MemoryClient(url, api_key=admin_key) as admin:
            key = await admin.administer(meta["tenant"]).keys.issue(
                "service", f"capeval-purge-{meta['run_id']}"
            )
        try:
            async with MemoryClient(url, api_key=key.token or "", timeout=120) as client:
                done = await purge(client, owned.items())
        finally:
            async with MemoryClient(url, api_key=admin_key) as admin:
                await admin.administer(meta["tenant"]).keys.revoke(key.key_id)
        print(f"{path}: {done.as_dict()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    paths = list(sys.argv[1:] if argv is None else argv)
    if not paths:
        print(__doc__)
        return 2
    return asyncio.run(_purge_kept(paths))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
