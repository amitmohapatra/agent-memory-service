"""Stored-brief HTTP read timing: real PostgreSQL, in-process ASGI and authorization.

Requires an explicitly supplied, already migrated benchmark database. Creates a unique
tenant without truncating any table. Setup uses hash embeddings; timed reads must not
invoke retrieval or generation. This is not an end-to-end retrieval latency benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import time
from pathlib import Path

import httpx

from benchmark.harness import headers, stats
from memory_service.api.app import create_app
from memory_service.application.container import Overrides
from memory_service.config.settings import Settings
from memory_service.domain.briefs import BriefSpec
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import ObservationKind
from memory_service.domain.ids import new_id
from memory_service.modules.briefs import service as brief_module
from memory_service.modules.jobs.registry import register_handlers


async def run(args: argparse.Namespace) -> dict:
    settings = Settings(
        _env_file=None,
        database={"url": args.database_url},
        service={"environment": "test", "log_level": "WARNING", "rate_limit_per_minute": 0},
        authentication={"mode": "trusted_dev", "trusted_dev_api_keys": ["bench"]},
        models={"llm": {"enabled": False, "uses": []}},
    )
    overrides = Overrides(
        cache="memory",
        search="memory",
        tasks="memory",
        authorization="memory",
        blob="memory",
        embedding="hash",
        nli="lexical",
        document_parser="builtin",
    )
    app = create_app(settings, overrides=overrides)
    ctx = MemoryExecutionContext(
        tenant_id=f"brief-bench-{new_id('request')}", user_id="owner", agent_id="research"
    )
    async with app.router.lifespan_context(app):
        container = app.state.container
        register_handlers(container)
        factory = container.services["uow_factory"]
        service = container.services["briefs"]
        async with factory() as uow:
            await container.services["memory"].submit_observation(
                uow,
                ctx,
                kind=ObservationKind.MESSAGE,
                content="I prefer concise answers with code samples and citations.",
            )
            await uow.commit()
        await container.tasks.drain()
        await container.tasks.drain()
        async with factory() as uow:
            brief = await service.create(
                uow, ctx, BriefSpec(title="Preferences", question="What format do I prefer?")
            )
            await uow.commit()
        await container.tasks.drain()

        # A measured read must work after retrieval and synthesis have been disabled.
        async def forbidden(*args, **kwargs):
            raise AssertionError("Stored brief reads must not retrieve or synthesize")

        service.builder.build = forbidden
        service.assist.structured = forbidden
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://benchmark",
            headers=headers(tenant=ctx.tenant_id, user=ctx.user_id),
        ) as client:
            path = f"/v1/briefs/{brief.brief_id}"
            payload_bytes = 0

            async def read() -> float:
                nonlocal payload_bytes
                start = time.perf_counter()
                response = await client.get(path, params={"agent_id": ctx.agent_id})
                elapsed = (time.perf_counter() - start) * 1000
                response.raise_for_status()
                body = response.json()
                assert body["status"] == "ready" and body["output"]["sources"]
                assert not body["output"]["generated"]
                payload_bytes = len(response.content)
                return elapsed

            for _ in range(20):
                await read()
            sequential = [await read() for _ in range(args.samples)]
            gate = asyncio.Semaphore(args.concurrency)

            async def limited_read() -> float:
                async with gate:
                    return await read()

            concurrent = await asyncio.gather(*(limited_read() for _ in range(args.samples)))
    source = Path(brief_module.__file__)
    return {
        "complete": True,
        "llm_calls": 0,
        "platform": platform.platform(),
        "service_sha256": hashlib.sha256(await asyncio.to_thread(source.read_bytes)).hexdigest(),
        "sequential_ms": stats(sequential),
        "concurrent_ms": stats(concurrent),
        "concurrency": args.concurrency,
        "response_bytes": payload_bytes,
        "limitations": [
            "ASGI transport and in-process cache/authorization; PostgreSQL is real.",
            "One small native brief after warmup, not a size/corpus scalability test.",
            "Concurrent timing starts after the client semaphore, includes server queueing.",
            "Rate limiting disabled; other host work may affect timings.",
            "Not retrieval, network, live OpenFGA or a production p99 measurement.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 100 or not 1 <= args.concurrency <= 32:
        parser.error("Require at least 100 samples and concurrency between 1 and 32")
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
