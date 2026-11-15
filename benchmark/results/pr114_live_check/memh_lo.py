"""memh_lo.py <label> <conversation>: PR #114 live check. One LoCoMo conversation ingested on a
private database with every model use on (the tenant policy names ALL_LLM_USES, reads assisted),
then the person's beliefs consolidated to completion, as lo4_exp.py's M7 arm did. No question is
asked, answered or scored. The model calls and their failures are read from this run's log by
memh_lo_count.py."""

import asyncio
import json
import sys
import time
from pathlib import Path

import benchmark.longmemeval_retrieval as lmr
from benchmark import locomo
from benchmark.public import memory_qa as qa
from sqlalchemy import text as sql

from memory_service.config.settings import ALL_LLM_USES
from memory_service.domain.context import MemoryExecutionContext

label, conv = sys.argv[1], int(sys.argv[2])
OUT = f"/var/tmp/cons/memh-tmp/lo_{label}_{conv}.json"


async def main(item: dict) -> dict:
    settings = lmr._settings()
    lmr.dedicated_database(settings.database.url.get_secret_value())
    base = lmr.bench_overrides()
    container = await lmr.build_container(
        settings, lmr.__version__, overrides=lmr.replace(base, retrieval=lmr.bench_retrieval(base))
    )
    lmr.register_handlers(container)
    ctx = MemoryExecutionContext(
        tenant_id=locomo.TENANT, user_id=f"locomo-{conv}", workspace_id="ws"
    )
    try:
        async with container.database.engine.begin() as c:
            await c.execute(sql(f"TRUNCATE {qa.TABLES} RESTART IDENTITY CASCADE"))
        await lmr.reset_store(container, locomo.TENANT)
        async with container.services["uow_factory"]() as uow:
            await container.services["model_policies"].set(
                uow, locomo.TENANT, uses=list(ALL_LLM_USES), read_assist=True, models={}
            )
            await uow.commit()
        t = time.perf_counter()
        turns = await locomo._ingest_conversation(container, ctx, item["conversation"])
        await container.tasks.drain()
        ingest = time.perf_counter() - t
        from memory_service.tools.consolidate import consolidate_tenant

        t = time.perf_counter()
        report = await consolidate_tenant(container, locomo.TENANT)
        await container.tasks.drain()
        done = {
            "label": label,
            "conv": conv,
            "turns": len(turns),
            "ingest_s": round(ingest),
            "consolidate_s": round(time.perf_counter() - t),
            "consolidation": report,
        }
        print(label, conv, "turns", len(turns), "ingest", round(ingest), flush=True)
    finally:
        await container.close()
    return done


conversation = json.loads(Path("/var/tmp/mx/ams/benchmark/data/locomo10.json").read_text())[conv]
Path(OUT).write_text(json.dumps(asyncio.run(main(conversation)), default=str))
print("all done", flush=True)
