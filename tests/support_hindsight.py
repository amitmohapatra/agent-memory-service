"""Real SDK against a local HTTP fixture; never contacts a model or memory server."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from aiohttp import web

from memory_service.adapters.models.hindsight import HindsightExtractor
from memory_service.config.settings import HindsightSettings


@dataclass
class PreviewServer:
    extractor: HindsightExtractor
    requests: list[dict] = field(default_factory=list)


@asynccontextmanager
async def preview_server(*, facts: list[dict], status: int = 200) -> AsyncIterator[PreviewServer]:
    requests: list[dict] = []

    async def preview(request: web.Request) -> web.Response:
        body = await request.json()
        requests.append({"path": request.path, "body": body})
        return web.json_response(
            {
                "facts": facts,
                "chunks": [{"text": body["content"], "fact_count": len(facts)}],
                "usage": {"input_tokens": 30, "output_tokens": 10, "thoughts_tokens": 2},
            },
            status=status,
        )

    app = web.Application()
    app.router.add_post("/v1/default/banks/{bank}/memories/dry-run-extract", preview)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        extractor = HindsightExtractor(HindsightSettings(base_url=f"http://127.0.0.1:{port}"))
        try:
            yield PreviewServer(extractor, requests)
        finally:
            await extractor.close()
    finally:
        await runner.cleanup()
