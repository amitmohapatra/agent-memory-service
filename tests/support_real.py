"""Isolation for real-component runs (``MEMORY_TEST_PROVIDERS=env``): a Qdrant server and a
Dragonfly keep state across tests, unlike the in-process stand-ins, so fixtures reset them
the way they truncate PostgreSQL."""

from __future__ import annotations

from memory_service.application.container import Container
from memory_service.modules.rag.indexer import KNOWLEDGE, MEMORIES


async def reset_real_backends(container: Container) -> None:
    search_cfg = container.settings.search
    if search_cfg.provider == "qdrant" and search_cfg.qdrant_local_path is None:
        indexer = container.services["indexer"]
        for base in (KNOWLEDGE, MEMORIES):
            await container.search.drop_collection(indexer.collection(base))
        await indexer.ensure_collections()
    if container.cache is not None and container.settings.cache.provider != "memory":
        keys = [key async for key in container.cache.scan("*")]
        if keys:
            await container.cache.delete(*keys)
