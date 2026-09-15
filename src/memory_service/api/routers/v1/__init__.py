"""Public /v1 routers. Each milestone appends its router to ``routers()``."""

from __future__ import annotations

from fastapi import APIRouter


def routers() -> list[APIRouter]:
    from memory_service.api.routers.v1 import conversation, files, graph, memory, retrieval

    return [conversation.router, files.router, retrieval.router, memory.router, graph.router]
