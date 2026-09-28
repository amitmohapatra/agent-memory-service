"""Public /v1 routers. Each milestone appends its router to ``routers()``."""

from __future__ import annotations

from fastapi import APIRouter


def routers() -> list[APIRouter]:
    from memory_service.api.routers.v1 import (
        admin,
        agent_credentials,
        briefs,
        conversation,
        files,
        graph,
        grounding,
        memory,
        retrieval,
        tenancy,
        tools,
    )

    return [
        admin.router,
        tenancy.router,
        agent_credentials.router,
        briefs.router,
        conversation.router,
        files.router,
        retrieval.router,
        grounding.router,
        memory.router,
        graph.router,
        tools.router,
    ]
