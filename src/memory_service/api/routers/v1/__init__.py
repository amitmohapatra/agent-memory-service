"""Public /v1 routers. Each milestone appends its router to ``routers()``."""

from __future__ import annotations

from fastapi import APIRouter


def routers() -> list[APIRouter]:
    from memory_service.api.routers.v1 import (
        admin,
        agent_tools,
        conversation,
        feedback,
        files,
        graph,
        grounding,
        memory,
        model_keys,
        profile,
        retrieval,
        tenancy,
        tools,
    )

    return [
        admin.router,
        tenancy.router,
        model_keys.router,
        conversation.router,
        files.router,
        retrieval.router,
        grounding.router,
        memory.router,
        profile.router,
        graph.router,
        tools.router,
        agent_tools.router,
        feedback.router,
    ]
