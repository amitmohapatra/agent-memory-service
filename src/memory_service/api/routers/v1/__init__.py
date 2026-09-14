"""Public /v1 routers. Each milestone appends its router to ``routers()``."""

from __future__ import annotations

from fastapi import APIRouter


def routers() -> list[APIRouter]:
    out: list[APIRouter] = []
    return out
