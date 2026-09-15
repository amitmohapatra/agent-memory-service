"""Eval fixtures reuse the integration container (real PostgreSQL + Qdrant local mode)."""

from __future__ import annotations

from tests.integration.conftest import container, uow_factory  # noqa: F401
