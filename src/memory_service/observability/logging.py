"""Structured JSON logging with request-scoped context (tenant/thread/turn/trace/...)."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog

_log_context: ContextVar[dict[str, str] | None] = ContextVar("memory_log_context", default=None)


def bind_log_context(**fields: str) -> None:
    current = dict(_log_context.get() or {})
    current.update({k: v for k, v in fields.items() if v})
    _log_context.set(current)


def clear_log_context() -> None:
    _log_context.set(None)


def _inject_context(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    for key, value in (_log_context.get() or {}).items():
        event_dict.setdefault(key, value)
    return event_dict


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _inject_context,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    if json_output:
        shared.append(structlog.processors.format_exc_info)
    renderer: Any = (
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=False,
    )
    logging.basicConfig(level=level.upper(), stream=sys.stdout, format="%(message)s")
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
