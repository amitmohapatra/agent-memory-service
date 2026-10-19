"""Structured JSON logging with request-scoped context (tenant/thread/turn/trace/...)."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any, TextIO

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


class LossySink:
    """Where the structured logs are written (stdout), losing a line rather than raising
    when it cannot be written: a log call raised ``OSError`` into whatever code made it when
    stdout's disk was full, and a worker whose job or heartbeat logged a warning then stopped
    with nothing in its log to say why. A lost line is counted
    (``memory_log_lines_dropped_total``)."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self._lost = False

    def write(self, text: str) -> int:
        try:
            return self.stream.write(text)
        except OSError:  # an unbuffered stream fails here, a buffered one on flush
            self._lost = True
            return 0

    def flush(self) -> None:
        """Ends each line (structlog prints with ``flush=True``): the line is counted once
        as dropped when any of its writes, or the flush, failed."""
        try:
            self.stream.flush()
        except OSError:
            self._lost = True
        if self._lost:
            self._lost = False
            from memory_service.observability.metrics import log_lines_dropped_total

            log_lines_dropped_total.inc()


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
        logger_factory=structlog.PrintLoggerFactory(LossySink(sys.stdout)),  # type: ignore[arg-type]
        cache_logger_on_first_use=False,
    )
    logging.basicConfig(level=level.upper(), stream=sys.stdout, format="%(message)s")
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
