"""A log line that cannot be written is lost, never raised into the code that logged it.

The job worker logged a warning while stdout's disk was full: the ``OSError`` came out of the
log call, and the worker stopped with nothing in its log to say why."""

from __future__ import annotations

import io
import logging

import pytest
import structlog

from memory_service.observability.logging import LossySink, configure_logging, get_logger
from memory_service.observability.metrics import REGISTRY

pytestmark = pytest.mark.unit


class _FullDisk(io.StringIO):
    def flush(self) -> None:
        raise OSError(28, "No space left on device")


def _dropped() -> float:
    return REGISTRY.get_sample_value("memory_log_lines_dropped_total") or 0.0


def test_a_line_that_cannot_be_written_is_counted_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disk = _FullDisk()
    monkeypatch.setattr("sys.stdout", disk)
    configured = structlog.get_config()
    handlers = logging.getLogger().handlers[:]
    configure_logging("INFO", json_output=True)
    before = _dropped()
    try:
        get_logger("test").warning("worker.db_retry", attempt=1)  # raised OSError before
        assert _dropped() == before + 1
        assert "worker.db_retry" in disk.getvalue(), "what could be buffered still is"
    finally:
        structlog.configure(**configured)
        logging.getLogger().handlers[:] = handlers


class _ClosedPipe(io.StringIO):
    """An unbuffered stream: the write itself fails."""

    def write(self, text: str) -> int:
        raise BrokenPipeError(32, "Broken pipe")


def test_a_line_whose_write_fails_is_counted_once() -> None:
    sink = LossySink(_ClosedPipe())
    before = _dropped()
    print("worker.db_retry attempt=1", file=sink, flush=True)  # type: ignore[arg-type]
    print("worker.db_retry attempt=2", file=sink, flush=True)  # type: ignore[arg-type]
    assert _dropped() == before + 2


def test_a_writable_stream_gets_every_line() -> None:
    out = io.StringIO()
    sink = LossySink(out)
    sink.write("line\n")
    sink.flush()
    assert out.getvalue() == "line\n"
