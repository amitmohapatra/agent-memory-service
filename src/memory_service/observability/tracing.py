"""OpenTelemetry setup. Spans cover API, auth, authz, DB, cache, retrieval, context,
ingestion stages, archive, graph and eval. Exporter is configuration (none|console|otlp)."""

from __future__ import annotations

import re
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from memory_service.config.constants import HEADERS
from memory_service.config.settings import ObservabilitySettings

_configured = False


def configure_tracing(settings: ObservabilitySettings, service_name: str, version: str) -> None:
    global _configured  # noqa: PLW0603
    if _configured or not settings.otel_enabled:
        return
    resource = Resource.create({"service.name": service_name, "service.version": version})
    provider = TracerProvider(resource=resource)
    if settings.otel_exporter == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif settings.otel_exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint))
        )
    trace.set_tracer_provider(provider)
    _configured = True


def tracer(name: str = "memory_service") -> trace.Tracer:
    return trace.get_tracer(name)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[trace.Span]:
    """Convenience: ``with span("retrieval.dense", tenant_id=...):``."""
    with tracer().start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    if ctx and ctx.trace_id:
        return format(ctx.trace_id, "032x")
    return None


# --- W3C Trace Context -------------------------------------------------------------------
#
# The API speaks ``traceparent`` on the way in and on the way out (ADR 0022), whether or not
# this process exports spans: Datadog (per service) and Langfuse (per agent) join on the
# 32-hex trace id, so one must exist for every request and it must be the same one everywhere.

TRACEPARENT_HEADER = HEADERS.traceparent
#: version-traceid-parentid-flags; a future version may append fields, version 00 may not
_TRACEPARENT = re.compile(r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})(-.*)?$")


@dataclass(frozen=True, slots=True)
class TraceParent:
    trace_id: str
    span_id: str
    flags: str


def parse_traceparent(value: str | None) -> TraceParent | None:
    """A W3C ``traceparent`` (level 1): ``00-<32 hex>-<16 hex>-<2 hex>``, lower case.

    Version ``ff`` and all-zero ids are invalid, as the specification says; an unparseable
    header is ignored rather than refused, so a broken proxy cannot take the API down.
    """
    if not value:
        return None
    match = _TRACEPARENT.fullmatch(value.strip())
    if match is None:
        return None
    version, trace_id, span_id, flags, tail = match.groups()
    if version == "ff" or (version == "00" and tail):
        return None
    if set(trace_id) == {"0"} or set(span_id) == {"0"}:
        return None
    return TraceParent(trace_id, span_id, flags)


def format_traceparent(trace_id: str, span_id: str, flags: str) -> str:
    return f"00-{trace_id}-{span_id}-{flags}"


def new_trace_id() -> str:
    return secrets.token_hex(16)


def new_span_id() -> str:
    return secrets.token_hex(8)


def current_span_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    if ctx and ctx.span_id:
        return format(ctx.span_id, "016x")
    return None


def current_trace_flags() -> str:
    """The active span's flags (``01`` when it is sampled), ``00`` when nothing is tracing."""
    ctx = trace.get_current_span().get_span_context()
    if ctx and ctx.trace_id:
        return format(int(ctx.trace_flags), "02x")
    return "00"
