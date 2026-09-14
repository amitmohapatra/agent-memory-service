"""OpenTelemetry setup. Spans cover API, auth, authz, DB, cache, retrieval, rerank, context,
ingestion stages, archive, graph and eval. Exporter is configuration (none|console|otlp)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

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
