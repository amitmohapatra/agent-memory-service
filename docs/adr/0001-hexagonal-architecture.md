# ADR 0001: Hexagonal architecture with provider registry

**Status:** accepted · **Date:** 2026-09-14

## Context
The service must run on GCP with Qdrant/Dragonfly/OpenFGA/Procrastinate today and allow
Vertex/OpenAI embeddings, other vector stores, Kafka, Valkey or S3 tomorrow without touching
product semantics. The specification forbids provider SDK imports in the domain.

## Decision
Ports are `typing.Protocol`s in `memory_service.ports`. Adapters live in
`memory_service.adapters` and are attached by `adapters/wiring.py`. Configuration (`Settings`)
selects the provider. (As decided, wiring went through a `ProviderRegistry` keyed by
`(port, provider_name)`; only the document parsers ever registered in it, and since 2026-11
they are a plain name-to-factory table, `adapters/parsers.PARSERS`.) Architecture tests and Ruff `banned-api` enforce the boundary.

## Consequences
- Adding a provider = new adapter module + registry entry; no switch statements.
- Every provider records a `ProviderInfo` (license, origin, locality) so the provider policy
  can allow/deny by configuration.
- Tests can run against in-memory fakes for every port.
