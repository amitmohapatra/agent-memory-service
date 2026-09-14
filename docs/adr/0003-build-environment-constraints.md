# ADR 0003: Build-environment constraints and test fallbacks

**Status:** accepted · **Date:** 2026-09-14

## Context
The environment used to build this repository has network egress limited to PyPI, npm and
Ubuntu apt. Docker Hub, GitHub release downloads, the Go module proxy and HuggingFace are
blocked. Therefore Qdrant, Dragonfly and OpenFGA servers and the Granite/MiniLM model
weights cannot be fetched there, while PostgreSQL 16 and Redis 7 can be installed from apt.

## Decision
Every backing service has two test paths:

| Port | Real integration (Docker/compose, `-m docker`) | Embedded/fake path (always runs) |
|---|---|---|
| PostgreSQL | `postgres:16` container | apt-installed PostgreSQL 16 in CI/sandbox (same engine) |
| Cache | Dragonfly container | Redis 7 (identical protocol) or `fakeredis` |
| Search | Qdrant server | `qdrant-client` local mode (`:memory:`), same client API |
| Authorization | OpenFGA server | in-memory ReBAC adapter implementing the same model semantics; OpenFGA adapter covered by contract tests against recorded HTTP responses |
| Blob | GCS (or SeaweedFS S3 profile) | filesystem adapter with generation + checksum semantics |
| Embeddings / reranker | Granite / MiniLM weights (`-m models`) | deterministic hash embedding + lexical reranker fakes |

The Docker/model paths are wired and must be run on a machine with registry access (the
`docker compose` stack) before any production-readiness claim. Results from the embedded
paths are labeled as such in benchmark artifacts.

## Consequences
- Gates that depend on real models (retrieval quality with Granite) are recorded as
  *not measured* until the `models` suite has run; the release gate treats missing evidence
  as failure.
- No adapter is skipped: all are implemented; only their execution environment differs.
