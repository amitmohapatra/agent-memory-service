"""How a benchmark is wired: one place, read from ``BENCH_*`` variables.

The service's own ``MEMORY__*`` surface is topology and credentials only. What a benchmark
adds on top - whether Qdrant is the real server or the in-process local mode, which stand-ins
replace the stores that a harness never wants to talk to - is decided here and handed to
``build_container(overrides=...)``, never smuggled through settings the product does not have.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Literal

from memory_service.application.container import Overrides


def _flag(name: str, default: str) -> str:
    return (os.environ.get(name) or default).strip().lower()


@dataclass(frozen=True)
class BenchEnv:
    #: ``qdrant``: the real server at ``MEMORY__SEARCH__QDRANT_URL``. ``memory``: qdrant-client
    #: local mode, an exact brute-force scan with no HNSW index - fine for a fixture-sized
    #: corpus and quietly O(n) beyond it (measured on SciFact: nDCG@10 0.012 against a
    #: published ~0.65, and it was the backend, not the retrieval).
    search: Literal["qdrant", "memory"] = "memory"

    @classmethod
    def from_environ(cls) -> BenchEnv:
        search = _flag("BENCH_SEARCH", cls.search)
        if search not in ("qdrant", "memory"):
            raise SystemExit(f"BENCH_SEARCH={search!r}: expected qdrant or memory")
        return cls(search=search)  # type: ignore[arg-type]

    def overrides(self, **changes: Any) -> Overrides:
        """The stand-ins every harness runs with: an in-process cache and queue (a benchmark
        drains its own jobs synchronously, which only the recording queue supports), plus
        local-mode Qdrant unless ``BENCH_SEARCH=qdrant``."""
        base = Overrides(
            cache="memory",
            tasks="memory",
            search="memory" if self.search == "memory" else None,
        )
        return replace(base, **changes) if changes else base


BENCH = BenchEnv.from_environ()


def bench_overrides(**changes: Any) -> Overrides:
    return BENCH.overrides(**changes)
