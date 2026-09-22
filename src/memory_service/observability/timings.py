"""Per-request stage timings, carried in a result's diagnostics.

``stage_seconds`` (metrics.py) is the process-wide histogram; it cannot say what *this*
query spent where, and a benchmark that wants a p99 with a stage split needs exactly that.
The engine and the context builder each time their stages into one dict that travels with
the bundle, so a result file can show "encode 120 ms, search 35 ms, graph 90 ms" per row
instead of a single number computed once and never checkable.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class Timings:
    __slots__ = ("ms",)

    def __init__(self, ms: dict[str, float] | None = None) -> None:
        self.ms: dict[str, float] = ms if ms is not None else {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = round(
                self.ms.get(name, 0.0) + (time.perf_counter() - started) * 1000, 1
            )
