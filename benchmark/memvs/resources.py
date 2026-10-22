"""CPU seconds and peak resident memory of this process and the processes a system runs in.

Read from ``/proc`` (Linux), so every interpreter the benchmark runs in can measure itself
without a dependency. A system that runs out of process (this service's API, worker and
Qdrant; the shared embedding server the libraries call) is measured by its pids, passed as
``MEMVS_PIDS="api=123,worker=456"``; resident memory is sampled every ``SAMPLE_SECONDS``.
"""

from __future__ import annotations

import os
import resource
import threading
from dataclasses import dataclass, field
from pathlib import Path

ENV_PIDS = "MEMVS_PIDS"
SAMPLE_SECONDS = 0.5
_TICKS = os.sysconf("SC_CLK_TCK")
_PAGE_KB = os.sysconf("SC_PAGE_SIZE") // 1024


def _cpu_seconds(pid: int) -> float:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except OSError:
        return 0.0
    # utime, stime, cutime, cstime: fields 14-17 of the stat line (11-14 after the name)
    return sum(int(x) for x in fields[11:15]) / _TICKS


def _rss_mb(pid: int) -> float:
    try:
        pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
    except (OSError, IndexError, ValueError):
        return 0.0
    return pages * _PAGE_KB / 1024


def external_pids() -> dict[str, int]:
    raw = os.environ.get(ENV_PIDS, "")
    out: dict[str, int] = {}
    for part in filter(None, raw.split(",")):
        name, _, pid = part.partition("=")
        if pid.isdigit():
            out[name] = int(pid)
    return out


@dataclass
class Usage:
    cpu_s: float
    peak_rss_mb: float
    by_process: dict[str, dict[str, float]] = field(default_factory=dict)


class Meter:
    """``with Meter() as m: ...`` then ``m.usage``: CPU spent and the peak resident set."""

    def __init__(self) -> None:
        self._pids = external_pids()
        self._peak: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.usage: Usage | None = None

    def _rss(self) -> dict[str, float]:
        return {"self": _rss_mb(os.getpid())} | {n: _rss_mb(p) for n, p in self._pids.items()}

    def _sample(self) -> None:
        while not self._stop.wait(SAMPLE_SECONDS):
            for name, rss in self._rss().items():
                self._peak[name] = max(self._peak.get(name, 0.0), rss)

    def __enter__(self) -> Meter:
        self._self_cpu0 = _self_cpu()
        self._cpu0 = {n: _cpu_seconds(p) for n, p in self._pids.items()}
        self._peak = self._rss()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        by: dict[str, dict[str, float]] = {
            "self": {
                "cpu_s": _self_cpu() - self._self_cpu0,
                # the kernel's own high-water mark beats any sampling for this process
                "peak_rss_mb": max(self._peak.get("self", 0.0), _self_max_rss_mb()),
            }
        }
        for name, pid in self._pids.items():
            by[name] = {
                "cpu_s": _cpu_seconds(pid) - self._cpu0[name],
                "peak_rss_mb": self._peak.get(name, 0.0),
            }
        self.usage = Usage(
            cpu_s=round(sum(v["cpu_s"] for v in by.values()), 2),
            peak_rss_mb=round(sum(v["peak_rss_mb"] for v in by.values()), 1),
            by_process={k: {m: round(x, 2) for m, x in v.items()} for k, v in by.items()},
        )


def _self_cpu() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + kids.ru_utime + kids.ru_stime


def _self_max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
