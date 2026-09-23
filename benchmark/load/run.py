"""Headless Locust run against a deployed instance, reduced to ``load_test.json``.

    uv run python -m benchmark.load.run --base-url http://localhost:8080 -u 10 -r 10 -t 300s

Runs ``benchmark/load/locustfile.py`` with ``--csv`` and turns Locust's per-endpoint
statistics (requests, failures, RPS, p50/p95/p99) and failure table into one artifact with
provenance. The Locust exit status is recorded, not interpreted: the artifact is evidence
for the final report, the p95 gates are ``performance_network.json``.

The offered rate is the user count (the locustfile paces each user at one request per
second), so ``-u 10`` aims at ten requests per second. ``--arm cold`` (the default) salts
every query so the bundle cache always misses; ``--arm warm`` repeats a small set.

While the run is in flight the container's own CPU and memory are sampled once a second
(``docker stats``) and summarised into the artifact, because a throughput number without
the CPU it cost cannot be extrapolated to another machine: what travels between boxes is
core-seconds per request, not the rate.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from benchmark.common import provenance, write_result

LOCUSTFILE = Path(__file__).with_name("locustfile.py")
AGGREGATED = "Aggregated"


def _num(value: str | None) -> float:
    try:
        return round(float(value or 0), 2)
    except ValueError:
        return 0.0


def parse_stats(csv_text: str) -> dict[str, Any]:
    """``<prefix>_stats.csv`` -> ``{"endpoints": {name: {...}}, "aggregated": {...}}``."""
    endpoints: dict[str, Any] = {}
    aggregated: dict[str, Any] = {}
    for row in csv.DictReader(io.StringIO(csv_text)):
        name = row.get("Name") or ""
        entry = {
            "method": row.get("Type") or "",
            "requests": int(_num(row.get("Request Count"))),
            "failures": int(_num(row.get("Failure Count"))),
            "rps": _num(row.get("Requests/s")),
            "failures_per_s": _num(row.get("Failures/s")),
            "p50_ms": _num(row.get("50%")),
            "p95_ms": _num(row.get("95%")),
            "p99_ms": _num(row.get("99%")),
            "avg_ms": _num(row.get("Average Response Time")),
            "min_ms": _num(row.get("Min Response Time")),
            "max_ms": _num(row.get("Max Response Time")),
        }
        if name == AGGREGATED:
            aggregated = entry
        else:
            endpoints[name] = entry
    return {"endpoints": endpoints, "aggregated": aggregated}


def parse_failures(csv_text: str) -> list[dict[str, Any]]:
    """``<prefix>_failures.csv`` -> one entry per (method, name, error)."""
    out: list[dict[str, Any]] = []
    for row in csv.DictReader(io.StringIO(csv_text)):
        out.append(
            {
                "method": row.get("Method") or "",
                "name": row.get("Name") or "",
                "error": (row.get("Error") or "")[:300],
                "occurrences": int(_num(row.get("Occurrences"))),
            }
        )
    return out


def load_report(
    stats_csv: str,
    failures_csv: str,
    *,
    base_url: str,
    users: int,
    spawn_rate: float,
    run_time: str,
    exit_status: int,
) -> dict[str, Any]:
    stats = parse_stats(stats_csv)
    failures = parse_failures(failures_csv)
    aggregated = stats["aggregated"]
    return {
        "transport": "tcp",
        "base_url": base_url,
        "users": users,
        "spawn_rate": spawn_rate,
        "run_time": run_time,
        "locustfile": LOCUSTFILE.name,
        "endpoints": stats["endpoints"],
        "aggregated": aggregated,
        "failures": failures,
        "total_requests": aggregated.get("requests", 0),
        "total_failures": aggregated.get("failures", 0),
        "failure_ratio": round(aggregated.get("failures", 0) / aggregated["requests"], 4)
        if aggregated.get("requests")
        else None,
        "locust_exit_status": exit_status,
    }


class _ResourceSampler:
    """``docker stats`` for the named containers, once a second, in a background thread.

    Sampling is best-effort: a box without a docker socket, or a container that restarts
    mid-run, must not fail a load test. What it cannot sample it reports as absent rather
    than as zero.
    """

    def __init__(self, containers: list[str], interval: float = 1.0) -> None:
        self.containers = containers
        self.interval = interval
        self.samples: dict[str, list[tuple[float, float]]] = {c: [] for c in containers}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _ResourceSampler:
        if self.containers:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(  # noqa: S603 - fixed argv
                    [
                        "docker",
                        "stats",
                        "--no-stream",
                        "--format",
                        "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}",
                        *self.containers,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except Exception:  # noqa: BLE001 - sampling must never fail the run
                self._stop.wait(self.interval)
                continue
            for line in out.stdout.splitlines():
                parts = line.split("\t")
                if len(parts) != 3 or parts[0] not in self.samples:
                    continue
                cpu = _num(parts[1].rstrip("%"))
                mem = re.match(r"([0-9.]+)\s*([KMGT]?i?B)", parts[2].strip())
                mib = 0.0
                if mem:
                    scale = {"B": 1 / 2**20, "KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0}
                    mib = float(mem.group(1)) * scale.get(mem.group(2), 1.0)
                self.samples[parts[0]].append((cpu, mib))
            self._stop.wait(self.interval)

    def report(self, *, requests: int, seconds: float) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, rows in self.samples.items():
            if not rows:
                out[name] = {"samples": 0}
                continue
            cpu = sorted(r[0] for r in rows)
            mem = [r[1] for r in rows]
            mean_cpu = sum(cpu) / len(cpu)
            entry = {
                "samples": len(rows),
                "cpu_percent_mean": round(mean_cpu, 1),
                "cpu_percent_p95": round(cpu[int(len(cpu) * 0.95) - 1], 1),
                "rss_mib_mean": round(sum(mem) / len(mem), 1),
                "rss_mib_max": round(max(mem), 1),
                "rss_drift_percent": round(100 * (mem[-1] - mem[0]) / max(mem[0], 1), 1),
            }
            if requests:
                # The portable number: one request's cost in core-seconds. Multiply by a
                # target rate to size any other box (cores >= rate x core_seconds).
                entry["core_seconds_per_request"] = round(mean_cpu / 100 * seconds / requests, 4)
            out[name] = entry
        return out


def _run_seconds(run_time: str) -> float:
    match = re.fullmatch(r"(\d+)([smh]?)", run_time.strip())
    if not match:
        return 0.0
    return int(match.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[match.group(2)]


def run_locust(
    base_url: str,
    users: int,
    spawn_rate: float,
    run_time: str,
    api_key: str,
    csv_prefix: Path,
    arm: str = "cold",
    docs: str = "text",
) -> int:
    argv = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(LOCUSTFILE),
        "--headless",
        "-u",
        str(users),
        "-r",
        str(spawn_rate),
        "-t",
        run_time,
        "--host",
        base_url,
        "--csv",
        str(csv_prefix),
        "--only-summary",
    ]
    env = {
        **os.environ,
        "MEMORY_API_KEY": api_key,
        "MEMORY_LOAD_ARM": arm,
        "MEMORY_LOAD_DOCS": docs,
    }
    return subprocess.call(argv, env=env)  # noqa: S603 - fixed argv


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default=os.environ.get("MEMORY_API_KEY", "dev-key"))
    parser.add_argument("-u", "--users", type=int, default=20)
    parser.add_argument("-r", "--spawn-rate", type=float, default=5)
    parser.add_argument("-t", "--run-time", default="60s")
    parser.add_argument(
        "--arm",
        choices=("cold", "warm"),
        default="cold",
        help="cold salts every query so the bundle cache misses; warm repeats a small set",
    )
    parser.add_argument(
        "--docs",
        choices=("text", "pdf"),
        default="text",
        help=(
            "what the upload task sends. text is markdown, handled by the builtin parser; "
            "pdf is a real 39 KB release and is the only way to reach docling, the most "
            "expensive model the service loads. Defaults to text, which is what every "
            "capacity number before this one measured - so those numbers describe traffic "
            "with no documents in it"
        ),
    )
    parser.add_argument(
        "--sample",
        default="",
        help="comma-separated container names to sample CPU and memory from during the run",
    )
    parser.add_argument("--out", default="load_test.json", help="artifact filename")
    args = parser.parse_args()
    containers = [c for c in args.sample.split(",") if c.strip()]
    with tempfile.TemporaryDirectory(prefix="memory-load-") as tmp:
        prefix = Path(tmp) / "load"
        started = time.perf_counter()
        with _ResourceSampler(containers) as sampler:
            status = run_locust(
                args.base_url,
                args.users,
                args.spawn_rate,
                args.run_time,
                args.api_key,
                prefix,
                args.arm,
                args.docs,
            )
        elapsed = time.perf_counter() - started
        stats_path = prefix.with_name("load_stats.csv")
        failures_path = prefix.with_name("load_failures.csv")
        stats_csv = stats_path.read_text(encoding="utf-8") if stats_path.is_file() else ""
        failures_csv = failures_path.read_text(encoding="utf-8") if failures_path.is_file() else ""
    payload = load_report(
        stats_csv,
        failures_csv,
        base_url=args.base_url,
        users=args.users,
        spawn_rate=args.spawn_rate,
        run_time=args.run_time,
        exit_status=status,
    )
    payload["arm"] = args.arm
    payload["docs"] = args.docs
    payload["target_rps"] = args.users
    payload["resources"] = sampler.report(
        requests=payload["total_requests"],
        seconds=min(elapsed, _run_seconds(args.run_time) or elapsed),
    )
    payload["provenance"] = provenance()
    path = write_result(args.out, payload)
    print(f"wrote {path}")
    agg = payload["aggregated"]
    print(
        f"requests={payload['total_requests']} failures={payload['total_failures']} "
        f"rps={agg.get('rps')} p50={agg.get('p50_ms')}ms p95={agg.get('p95_ms')}ms "
        f"p99={agg.get('p99_ms')}ms"
    )
    for name, e in payload["endpoints"].items():
        print(f"  {name:32} n={e['requests']:6} fail={e['failures']:4} p95={e['p95_ms']:8.1f}ms")
    for name, r in payload["resources"].items():
        if r.get("samples"):
            print(
                f"  {name:32} cpu_mean={r['cpu_percent_mean']}% rss_max={r['rss_mib_max']}MiB "
                f"core_s/req={r.get('core_seconds_per_request')}"
            )
    return 0 if stats_csv else 1


if __name__ == "__main__":
    raise SystemExit(main())
