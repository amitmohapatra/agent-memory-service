"""Headless Locust run against a deployed instance, reduced to ``load_test.json``.

    uv run python -m benchmark.load.run --base-url http://memory-api:8080 -u 20 -r 5 -t 60s

Runs ``benchmark/load/locustfile.py`` with ``--csv`` and turns Locust's per-endpoint
statistics (requests, failures, RPS, p50/p95/p99) and failure table into one artifact with
provenance. The Locust exit status is recorded, not interpreted: the artifact is evidence
for the final report, the p95 gates are ``performance_network.json``.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
import tempfile
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


def run_locust(
    base_url: str, users: int, spawn_rate: float, run_time: str, api_key: str, csv_prefix: Path
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
    env = {**os.environ, "MEMORY_API_KEY": api_key}
    return subprocess.call(argv, env=env)  # noqa: S603 - fixed argv


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default=os.environ.get("MEMORY_API_KEY", "dev-key"))
    parser.add_argument("-u", "--users", type=int, default=20)
    parser.add_argument("-r", "--spawn-rate", type=float, default=5)
    parser.add_argument("-t", "--run-time", default="60s")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="memory-load-") as tmp:
        prefix = Path(tmp) / "load"
        status = run_locust(
            args.base_url, args.users, args.spawn_rate, args.run_time, args.api_key, prefix
        )
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
    payload["provenance"] = provenance()
    path = write_result("load_test.json", payload)
    print(f"wrote {path}")
    agg = payload["aggregated"]
    print(
        f"requests={payload['total_requests']} failures={payload['total_failures']} "
        f"rps={agg.get('rps')} p50={agg.get('p50_ms')}ms p95={agg.get('p95_ms')}ms "
        f"p99={agg.get('p99_ms')}ms"
    )
    for name, e in payload["endpoints"].items():
        print(f"  {name:32} n={e['requests']:6} fail={e['failures']:4} p95={e['p95_ms']:8.1f}ms")
    return 0 if stats_csv else 1


if __name__ == "__main__":
    raise SystemExit(main())
