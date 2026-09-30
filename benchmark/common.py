"""Benchmark provenance. No benchmark result is stored without it."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parent / "results"

#: Databases a harness may truncate. A benchmark owns its database outright - it resets the
#: whole schema between runs - so the name is the only thing standing between a measurement
#: and somebody else's data. The benchmark databases, the phase-scoped ones, and the test
#: suite's own database (``tests/conftest.DB_URL``, which the suite already migrates and
#: truncates) qualify. ``memory`` and ``harness_live`` are deployed stores on the same server
#: and are deliberately absent: ``memory`` is what an unset ``MEMORY__DATABASE__URL`` selects,
#: and what the checked-in ``.env`` names, so it is the one an accident lands on.
DEDICATED_DATABASES: tuple[str, ...] = (
    "memory_bench",
    "memory_hi_",
    "p7_",
    "p9_",
    "p9b_",
    "memory_tests",
)
#: The isolated Qdrant. The shared dev store on 6333 serves a running API; a benchmark that
#: ingested a corpus into it would both pollute those reads and be measured against them.
ISOLATED_QDRANT_PORT = 16333


def dedicated_database(url: str) -> str:
    """The database name in ``url``, or a refusal if a harness may not reset it.

    Enforced here rather than in each harness: ``reset_store`` below TRUNCATEs 25 tables,
    and the benchmark default URL names the shared ``memory`` database, so one unset
    environment variable is the difference between a clean run and deleting a deployment.
    One guard where the damage happens covers every caller, present and future.
    """
    from sqlalchemy.engine import make_url

    database = make_url(url).database or ""
    if not database.startswith(DEDICATED_DATABASES):
        raise ValueError(
            f"{database!r} is not a benchmark database: a harness only resets one named "
            f"{DEDICATED_DATABASES}. Set MEMORY__DATABASE__URL to a dedicated database."
        )
    return database


def isolated_qdrant(url: str) -> str:
    """``url`` if it is the isolated Qdrant, else a refusal."""
    from sqlalchemy.engine import make_url

    if make_url(url.replace("http", "postgresql", 1)).port != ISOLATED_QDRANT_PORT:
        raise ValueError(
            f"{url} is not the isolated Qdrant on {ISOLATED_QDRANT_PORT}: a benchmark never "
            "writes to the shared vector store."
        )
    return url


def file_sha256(path: Path) -> str:
    """Stream artifact identity without loading a model-sized file into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _git_commit() -> str:
    """The commit a result was produced from. Inside the benchmark image the checkout is a
    bind mount with no usable .git, so the Makefile passes GIT_COMMIT; a host run asks git."""
    if commit := os.environ.get("GIT_COMMIT", "").strip():
        return commit
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def local_model_runtime() -> dict[str, Any]:
    """Record installed CPU inference versions without importing heavy runtimes."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": {
            name: _pkg(name)
            for name in ("onnxruntime", "tokenizers", "numpy", "torch", "transformers")
        },
    }


def _model_manifest() -> dict[str, Any]:
    """Model name + revision hash per weight directory (written when the weights are
    downloaded into ``./models``, or ``BENCH_MODELS_DIR``)."""
    root = Path(os.environ.get("BENCH_MODELS_DIR", "models"))
    path = root / "MANIFEST.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: {"repo": v.get("repo"), "revision": v.get("revision")} for k, v in data.items()}


def _llm_provenance() -> dict[str, Any]:
    """Gateway + model names only; never a key."""
    from memory_service.config.settings import Settings

    llm = Settings().models.llm
    out: dict[str, Any] = {
        "enabled": llm.enabled,
        "provider": "bifrost" if llm.enabled else "disabled",
    }
    if llm.enabled:
        out.update({"model": llm.model, "fast_model": llm.fast_model, "uses": list(llm.uses)})
        try:
            import httpx

            resp = httpx.get(f"{llm.base_url.rstrip('/').removesuffix('/v1')}/version", timeout=2)
            if resp.status_code < 400:
                out["gateway_version"] = resp.text.strip()[:80]
        except Exception:  # noqa: BLE001 - provenance only
            pass
    return out


def provenance(**extra: Any) -> dict[str, Any]:
    cpu_count = os.cpu_count() or 0
    mem_gb = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    mem_gb = round(int(line.split()[1]) / 1024 / 1024, 1)
                    break
    except OSError:
        pass
    # Host load AT RUN TIME. The single largest latency contaminant this project has hit:
    # one p99 was inflated 5.3x by a load average of 7.46 on four cores, another run reached
    # 119, and three separate measurements were invalidated by it. None of that was
    # recoverable afterwards, because the load was recorded nowhere - a slower box and a real
    # regression left identical artifacts. It costs one syscall.
    try:
        load1, load5, load15 = os.getloadavg()
    except (OSError, AttributeError):
        load1 = load5 = load15 = -1.0
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "host_load": {
            "loadavg_1m": round(load1, 2),
            "loadavg_5m": round(load5, 2),
            "loadavg_15m": round(load15, 2),
            #: the number that matters - above ~1.0 the box is oversubscribed and every
            #: latency percentile in this file should be read as an upper bound
            "load_per_core": round(load1 / cpu_count, 2) if cpu_count else None,
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": cpu_count,
        "memory_gb": mem_gb,
        "packages": {
            p: _pkg(p)
            for p in (
                "fastapi",
                "sqlalchemy",
                "qdrant-client",
                "procrastinate",
                "zstandard",
                "sentence-transformers",
                "torch",
                "transformers",
                "onnxruntime",
                "docling",
            )
        },
        "models": _model_manifest(),
        "llm": _llm_provenance(),
        **extra,
    }


def write_result(name: str, payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return path


async def reset_store(container: object, tenant_id: str) -> dict[str, int]:
    """Clear everything a benchmark wrote, in *both* stores.

    Every harness here began with ``TRUNCATE`` over the SQL tables and stopped there — and the
    vector store is a separate server that a SQL truncation does not touch. So each run left
    its vectors behind and the next run competed against them. Measured: a 600-document
    SciFact run found 4,288 points under its tenant where ~780 belonged to it, and orphaned
    vectors from earlier runs took top-ten slots that could never map back to the corpus. The
    same benchmark scored nDCG@10 = 0.766 on a clean store and 0.358 on a dirty one, with no
    code change between them.

    Returns what was removed, so a caller can log it rather than assume it worked.

    Refuses outright unless the engine points at a dedicated benchmark database
    (``dedicated_database``). The check is here, at the TRUNCATE, because that is the only
    place every harness passes through: the source harness carried its own guard and the
    judged harness did not, while the benchmark default URL names the shared ``memory``
    database that a deployed API is reading.
    """
    from sqlalchemy import text

    from benchmark.retrieval import TABLES
    from memory_service.modules.rag.indexer import KNOWLEDGE, MEMORIES
    from memory_service.ports.search import SearchFilter

    engine = container.database.engine  # type: ignore[attr-defined]
    dedicated_database(str(engine.url.render_as_string(hide_password=True)))

    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))

    removed: dict[str, int] = {}
    indexer = container.services["indexer"]  # type: ignore[attr-defined]
    flt = SearchFilter(tenant_id=tenant_id)
    for base in (KNOWLEDGE, MEMORIES):
        name = indexer.collection(base)
        try:
            removed[base] = await container.search.delete_by_filter(name, flt)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - a missing collection is not a failure
            removed[base] = -1
            sys.stderr.write(f"reset_store: {name}: {type(exc).__name__}: {exc}\n")
    return removed
