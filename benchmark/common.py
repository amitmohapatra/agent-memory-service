"""Benchmark provenance. No benchmark result is stored without it."""

from __future__ import annotations

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


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


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
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
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
                "fastembed",
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
    """
    from sqlalchemy import text

    from benchmark.retrieval import TABLES
    from memory_service.modules.rag.indexer import KNOWLEDGE, MEMORIES
    from memory_service.ports.search import SearchFilter

    async with container.database.engine.begin() as conn:  # type: ignore[attr-defined]
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
