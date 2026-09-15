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
    downloaded into ``MEMORY_MODELS_DIR`` / ``./models``)."""
    root = Path(os.environ.get("MEMORY_MODELS_DIR", "models"))
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
    out: dict[str, Any] = {"enabled": llm.enabled, "provider": llm.provider}
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
