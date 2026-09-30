"""Every public route in the committed OpenAPI schema is reachable through the SDK: the
package source must reference each path (templated ids become f-string prefixes)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
OPENAPI = ROOT / "docs" / "openapi.json"
PACKAGE = ROOT / "sdk" / "python" / "src" / "trellis" / "memory"

# ops routes are reached through MemoryClient.health()/alive()/version()/metrics().
# Deprecated routes are one-release aliases (ADR 0022): the SDK speaks the canonical noun only.
EXEMPT: set[tuple[str, str]] = set()


@pytest.mark.unit
def test_every_openapi_route_has_an_sdk_call() -> None:
    schema = json.loads(OPENAPI.read_text())
    source = "\n".join(p.read_text() for p in sorted(PACKAGE.glob("*.py")))
    missing = []
    for path, ops in schema["paths"].items():
        for method, op in ops.items():
            key = (method.upper(), path)
            if key in EXEMPT or op.get("deprecated"):
                continue
            # "/v1/memories/{memory_id}" -> "/v1/memories/{" ; "/health/live" -> as is
            needle = re.sub(r"\{[^}]+\}.*$", "{", path)
            if needle not in source:
                missing.append(f"{method.upper()} {path}")
    assert not missing, missing
