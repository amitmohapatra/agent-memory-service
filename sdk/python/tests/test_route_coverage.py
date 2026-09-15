"""Every public route in the committed OpenAPI schema is reachable through the SDK: the
client source must reference each path (templated ids become f-string prefixes)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
OPENAPI = ROOT / "docs" / "openapi.json"
CLIENT = ROOT / "sdk" / "python" / "src" / "universal_memory" / "client.py"

# ops routes are reached through MemoryClient.health()/version(); /metrics is for Prometheus
EXEMPT = {("GET", "/metrics")}


@pytest.mark.unit
def test_every_openapi_route_has_an_sdk_call() -> None:
    schema = json.loads(OPENAPI.read_text())
    source = CLIENT.read_text()
    missing = []
    for path, ops in schema["paths"].items():
        for method in ops:
            key = (method.upper(), path)
            if key in EXEMPT:
                continue
            # "/v1/memories/{memory_id}" -> "/v1/memories/{" ; "/health/live" -> as is
            needle = re.sub(r"\{[^}]+\}.*$", "{", path)
            if needle not in source:
                missing.append(f"{method.upper()} {path}")
    assert not missing, missing
