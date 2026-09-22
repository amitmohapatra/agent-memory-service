"""Contract: an invalid closed-set value is a 422 naming the allowed values.

Two cases the roadmap names explicitly, because both used to fail differently: an unknown
recall kind was dropped and the call returned an empty result, and an unknown tool
visibility raised ``ValueError`` out of the handler and came back as a 500.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.contract

HEADERS = {"X-API-Key": "test-key", "X-Memory-Tenant": "acme", "X-Memory-User": "u1"}
SCOPE = {"thread_id": "thr_1"}


def test_recall_with_an_unknown_kind_is_422(client: TestClient) -> None:
    r = client.post(
        "/v1/recall", headers=HEADERS, json={"scope": SCOPE, "query": "q", "kinds": ["fact"]}
    )
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "VALIDATION"
    assert error["details"]["errors"][0]["loc"] == ["body", "kinds", "0"]
    assert "'chunk', 'memory' or 'summary'" in error["details"]["errors"][0]["msg"]


def test_tool_record_with_an_unknown_visibility_is_422(client: TestClient) -> None:
    r = client.post(
        "/v1/tools/record",
        headers=HEADERS,
        json={"scope": SCOPE, "tool": "pricing.lookup_price", "args": {}, "visibility": "NOPE"},
    )
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "VALIDATION"
    assert error["details"]["errors"][0]["loc"] == ["body", "visibility"]
    assert "PRIVATE" in error["details"]["errors"][0]["msg"]
