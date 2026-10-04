"""The one-release aliases 0.2.0 kept (ADR 0022) are gone in 0.3.0: one route per operation,
one spelling per header."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e
OLD_H = {"X-API-Key": "test-key", "X-Memory-Tenant": "acme", "X-Memory-User": "u1"}
H = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "u1"}


def test_post_files_is_not_a_route(client) -> None:
    r = client.post("/v1/files", headers=H, files={"file": ("a.md", b"# a", "text/markdown")})
    assert r.status_code == 404
    assert "Deprecation" not in r.headers


def test_the_old_header_spellings_name_no_scope(client) -> None:
    """A request that names its tenant and user only under the removed spelling names
    neither: it never silently acts for ``acme``/``u1``. (With a development key it acts in
    the development tenant with no user, which reads none of u1's memories.)"""
    written = client.post(
        "/v1/memories",
        headers=H,
        json={"scope": {}, "content": "u1 keeps the quarterly numbers.", "visibility": "USER"},
    )
    assert written.status_code in (200, 201), written.text
    r = client.get("/v1/memories", headers=OLD_H)
    if r.status_code == 200:
        assert written.json()["memory_id"] not in {m["memory_id"] for m in r.json()["memories"]}
    else:
        assert r.status_code in (400, 403, 422), r.text
    listed = client.get("/v1/memories", headers=H)
    assert listed.status_code == 200
    assert written.json()["memory_id"] in {m["memory_id"] for m in listed.json()["memories"]}
