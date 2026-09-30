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
    """A request that names its user only under the removed spelling has no user: the
    user-scoped read is refused instead of silently acting for ``u1``."""
    r = client.get("/v1/memories", headers=OLD_H)
    assert r.status_code in (400, 403, 422), r.text
    assert client.get("/v1/memories", headers=H).status_code == 200
