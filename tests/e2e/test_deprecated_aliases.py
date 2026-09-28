"""The one-release aliases of the renamed nouns (ADR 0022) are the same routes: the same
handler, the same idempotency, the same rows, and they say on the wire that they are aliases."""

from __future__ import annotations

import json

import pytest

from memory_service.api.headers import ALIASES_DEPRECATED_AT
from memory_service.domain.ids import new_id

pytestmark = pytest.mark.e2e
H = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "u1"}
OLD_H = {"X-API-Key": "test-key", "X-Memory-Tenant": "acme", "X-Memory-User": "u1"}


def _assert_marked_deprecated(response, successor: str) -> None:
    assert response.headers["Deprecation"] == ALIASES_DEPRECATED_AT
    assert response.headers["Link"] == f'<{successor}>; rel="successor-version"'


def test_post_files_is_post_documents(client) -> None:
    scope = {
        "thread_id": new_id("thread"),
        "session_id": new_id("session"),
        "turn_id": new_id("turn"),
    }
    # attached to a message, as an attachment is: the caller can read it back through the
    # thread, which is what makes the second upload a duplicate rather than a new document
    opened = client.post(
        "/v1/messages",
        headers=H,
        json={"scope": scope, "role": "USER", "content": "here is the note"},
    )
    assert opened.status_code == 202, opened.text
    data = b"# Alias\n\nThe same bytes through both spellings of the route.\n"
    canonical = client.post(
        "/v1/documents",
        headers=H,
        files={"file": ("alias.md", data, "text/markdown")},
        data={"scope": json.dumps(scope), "message_id": opened.json()["message_id"]},
    )
    assert canonical.status_code == 202, canonical.text
    assert "Deprecation" not in canonical.headers
    legacy = client.post(
        "/v1/files",
        headers=OLD_H,
        files={"file": ("alias.md", data, "text/markdown")},
        data={"scope": json.dumps(scope)},
    )
    assert legacy.status_code == 202, legacy.text
    assert legacy.json()["deduplicated"] is True
    assert legacy.json()["document_id"] == canonical.json()["document_id"]
    _assert_marked_deprecated(legacy, "/v1/documents")


def test_post_tools_record_is_post_tools_invocations(client) -> None:
    body = {
        "scope": {
            "thread_id": new_id("thread"),
            "agent_id": "bot",
            "agent_run_id": new_id("agent_run"),
        },
        "tool": "search",
        "args": {"q": "alias"},
        "output_summary": "one hit",
        "task": "find the alias",
        "step": 0,
    }
    first = client.post("/v1/tools/invocations", headers=H, json=body)
    assert first.status_code == 202, first.text
    assert first.json()["recorded"] is True and "Deprecation" not in first.headers
    again = client.post("/v1/tools/record", headers=OLD_H, json=body)
    assert again.status_code == 202, again.text
    assert again.json()["recorded"] is False
    assert again.json()["invocation_id"] == first.json()["invocation_id"]
    _assert_marked_deprecated(again, "/v1/tools/invocations")
