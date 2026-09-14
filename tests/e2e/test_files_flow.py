from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from memory_service.domain.ids import new_id
from tests.e2e.conftest import sdk_client

pytestmark = pytest.mark.e2e
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "acme_fy26_annual_report.md"
H = {"X-API-Key": "test-key", "X-Memory-Tenant": "acme", "X-Memory-User": "u1"}


def test_upload_parse_and_status(client) -> None:
    scope = {
        "thread_id": new_id("thread"),
        "session_id": new_id("session"),
        "turn_id": new_id("turn"),
    }
    msg = client.post(
        "/v1/messages",
        headers=H,
        json={"scope": scope, "role": "USER", "content": "here is the report"},
    ).json()
    r = client.post(
        "/v1/files",
        headers=H,
        files={"file": ("acme_fy26_annual_report.md", FIXTURE.read_bytes(), "text/markdown")},
        data={"scope": json.dumps(scope), "message_id": msg["message_id"], "title": "ACME FY26"},
    )
    assert r.status_code == 202, r.text
    ack = r.json()
    assert ack["job_ids"] and ack["checksum"]
    doc = client.get(f"/v1/documents/{ack['document_id']}", headers=H).json()
    assert (
        doc["status"] == "READY"
        and doc["archive_status"] == "ARCHIVED"
        and doc["title"] == "ACME FY26"
    )
    job = client.get(f"/v1/jobs/{ack['job_ids'][0]}", headers=H).json()
    assert job["status"] == "SUCCEEDED"
    # same bytes again -> dedup
    again = client.post(
        "/v1/files",
        headers=H,
        files={"file": ("copy.md", FIXTURE.read_bytes(), "text/markdown")},
        data={"scope": json.dumps(scope)},
    ).json()
    assert again["deduplicated"] and again["document_id"] == ack["document_id"]
    # other user cannot see the document (thread-scoped)
    assert (
        client.get(
            f"/v1/documents/{ack['document_id']}", headers={**H, "X-Memory-User": "u2"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/v1/files",
            headers=H,
            files={"file": ("x.exe", b"MZ", "application/x-msdownload")},
            data={"scope": "{}"},
        ).status_code
        == 422
    )


async def test_sdk_attachments(app, client) -> None:
    memory = sdk_client(app)
    ctx = memory.bind(
        tenant_id="acme",
        user_id="u1",
        thread_id=new_id("thread"),
        session_id=new_id("session"),
        turn_id=new_id("turn"),
    )
    ack = await ctx.chat.user(
        "summarise this", attachments=[("notes.md", b"# Notes\n\nA short note.", "text/markdown")]
    )
    handle = await ctx.files.add(
        io.BytesIO(b"# Two\n\nAnother.").getvalue(),
        filename="two.md",
        media_type="text/markdown",
        message_id=ack.message_id,
    )
    assert handle.document_id and handle.size_bytes == len(b"# Two\n\nAnother.")
    await memory.aclose()
