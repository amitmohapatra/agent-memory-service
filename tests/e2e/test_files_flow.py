from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from memory_service.domain.ids import new_id
from tests.e2e.conftest import post_message, sdk_client

pytestmark = pytest.mark.e2e
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "acme_fy26_annual_report.md"
H = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "u1"}


def test_upload_parse_and_status(client) -> None:
    scope = {
        "thread_id": new_id("thread"),
        "session_id": new_id("session"),
        "turn_id": new_id("turn"),
    }
    msg = post_message(
        client, H, {"scope": scope, "role": "USER", "content": "here is the report"}
    ).json()
    r = client.post(
        "/v1/documents",
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
        "/v1/documents",
        headers=H,
        files={"file": ("copy.md", FIXTURE.read_bytes(), "text/markdown")},
        data={"scope": json.dumps(scope)},
    ).json()
    assert again["deduplicated"] and again["document_id"] == ack["document_id"]
    # other user cannot see the document (thread-scoped)
    assert (
        client.get(
            f"/v1/documents/{ack['document_id']}", headers={**H, "X-Trellis-User": "u2"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/v1/documents",
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
    [ack] = await ctx.history.add([("USER", "summarise this")])
    notes = await ctx.advanced.documents.add(
        ("notes.md", b"# Notes\n\nA short note.", "text/markdown"), message_id=ack.message_id
    )
    assert notes.document_id
    handle = await ctx.advanced.documents.add(
        io.BytesIO(b"# Two\n\nAnother.").getvalue(),
        filename="two.md",
        media_type="text/markdown",
        message_id=ack.message_id,
    )
    assert handle.document_id and handle.size_bytes == len(b"# Two\n\nAnother.")
    await memory.aclose()


def test_upload_into_a_new_thread_creates_it_and_is_retrievable(client) -> None:
    """A document uploaded into a thread nobody has written to yet is retrievable from it.

    Messages create their thread on demand, and so does an upload: the uploader owns the new
    thread, so the document's THREAD audience is one the uploader can read. Before, the
    upload stored a THREAD key for a thread that did not exist, nobody was granted it, and the
    document reached READY while recall and context never returned it.
    """
    scope = {"thread_id": new_id("thread")}
    r = client.post(
        "/v1/documents",
        headers=H,
        files={"file": ("acme_fy26_annual_report.md", FIXTURE.read_bytes(), "text/markdown")},
        data={"scope": json.dumps(scope), "title": "ACME FY26"},
    )
    assert r.status_code == 202, r.text
    doc_id = r.json()["document_id"]
    doc = client.get(f"/v1/documents/{doc_id}", headers=H).json()
    assert doc["status"] == "READY" and doc["thread_id"] == scope["thread_id"]

    query = "Why did Adjusted EBITDA increase despite lower revenue?"
    recall = client.post("/v1/recall", headers=H, json={"scope": scope, "query": query})
    assert recall.status_code == 200, recall.text
    assert any(i.get("document_id") == doc_id for i in recall.json()["items"])
    bundle = client.post(
        "/v1/context", headers=H, json={"scope": scope, "query": query, "format": "full"}
    )
    assert bundle.status_code == 200, bundle.text
    assert any(k.get("document_id") == doc_id for k in bundle.json()["knowledge"])

    # the thread exists now, and is the uploader's
    thread = client.get(f"/v1/threads/{scope['thread_id']}", headers=H)
    assert thread.status_code == 200, thread.text

    # still the thread's: another user of the tenant reads neither the thread nor the document
    u2 = {**H, "X-Trellis-User": "u2"}
    assert client.get(f"/v1/threads/{scope['thread_id']}", headers=u2).status_code == 403
    assert client.get(f"/v1/documents/{doc_id}", headers=u2).status_code == 403
    other = client.post("/v1/recall", headers=u2, json={"scope": scope, "query": query})
    assert other.status_code == 200, other.text
    assert all(i.get("document_id") != doc_id for i in other.json()["items"])
