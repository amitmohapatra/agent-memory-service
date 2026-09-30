"""The GCS blob provider end to end, against a fake GCS server (opt-in).

Every path that writes to the blob store, through the real ``GCSBlobStore`` and the real
google-cloud-storage client: a chat thread archived into a segment (verified by checksum,
its hot payloads purged only after the grace period, and still readable from the bucket),
an uploaded document's raw bytes, and a tool output over the inline limit.

Opt in with a fake GCS server (``fsouza/fake-gcs-server``) and its address::

    docker run -d --name fake-gcs -p 4443:4443 fsouza/fake-gcs-server:1.52 \\
        -scheme http -port 4443 -backend memory -external-url http://localhost:4443
    MEMORY_TEST_GCS_EMULATOR=http://localhost:4443 uv run pytest tests/integration/test_gcs_blob.py

``STORAGE_EMULATOR_HOST`` is the client library's own switch: with it set the client talks
to the emulator with anonymous credentials, so the adapter under test is unchanged.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from memory_service.__about__ import __version__
from memory_service.application.container import Container, build_container
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.conversation import Thread
from memory_service.domain.enums import ArchiveStatus
from memory_service.modules.tools.service import INLINE_OUTPUT_LIMIT
from tests.conftest import PG_AVAILABLE
from tests.integration.conftest import TABLES, integration_overrides, integration_settings
from tests.integration.test_archive import TENANT, _seed

EMULATOR = os.environ.get("MEMORY_TEST_GCS_EMULATOR")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not EMULATOR, reason="set MEMORY_TEST_GCS_EMULATOR to a fake GCS server"),
]


@pytest.fixture
async def gcs(make_settings, monkeypatch) -> AsyncIterator[Container]:
    if not PG_AVAILABLE:
        pytest.skip("PostgreSQL not reachable")
    pytest.importorskip("google.cloud.storage")
    from google.cloud import storage

    monkeypatch.setenv("STORAGE_EMULATOR_HOST", str(EMULATOR))
    run = uuid.uuid4().hex[:8]
    chat, files = f"chat-{run}", f"files-{run}"
    client = storage.Client(project="memory-tests")
    for bucket in (chat, files):
        client.create_bucket(bucket)
    settings = integration_settings(
        make_settings,
        blob={
            "provider": "gcs",
            "chat_bucket": chat,
            "file_bucket": files,
            "gcs_project": "memory-tests",
        },
    )
    c = await build_container(settings, __version__, overrides=integration_overrides(blob=None))
    async with c.database.engine.begin() as conn:
        await conn.execute(text("SET LOCAL statement_timeout = 0"))
        await conn.execute(text("TRUNCATE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE"))
    try:
        yield c
    finally:
        await c.aclose()
        for bucket in (chat, files):
            b = client.bucket(bucket)
            for blob in client.list_blobs(bucket):
                blob.delete()
            b.delete()


def _objects(container: Container, bucket: str) -> dict[str, bytes]:
    from google.cloud import storage

    client = storage.Client(project="memory-tests")
    return {b.name: b.download_as_bytes() for b in client.list_blobs(bucket)}


async def test_the_provider_is_gcs_and_ready(gcs) -> None:
    from memory_service.adapters.blob.gcs import GCSBlobStore

    assert isinstance(gcs.blob, GCSBlobStore)
    assert await gcs.blob.ping()


async def test_a_chat_thread_is_archived_verified_and_purged_after_grace(gcs) -> None:
    factory = gcs.services["uow_factory"]
    archive = gcs.services["archive_service"]
    thread = Thread(tenant_id=TENANT, owner_user_id="u1")
    msgs = await _seed(factory, thread, 6)

    [segment_id] = await archive.archive_thread(TENANT, thread.thread_id)
    async with factory() as uow:
        seg = await uow.archive.get(segment_id)
    assert seg is not None and seg.status == "VERIFIED" and seg.generation
    stored = _objects(gcs, gcs.settings.blob.chat_bucket)
    assert list(stored) == [seg.key]
    # the object in the bucket is the segment the database recorded, byte for byte
    assert hashlib.sha256(stored[seg.key]).hexdigest() == seg.checksum_sha256
    assert await gcs.blob.verify(await gcs.blob.head(seg.bucket, seg.key))

    assert await archive.purge_staged_payloads() == 0, "nothing is purged inside the grace"
    later = datetime.now(UTC) + timedelta(seconds=gcs.tuning.archive.purge_grace_seconds + 1)
    assert await archive.purge_staged_payloads(now=later) == 6
    async with factory() as uow:
        purged = await uow.messages.get(TENANT, msgs[3].message_id)
    assert purged is not None and purged.archive_status is ArchiveStatus.PURGED
    assert purged.content == ""
    # read back from GCS, hash-checked
    assert await archive.load_message_content(purged) == msgs[3].content


async def test_an_uploaded_document_lands_in_the_file_bucket(gcs) -> None:
    ingestion = gcs.services["ingestion"]
    ctx = MemoryExecutionContext(tenant_id=TENANT, user_id="u1")
    body = b"Release policy.\n\nEvery release needs a named approver.\n"
    async with gcs.services["uow_factory"]() as uow:
        ack = await ingestion.accept_file(
            uow, ctx, filename="policy.txt", media_type="text/plain", data=body
        )
        await uow.commit()
    segment_id = await ingestion.archive_raw_file(TENANT, ack.document_id)
    assert segment_id is not None
    [(key, data)] = _objects(gcs, gcs.settings.blob.file_bucket).items()
    assert data == body and key.endswith(f"{hashlib.sha256(body).hexdigest()}.bin")
    async with gcs.services["uow_factory"]() as uow:
        document = await uow.documents.get(TENANT, ack.document_id)
        staged = await uow.documents.staged_bytes(TENANT, ack.document_id)
    assert document is not None and document.archive_status is ArchiveStatus.ARCHIVED
    assert staged is None, "the hot copy is purged once the archive is verified"


async def test_a_tool_output_over_the_inline_limit_lands_in_the_bucket(gcs) -> None:
    tools = gcs.services["tool_memory"]
    ctx = MemoryExecutionContext(
        tenant_id=TENANT, user_id="u1", agent_id="bot", agent_run_id="run_gcs"
    )
    output = "row " * (INLINE_OUTPUT_LIMIT // 2)
    for step in (0, 1):  # the same output twice: one object, both calls point at it
        async with gcs.services["uow_factory"]() as uow:
            invocation, _ = await tools.record(
                uow, ctx, tool="erp-export", args={"page": step}, output=output, step=step
            )
            await uow.commit()
        assert invocation.output_blob_ref is not None
    stored = _objects(gcs, gcs.settings.blob.file_bucket)
    [(key, data)] = stored.items()
    assert invocation.output_blob_ref == f"{gcs.settings.blob.file_bucket}/{key}"
    assert data.decode() == output and key.startswith(f"{TENANT}/tool-output/")
