"""A tool call's output: the row keeps a summary (what the context reads), the blob store the
whole output once it is longer than that, compressed and addressed by its digest under the
tenant. Nothing is cut: an upload that fails is retried by the outbox until it is stored."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
import zstandard

from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.tools.service import (
    SUMMARY_CHARS,
    TASK_TOOLS_OUTPUT,
    output_key,
)

pytestmark = pytest.mark.integration


async def read_output(blob: Any, ref: str) -> str:
    """A stored output, whole: ``.txt.zst`` (this release) or ``.txt`` (earlier ones)."""
    bucket, _, key = ref.partition("/")
    data = await blob.get(bucket, key)
    if key.endswith(".zst"):
        data = zstandard.ZstdDecompressor().decompress(data)
    return data.decode("utf-8", "replace")


def _ctx(tenant: str = "acme") -> MemoryExecutionContext:
    return MemoryExecutionContext(
        tenant_id=tenant, user_id="u1", agent_id="bot", agent_run_id=f"run_{tenant}"
    )


async def _record(container: Any, output: str, *, step: int = 0, tenant: str = "acme") -> Any:
    async with container.services["uow_factory"]() as uow:
        invocation, _ = await container.services["tool_memory"].record(
            uow, _ctx(tenant), tool="erp-export", args={"step": step}, output=output, step=step
        )
        await uow.commit()
    return invocation


@pytest.mark.parametrize("size", [600, 601, 4096, 4097])
async def test_an_output_past_the_summary_is_kept_whole(container: Any, size: int) -> None:
    output = ("ünïcode row;" * (size // 12 + 1))[:size]
    invocation = await _record(container, output)
    assert invocation.output_summary == output[:SUMMARY_CHARS]
    assert invocation.output_digest == hashlib.sha256(output.encode()).hexdigest()
    if size <= SUMMARY_CHARS:
        assert invocation.output_blob_ref is None, "the summary is the whole output"
        return
    assert invocation.output_blob_ref.endswith(f"{output_key('acme', invocation.output_digest)}")
    whole = await read_output(container.blob, invocation.output_blob_ref)
    assert (
        whole == output and hashlib.sha256(whole.encode()).hexdigest() == invocation.output_digest
    )


async def test_the_same_output_is_stored_once_and_each_tenant_has_its_own(container: Any) -> None:
    output = "x" * 5000
    first = await _record(container, output, step=0)
    again = await _record(container, output, step=1)
    other = await _record(container, output, tenant="globex")
    assert first.output_blob_ref == again.output_blob_ref
    assert other.output_blob_ref != first.output_blob_ref
    assert "/tenant=acme/" in first.output_blob_ref and "/tenant=globex/" in other.output_blob_ref
    assert first.output_blob_ref.split("/", 1)[1].startswith("tenant-shard=")
    assert await read_output(container.blob, other.output_blob_ref) == output


async def test_an_upload_that_fails_is_retried_until_stored_never_cut(
    container: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    blob = container.blob
    original = blob.put

    async def down(*args: Any, **kwargs: Any) -> Any:
        raise OSError("the store is down")

    monkeypatch.setattr(blob, "put", down)
    output = "row\n" * 3000
    invocation = await _record(container, output)
    assert invocation.output_blob_ref is not None, "the reference holds: the content's address"
    with pytest.raises(Exception):  # noqa: B017 - not stored yet
        await read_output(blob, invocation.output_blob_ref)
    queue = container.tasks
    await container.services["outbox_relay"].sweep(older_than_seconds=0)
    assert [j.task_name for j in queue.jobs.values()] == [TASK_TOOLS_OUTPUT]
    monkeypatch.setattr(blob, "put", original)  # the store is back: the outbox's job stores it
    await queue.drain()
    assert await read_output(blob, invocation.output_blob_ref) == output


async def test_an_output_an_earlier_release_stored_uncompressed_still_reads(container: Any) -> None:
    output = "legacy " * 1000
    digest = hashlib.sha256(output.encode()).hexdigest()
    bucket = container.settings.blob.file_bucket
    await container.blob.put(bucket, f"acme/tool-output/{digest}.txt", output.encode())
    assert await read_output(container.blob, f"{bucket}/acme/tool-output/{digest}.txt") == output
