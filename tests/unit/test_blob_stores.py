"""BlobStore contract: immutability, generations, checksums, verification, listing."""

from __future__ import annotations

import pytest

from memory_service.adapters.blob.filesystem import BlobAlreadyExists, FilesystemBlobStore
from memory_service.adapters.blob.memory import MemoryBlobStore
from memory_service.ports.blob import BlobChecksumMismatch, BlobNotFound


@pytest.fixture(params=["filesystem", "memory"])
def store(request, tmp_path):
    return (
        FilesystemBlobStore(tmp_path / "blob")
        if request.param == "filesystem"
        else MemoryBlobStore()
    )


async def test_put_is_immutable_and_versioned(store) -> None:
    ref = await store.put("bucket", "a/b/c.bin", b"hello", content_type="text/plain")
    assert ref.generation == "1" and ref.size_bytes == 5 and ref.checksum_sha256
    with pytest.raises(BlobAlreadyExists):
        await store.put("bucket", "a/b/c.bin", b"other")
    ref2 = await store.put("bucket", "a/b/c.bin", b"other", if_generation_match=None)
    assert ref2.generation == "2"
    assert await store.get("bucket", "a/b/c.bin") == b"other"
    with pytest.raises(BlobNotFound):
        await store.get("bucket", "a/b/c.bin", generation="1")
    assert await store.verify(ref) is False  # superseded generation no longer verifies
    assert await store.verify(ref2) is True


async def test_checksum_and_missing(store) -> None:
    with pytest.raises(BlobChecksumMismatch):
        await store.put("bucket", "x", b"data", checksum_sha256="0" * 64)
    with pytest.raises(BlobNotFound):
        await store.head("bucket", "missing")
    assert (
        await store.verify(
            (await store.put("bucket", "y", b"1")).model_copy(update={"checksum_sha256": "f" * 64})
        )
        is False
    )


async def test_list_and_delete(store) -> None:
    await store.put("bucket", "p/1", b"1")
    await store.put("bucket", "p/2", b"22")
    await store.put("bucket", "q/1", b"3")
    keys = [r.key async for r in store.list("bucket", "p/")]
    assert keys == ["p/1", "p/2"]
    await store.delete("bucket", "p/1")
    with pytest.raises(BlobNotFound):
        await store.head("bucket", "p/1")


async def test_outage_surfaces_as_error(store) -> None:
    store.available = False
    with pytest.raises(ConnectionError):
        await store.put("bucket", "z", b"1")
    assert await store.ping() is False


async def test_filesystem_detects_tampering(tmp_path) -> None:
    store = FilesystemBlobStore(tmp_path)
    ref = await store.put("b", "k", b"original")
    (tmp_path / "b" / "k").write_bytes(b"tampered")
    assert await store.verify(ref) is False
    with pytest.raises(BlobChecksumMismatch):
        await store.get("b", "k")
    with pytest.raises(ValueError):
        await store.put("b", "../escape", b"x")
