"""One contract, every BlobStore adapter.

The archive path depends on three promises: an upload is immutable, a checksum mismatch is
detectable, and a missing object raises rather than returning empty bytes. Every adapter has
to keep all three, or ``blob.provider`` is not a free choice.
"""

from __future__ import annotations

import hashlib
import uuid

import pytest
import pytest_asyncio

from memory_service.ports.blob import BlobNotFound

pytestmark = pytest.mark.contract

ADAPTERS = ("memory", "filesystem", "gcs")
BUCKET = "contract"


def _build(name: str, tmp_path):
    if name == "memory":
        from memory_service.adapters.blob.memory import MemoryBlobStore

        return MemoryBlobStore()
    if name == "filesystem":
        from memory_service.adapters.blob.filesystem import FilesystemBlobStore

        return FilesystemBlobStore(tmp_path / "blob")
    from memory_service.adapters.blob.gcs import GCSBlobStore
    from memory_service.config.settings import BlobSettings

    return GCSBlobStore(BlobSettings(provider="gcs"))


@pytest_asyncio.fixture(params=ADAPTERS, loop_scope="function")
async def blob(request: pytest.FixtureRequest, tmp_path):
    try:
        store = _build(request.param, tmp_path)
    except Exception as exc:  # an adapter whose SDK or credentials are absent
        pytest.skip(f"{request.param} blob store unavailable: {type(exc).__name__}: {exc}")
    if not await store.ping():
        pytest.skip(f"{request.param} blob store not reachable")
    return store


def _key() -> str:
    return f"contract/{uuid.uuid4().hex}"


async def test_a_blob_survives_a_round_trip_with_its_checksum(blob) -> None:
    key, payload = _key(), b"archive segment bytes"
    ref = await blob.put(BUCKET, key, payload)
    assert ref.size_bytes == len(payload)
    assert ref.checksum_sha256 == hashlib.sha256(payload).hexdigest()
    assert await blob.get(BUCKET, key) == payload


async def test_a_missing_object_raises_rather_than_returning_empty(blob) -> None:
    with pytest.raises(BlobNotFound):
        await blob.get(BUCKET, _key())


async def test_an_upload_is_immutable_by_default(blob) -> None:
    """``if_generation_match=0`` means create-only. An archive that can be overwritten in
    place is not an archive."""
    key = _key()
    await blob.put(BUCKET, key, b"first")
    with pytest.raises(Exception):  # noqa: B017 - each store raises its own precondition error
        await blob.put(BUCKET, key, b"second")
    assert await blob.get(BUCKET, key) == b"first", "the original must be untouched"


async def test_head_and_verify_agree_with_what_was_written(blob) -> None:
    key, payload = _key(), b"verify me"
    ref = await blob.put(BUCKET, key, payload)
    head = await blob.head(BUCKET, key)
    assert head.checksum_sha256 == ref.checksum_sha256
    assert head.size_bytes == len(payload)
    assert await blob.verify(ref) is True


async def test_verify_rejects_a_ref_whose_checksum_does_not_match(blob) -> None:
    """The point of verify: catch the bytes changing under a reference we still hold."""
    key = _key()
    ref = await blob.put(BUCKET, key, b"original")
    tampered = ref.model_copy(update={"checksum_sha256": hashlib.sha256(b"other").hexdigest()})
    assert await blob.verify(tampered) is False


async def test_list_returns_only_the_prefix_asked_for(blob) -> None:
    tag = uuid.uuid4().hex[:8]
    mine = [f"contract/{tag}/{i}" for i in range(3)]
    for key in mine:
        await blob.put(BUCKET, key, b"x")
    await blob.put(BUCKET, f"contract/{uuid.uuid4().hex[:8]}/other", b"x")
    seen = {ref.key async for ref in blob.list(BUCKET, f"contract/{tag}/")}
    assert seen == set(mine)


async def test_delete_removes_the_object(blob) -> None:
    key = _key()
    await blob.put(BUCKET, key, b"gone soon")
    await blob.delete(BUCKET, key)
    with pytest.raises(BlobNotFound):
        await blob.get(BUCKET, key)
