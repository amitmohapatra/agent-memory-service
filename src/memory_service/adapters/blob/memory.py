"""In-memory BlobStore (unit tests)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from memory_service.adapters.blob.filesystem import BlobAlreadyExists
from memory_service.domain.ids import content_hash
from memory_service.ports.blob import BlobChecksumMismatch, BlobNotFound, BlobRef
from memory_service.ports.models import ProviderInfo


class MemoryBlobStore:
    info = ProviderInfo(name="memory", license="Apache-2.0", origin="internal", locality="local")

    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], tuple[bytes, BlobRef]] = {}
        self.available = True
        self.corrupt_next_verify = False

    def _check(self) -> None:
        if not self.available:
            raise ConnectionError("simulated blob store outage")

    async def put(
        self,
        bucket: str,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        checksum_sha256: str | None = None,
        if_generation_match: int | None = 0,
        metadata: dict[str, str] | None = None,
    ) -> BlobRef:
        self._check()
        digest = content_hash(data)
        if checksum_sha256 is not None and checksum_sha256 != digest:
            raise BlobChecksumMismatch("checksum mismatch")
        existing = self._objects.get((bucket, key))
        if if_generation_match == 0 and existing is not None:
            raise BlobAlreadyExists(f"{bucket}/{key} exists")
        generation = int(existing[1].generation or 0) + 1 if existing else 1
        ref = BlobRef(
            bucket=bucket,
            key=key,
            generation=str(generation),
            size_bytes=len(data),
            checksum_sha256=digest,
            checksum_store=digest,
            storage_class="STANDARD",
        )
        self._objects[(bucket, key)] = (data, ref)
        return ref

    async def get(self, bucket: str, key: str, *, generation: str | None = None) -> bytes:
        self._check()
        item = self._objects.get((bucket, key))
        if item is None or (generation is not None and item[1].generation != generation):
            raise BlobNotFound(f"{bucket}/{key}")
        return item[0]

    async def head(self, bucket: str, key: str) -> BlobRef:
        self._check()
        item = self._objects.get((bucket, key))
        if item is None:
            raise BlobNotFound(f"{bucket}/{key}")
        return item[1]

    async def verify(self, ref: BlobRef) -> bool:
        self._check()
        if self.corrupt_next_verify:
            self.corrupt_next_verify = False
            return False
        item = self._objects.get((ref.bucket, ref.key))
        return (
            item is not None
            and item[1].generation == ref.generation
            and content_hash(item[0]) == ref.checksum_sha256
        )

    async def delete(self, bucket: str, key: str) -> None:
        self._check()
        self._objects.pop((bucket, key), None)

    async def list(self, bucket: str, prefix: str) -> AsyncIterator[BlobRef]:
        self._check()
        for (b, k), (_, ref) in sorted(self._objects.items()):
            if b == bucket and k.startswith(prefix):
                yield ref

    async def ping(self) -> bool:
        return self.available
