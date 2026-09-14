"""BlobStore port. GCS in production; filesystem (or SeaweedFS S3 profile) locally.

Writes are immutable: ``put`` returns the object generation and checksum reported by the
store, and ``verify`` re-reads metadata so the archive worker can confirm before it
marks a staged payload as archived.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class BlobRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    bucket: str
    key: str
    generation: str | None = Field(default=None, description="store-assigned immutable version id")
    size_bytes: int
    checksum_sha256: str
    checksum_store: str | None = Field(
        default=None, description="store-native checksum, e.g. crc32c"
    )
    storage_class: str | None = None

    @property
    def uri(self) -> str:
        return f"{self.bucket}/{self.key}"


class BlobNotFound(Exception):
    pass


class BlobChecksumMismatch(Exception):
    pass


@runtime_checkable
class BlobStore(Protocol):
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
        """Immutable upload. ``if_generation_match=0`` means 'create only, never overwrite'."""
        ...

    async def get(self, bucket: str, key: str, *, generation: str | None = None) -> bytes: ...

    async def head(self, bucket: str, key: str) -> BlobRef: ...

    async def verify(self, ref: BlobRef) -> bool:
        """True when the object exists with the same generation and checksum."""
        ...

    async def delete(self, bucket: str, key: str) -> None: ...

    async def list(self, bucket: str, prefix: str) -> AsyncIterator[BlobRef]: ...

    async def ping(self) -> bool: ...
