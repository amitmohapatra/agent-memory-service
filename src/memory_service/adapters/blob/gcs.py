"""Google Cloud Storage BlobStore (production). Requires the ``gcp`` extra.

Immutable uploads use ``if_generation_match=0``; verification re-reads object metadata and
compares generation + CRC32C/MD5-derived checksum. Calls run in a thread (the GCS client
is synchronous) so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator

from memory_service.config.settings import BlobSettings
from memory_service.domain.errors import DependencyUnavailable
from memory_service.domain.ids import content_hash
from memory_service.ports.blob import BlobChecksumMismatch, BlobNotFound, BlobRef
from memory_service.ports.models import ProviderInfo


class GCSBlobStore:
    info = ProviderInfo(
        name="gcs",
        license="Apache-2.0",
        origin="googleapis/python-storage",
        locality="remote",
        data_residency="gcp-region",
    )

    def __init__(self, settings: BlobSettings) -> None:
        try:
            from google.cloud import storage  # type: ignore[attr-defined]
        except ImportError as exc:
            raise DependencyUnavailable("google-cloud-storage is required (install [gcp])") from exc
        self._client = storage.Client(project=settings.gcs_project)
        self.settings = settings

    @staticmethod
    def _ref(blob, bucket: str, key: str, sha256: str | None) -> BlobRef:  # type: ignore[no-untyped-def]
        md5 = blob.md5_hash
        crc = blob.crc32c
        return BlobRef(
            bucket=bucket,
            key=key,
            generation=str(blob.generation),
            size_bytes=int(blob.size or 0),
            checksum_sha256=sha256 or (blob.metadata or {}).get("sha256", ""),
            checksum_store=f"crc32c:{crc}" if crc else (f"md5:{md5}" if md5 else None),
            storage_class=blob.storage_class,
        )

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
        digest = content_hash(data)
        if checksum_sha256 is not None and checksum_sha256 != digest:
            raise BlobChecksumMismatch("client checksum mismatch")

        def _upload() -> BlobRef:
            from google.api_core import exceptions as gexc

            blob = self._client.bucket(bucket).blob(key)
            blob.metadata = {**(metadata or {}), "sha256": digest}
            try:
                blob.upload_from_string(
                    data,
                    content_type=content_type,
                    if_generation_match=if_generation_match,
                    checksum="crc32c",
                )
            except gexc.PreconditionFailed as exc:
                from memory_service.adapters.blob.filesystem import BlobAlreadyExists

                raise BlobAlreadyExists(f"{bucket}/{key} exists") from exc
            blob.reload()
            return self._ref(blob, bucket, key, digest)

        return await asyncio.to_thread(_upload)

    async def get(self, bucket: str, key: str, *, generation: str | None = None) -> bytes:
        def _download() -> bytes:
            from google.api_core import exceptions as gexc

            blob = self._client.bucket(bucket).blob(
                key, generation=int(generation) if generation else None
            )
            try:
                return blob.download_as_bytes()
            except gexc.NotFound as exc:
                raise BlobNotFound(f"{bucket}/{key}") from exc

        return await asyncio.to_thread(_download)

    async def head(self, bucket: str, key: str) -> BlobRef:
        def _head() -> BlobRef:
            blob = self._client.bucket(bucket).get_blob(key)
            if blob is None:
                raise BlobNotFound(f"{bucket}/{key}")
            return self._ref(blob, bucket, key, None)

        return await asyncio.to_thread(_head)

    async def verify(self, ref: BlobRef) -> bool:
        try:
            current = await self.head(ref.bucket, ref.key)
        except BlobNotFound:
            return False
        if current.generation != ref.generation:
            return False
        if (
            current.checksum_sha256
            and ref.checksum_sha256
            and current.checksum_sha256 != ref.checksum_sha256
        ):
            return False
        return current.size_bytes == ref.size_bytes

    async def delete(self, bucket: str, key: str) -> None:
        await asyncio.to_thread(self._client.bucket(bucket).blob(key).delete)

    async def list(self, bucket: str, prefix: str) -> AsyncIterator[BlobRef]:
        blobs = await asyncio.to_thread(
            lambda: list(self._client.list_blobs(bucket, prefix=prefix))
        )
        for blob in blobs:
            yield self._ref(blob, bucket, blob.name, None)

    async def ping(self) -> bool:
        try:
            await asyncio.to_thread(lambda: self._client.bucket(self.settings.chat_bucket).exists())
            return True
        except Exception:
            return False


def crc32c_b64(data: bytes) -> str:  # pragma: no cover - helper for tooling
    import google_crc32c  # type: ignore[import-not-found]

    return base64.b64encode(google_crc32c.value(data).to_bytes(4, "big")).decode()
