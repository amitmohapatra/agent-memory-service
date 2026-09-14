"""Filesystem BlobStore with GCS-like semantics: immutable objects, generations, checksums.

Layout: ``<root>/<bucket>/<key>`` plus ``<root>/<bucket>/<key>.meta.json`` holding the
generation (monotonic per key), sha256, size, content type and user metadata.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from memory_service.domain.ids import content_hash
from memory_service.ports.blob import BlobChecksumMismatch, BlobNotFound, BlobRef
from memory_service.ports.models import ProviderInfo


class BlobAlreadyExists(Exception):
    pass


class FilesystemBlobStore:
    info = ProviderInfo(
        name="filesystem", license="Apache-2.0", origin="internal", locality="local"
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.available = True  # flip to False to simulate an outage
        self.corrupt_next_verify = False  # test hook

    def _path(self, bucket: str, key: str) -> Path:
        if ".." in key.split("/"):
            raise ValueError("invalid key")
        return self.root / bucket / key

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
            raise BlobChecksumMismatch(f"client checksum {checksum_sha256} != {digest}")
        path = self._path(bucket, key)
        meta_path = path.with_name(path.name + ".meta.json")
        return await asyncio.to_thread(
            self._write,
            path,
            meta_path,
            data,
            digest,
            content_type,
            if_generation_match,
            metadata or {},
        )

    def _write(
        self,
        path: Path,
        meta_path: Path,
        data: bytes,
        digest: str,
        content_type: str,
        if_generation_match: int | None,
        metadata: dict[str, str],
    ) -> BlobRef:
        path.parent.mkdir(parents=True, exist_ok=True)
        current = self._read_meta(meta_path)
        if if_generation_match == 0 and current is not None:
            raise BlobAlreadyExists(f"{path} already exists (generation {current['generation']})")
        if if_generation_match not in (None, 0) and (
            current is None or int(current["generation"]) != if_generation_match
        ):
            raise BlobAlreadyExists("generation precondition failed")
        generation = (int(current["generation"]) + 1) if current else 1
        # atomic replace: write temp + fsync + rename
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".part")
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        meta = {
            "generation": generation,
            "sha256": digest,
            "size": len(data),
            "content_type": content_type,
            "metadata": metadata,
        }
        tmp_meta = meta_path.with_suffix(".part")
        tmp_meta.write_text(json.dumps(meta), encoding="utf-8")
        os.replace(tmp_meta, meta_path)
        return BlobRef(
            bucket=path.relative_to(self.root).parts[0],
            key="/".join(path.relative_to(self.root).parts[1:]),
            generation=str(generation),
            size_bytes=len(data),
            checksum_sha256=digest,
            checksum_store=digest,
            storage_class="STANDARD",
        )

    @staticmethod
    def _read_meta(meta_path: Path) -> dict | None:
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))

    async def get(self, bucket: str, key: str, *, generation: str | None = None) -> bytes:
        self._check()
        path = self._path(bucket, key)
        if not path.exists():
            raise BlobNotFound(f"{bucket}/{key}")
        meta = self._read_meta(path.with_name(path.name + ".meta.json")) or {}
        if generation is not None and str(meta.get("generation")) != str(generation):
            raise BlobNotFound(f"{bucket}/{key}@{generation}")
        data = await asyncio.to_thread(path.read_bytes)
        if meta.get("sha256") and content_hash(data) != meta["sha256"]:
            raise BlobChecksumMismatch(f"{bucket}/{key} content does not match stored checksum")
        return data

    async def head(self, bucket: str, key: str) -> BlobRef:
        self._check()
        path = self._path(bucket, key)
        meta = self._read_meta(path.with_name(path.name + ".meta.json"))
        if meta is None or not path.exists():
            raise BlobNotFound(f"{bucket}/{key}")
        return BlobRef(
            bucket=bucket,
            key=key,
            generation=str(meta["generation"]),
            size_bytes=int(meta["size"]),
            checksum_sha256=meta["sha256"],
            checksum_store=meta["sha256"],
            storage_class="STANDARD",
        )

    async def verify(self, ref: BlobRef) -> bool:
        self._check()
        if self.corrupt_next_verify:
            self.corrupt_next_verify = False
            return False
        try:
            current = await self.head(ref.bucket, ref.key)
        except BlobNotFound:
            return False
        if current.generation != ref.generation or current.checksum_sha256 != ref.checksum_sha256:
            return False
        # re-read bytes: a truncated/corrupted file must never be treated as archived
        data = await asyncio.to_thread(self._path(ref.bucket, ref.key).read_bytes)
        return content_hash(data) == ref.checksum_sha256 and len(data) == ref.size_bytes

    async def delete(self, bucket: str, key: str) -> None:
        self._check()
        path = self._path(bucket, key)
        for p in (path, path.with_name(path.name + ".meta.json")):
            if p.exists():
                p.unlink()

    async def list(self, bucket: str, prefix: str) -> AsyncIterator[BlobRef]:
        self._check()
        base = self.root / bucket
        if not base.exists():
            return
        for meta_path in sorted(base.rglob("*.meta.json")):
            key = str(meta_path.relative_to(base))[: -len(".meta.json")]
            if key.startswith(prefix):
                yield await self.head(bucket, key)

    async def ping(self) -> bool:
        return self.available and self.root.exists()
