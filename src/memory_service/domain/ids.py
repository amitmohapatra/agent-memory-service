"""Identifier generation and validation.

IDs are prefixed ULIDs: time-sortable (good B-tree locality), globally unique, and
self-describing (``thr_...`` is obviously a thread). Client-supplied IDs are accepted
as long as they satisfy :data:`ID_PATTERN` so upstream systems can keep their own IDs.
"""

from __future__ import annotations

import hashlib
import re
from typing import Final

from ulid import ULID

ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,199}$")

PREFIXES: Final[dict[str, str]] = {
    "thread": "thr",
    "session": "ses",
    "turn": "trn",
    "message": "msg",
    "message_version": "msv",
    "attachment": "att",
    "agent_run": "run",
    "observation": "obs",
    "memory": "mem",
    "evidence": "evd",
    "document": "doc",
    "document_version": "dcv",
    "node": "nod",
    "chunk": "chk",
    "entity": "ent",
    "relation": "rel",
    "job": "job",
    "segment": "seg",
    "manifest": "man",
    "request": "req",
    "work": "wrk",
    "task": "tsk",
    "import": "imp",
    "summary": "sum",
}


def new_id(kind: str) -> str:
    """Return a new prefixed ULID for ``kind`` (e.g. ``new_id("thread") -> "thr_01J..."``)."""
    try:
        prefix = PREFIXES[kind]
    except KeyError as exc:  # pragma: no cover - programming error
        raise ValueError(f"unknown id kind: {kind}") from exc
    return f"{prefix}_{ULID()}"


def is_valid_id(value: str) -> bool:
    return bool(ID_PATTERN.match(value))


def content_hash(data: bytes | str) -> str:
    """Stable SHA-256 hex digest used for exact-hash dedup, checksums and idempotency."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def stable_key(*parts: str) -> str:
    """Deterministic short key from parts (used for cache keys and idempotency defaults)."""
    joined = "\x1f".join(parts)
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=16).hexdigest()
