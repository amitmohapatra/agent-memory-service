"""What a built bundle leaves behind for the calls that refer back to it: the handles its
prompt cites by (``m1``, ``d2``, ...) and the evidence it carried.

A prompt cites ``[m3]`` instead of a 30-character memory id. Updating, forgetting or verifying
by that handle has to resolve it to the item it named in *that* bundle, so every built bundle
records its handle map and its evidence under the caller's scope for ``RECORD_TTL_SECONDS``,
and a run records which bundle it was given last. A handle therefore resolves only for the
scope that was shown it; the item it resolves to is then authorized like any other id.

The record is not the bundle cache: that one is keyed by the revisions the bundle was built
from and stops resolving the moment anything changes, which is exactly when an agent that just
updated ``m1`` goes on to forget ``m2``.
"""

from __future__ import annotations

import contextlib
import re
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.context_bundle import ContextBundle
from memory_service.domain.errors import NotFound
from memory_service.domain.memory import unverified_representation
from memory_service.ports.cache import CacheProvider, CacheUnavailable

#: How long a bundle's handles and evidence can be referred back to: longer than a run.
RECORD_TTL_SECONDS: Final = 1800
#: ``m1``, ``d12``, ...: a handle, as opposed to a record id.
HANDLE: Final = re.compile(r"^[mfsd][1-9]\d{0,3}$")


class RecordedEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    kind: str
    text: str
    citation: str = ""


class BundleRecord(BaseModel):
    """What ``/v1/verify`` and handle resolution read back about one bundle."""

    model_config = ConfigDict(frozen=True)

    bundle_id: str
    handles: dict[str, str] = Field(default_factory=dict)
    evidence: list[RecordedEvidence] = Field(default_factory=list)
    unused: list[RecordedEvidence] = Field(default_factory=list)


def record_of(bundle: ContextBundle) -> BundleRecord:
    """The handles and the evidence of a bundle, in the order the prompt numbers them. A
    model-extracted item keeps its place but carries no text: it cannot prove itself."""
    handles = bundle.handles()
    item_of = {
        item.item_id: item
        for name in ("memories", "graph_facts", "summaries", "knowledge")
        for item in getattr(bundle, name)
    }
    return BundleRecord(
        bundle_id=bundle.bundle_id,
        handles=handles,
        evidence=[
            RecordedEvidence(
                item_id=item_id,
                kind=item_of[item_id].representation.value,
                text=""
                if unverified_representation(item_of[item_id].attributes)
                else item_of[item_id].text,
                citation=handle,
            )
            for handle, item_id in handles.items()
        ],
        unused=[
            RecordedEvidence(item_id=u.item_id, kind=u.kind, text=u.text)
            for u in bundle.evidence.unused
        ],
    )


def is_handle(ref: str) -> bool:
    return HANDLE.match(ref) is not None


class BundleRecords:
    def __init__(self, cache: CacheProvider | None) -> None:
        self.cache = cache

    @staticmethod
    def _key(ctx: MemoryExecutionContext, bundle_id: str) -> str:
        return f"ctxrec:{ctx.tenant_id}:{ctx.scope_fingerprint()}:{bundle_id}"

    @staticmethod
    def _run_key(ctx: MemoryExecutionContext) -> str:
        return f"ctxrun:{ctx.tenant_id}:{ctx.scope_fingerprint()}:{ctx.agent_run_id}"

    async def store(self, ctx: MemoryExecutionContext, record: BundleRecord) -> None:
        if self.cache is None:
            return
        with contextlib.suppress(CacheUnavailable):
            await self.cache.set(
                self._key(ctx, record.bundle_id),
                record.model_dump_json().encode(),
                ttl_seconds=RECORD_TTL_SECONDS,
            )

    async def given(self, ctx: MemoryExecutionContext, bundle_id: str) -> None:
        """The run was given this bundle: its handles are the ones the run's agent sees."""
        if self.cache is None or not ctx.agent_run_id:
            return
        with contextlib.suppress(CacheUnavailable):
            await self.cache.set(
                self._run_key(ctx), bundle_id.encode(), ttl_seconds=RECORD_TTL_SECONDS
            )

    async def load(self, ctx: MemoryExecutionContext, bundle_id: str) -> BundleRecord | None:
        if self.cache is None or not bundle_id:
            return None
        try:
            raw = await self.cache.get(self._key(ctx, bundle_id))
        except CacheUnavailable:
            return None
        return BundleRecord.model_validate_json(raw) if raw is not None else None

    async def latest(self, ctx: MemoryExecutionContext) -> str | None:
        """The bundle this run was given last, if any."""
        if self.cache is None or not ctx.agent_run_id:
            return None
        try:
            raw = await self.cache.get(self._run_key(ctx))
        except CacheUnavailable:
            return None
        return raw.decode() if raw is not None else None

    async def resolve(
        self, ctx: MemoryExecutionContext, ref: str, *, bundle_id: str | None = None
    ) -> str:
        """``ref`` itself when it is a record id; the item a handle named in ``bundle_id`` (or
        the run's latest bundle) otherwise. A handle nothing resolves is not found."""
        if not is_handle(ref):
            return ref
        bundle = bundle_id or await self.latest(ctx)
        record = await self.load(ctx, bundle) if bundle else None
        item_id = record.handles.get(ref) if record is not None else None
        if item_id is None:
            raise NotFound(
                f"handle {ref} does not resolve: it names an item of a bundle this scope was "
                "given in the last 30 minutes (pass bundle_id, or the record id)"
            )
        return item_id
