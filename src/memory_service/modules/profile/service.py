"""Pinned profile blocks: read, replace, edit, and the ``user`` block kept from memory.

Reads and writes are one indexed lookup. Every write bumps the revision the pushed context of
that scope depends on (USER, AGENT, or TENANT for a workspace block), so a cached bundle that
showed the old text stops being served.

The ``profile.refresh`` job keeps the ``user`` block from the user's USER and PREFERENCE
memories: rewritten while the job owns it (``learned``; by the tenant's model when one is
available, a line per memory otherwise), and only ever appended to once a person or an agent
has edited it (``edited``) - with the facts learned since the edit - so an edit is never
lost or undone.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Final

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import MemoryType
from memory_service.domain.errors import Conflict, NotFound
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.profile import (
    PROFILE_BLOCK_MAX_CHARS,
    USER_BLOCK,
    ProfileBlock,
    block_scope,
    profile_scopes,
)
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.authz.service import AuthorizationService
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.tenancy.gate import require_workspace_member
from memory_service.observability.logging import get_logger
from memory_service.ports.credentials import ModelIdentity
from memory_service.ports.tasks import JobSpec, Queue
from memory_service.ports.uow import UnitOfWork, UnitOfWorkFactory

log = get_logger(__name__)

TASK_PROFILE_REFRESH: Final = "profile.refresh"
#: The memory kinds the ``user`` block is kept from.
PROFILE_MEMORY_TYPES: Final = (MemoryType.USER, MemoryType.PREFERENCE)
#: The user's newest memories of those kinds one refresh reads.
PROFILE_MEMORIES_MAX: Final = 50

PROFILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"profile": {"type": "string"}},
    "required": ["profile"],
    "additionalProperties": False,
}
_PROFILE_SYSTEM = (
    "You maintain the profile an assistant keeps of the person it works for. From the facts "
    "and preferences given, write a concise profile: one line per fact, as 'name: value' "
    "when the fact has a name. Merge duplicates, keep the newest when two disagree, and add "
    "nothing that is not given. Write in the language of the facts. At most {max_chars} "
    'characters. Return JSON only: {{"profile": "..."}}.'
)

_REVISION_OF_LEVEL: Final = {
    "user": RevisionKind.USER,
    "agent": RevisionKind.AGENT,
    "workspace": RevisionKind.TENANT,
}


def _level(block: str) -> str:
    return block.split(".", 1)[0]


def template(memories: Sequence[CanonicalMemory]) -> str:
    """A line per memory, newest first, bounded: the ``user`` block without a model."""
    lines = list(dict.fromkeys(f"- {' '.join(m.content.split())}" for m in memories))
    return _bounded(lines)


def _bounded(lines: Sequence[str]) -> str:
    out: list[str] = []
    used = 0
    for line in lines:
        if used + len(line) + 1 > PROFILE_BLOCK_MAX_CHARS:
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)


def appended(text: str, memories: Sequence[CanonicalMemory], since: datetime) -> str:
    """An edited block with the facts learned after the edit, that it does not already
    state, added below it: what the edit changed is never put back."""
    known = text.casefold()
    new = [
        f"- {' '.join(m.content.split())}"
        for m in memories
        if m.created_at > since and " ".join(m.content.split()).casefold() not in known
    ]
    return _bounded([*text.splitlines(), *dict.fromkeys(new)]) if new else text


class ProfileService:
    def __init__(
        self, uow_factory: UnitOfWorkFactory, authz: AuthorizationService, assist: LLMAssist
    ) -> None:
        self.uow_factory = uow_factory
        self.authz = authz
        self.assist = assist

    async def blocks(self, uow: UnitOfWork, ctx: MemoryExecutionContext) -> list[ProfileBlock]:
        """Every block of the caller's user, agent and workspace."""
        return await uow.profiles.blocks(ctx.tenant_id, list(profile_scopes(ctx).values()))

    async def put(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, block: str, text: str
    ) -> ProfileBlock:
        scope = await self._writable(uow, ctx, block)
        return await self._write(uow, ctx, block, scope, text)

    async def edit(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, block: str, old: str, new: str
    ) -> ProfileBlock:
        """Replace ``old`` with ``new`` once; 409 when ``old`` is not in the block (it
        changed since it was read), 404 when there is no block."""
        scope = await self._writable(uow, ctx, block)
        current = await uow.profiles.get(ctx.tenant_id, scope, block)
        if current is None:
            raise NotFound(f"no {block} block")
        if old not in current.text:
            raise Conflict(
                "the text to replace is not in the block: read it again",
                details={"block": block, "version": current.version},
            )
        return await self._write(uow, ctx, block, scope, current.text.replace(old, new, 1))

    async def _writable(self, uow: UnitOfWork, ctx: MemoryExecutionContext, block: str) -> str:
        scope = block_scope(block, ctx)
        if _level(block) == "workspace":
            await require_workspace_member(uow, self.authz, ctx, team_only=False)
        await uow.serialize(f"profile:{ctx.tenant_id}:{scope}:{block}")
        return scope

    async def _write(
        self, uow: UnitOfWork, ctx: MemoryExecutionContext, block: str, scope: str, text: str
    ) -> ProfileBlock:
        stored = await uow.profiles.put(
            ProfileBlock(
                tenant_id=ctx.tenant_id, scope_key=scope, block=block, text=text, source="edited"
            )
        )
        await _bump(uow, ctx.tenant_id, block, scope)
        return stored

    # ------------------------------------------------------------------ the profile job
    @staticmethod
    async def enqueue_refresh(uow: UnitOfWork, tenant_id: str, user_id: str) -> None:
        await uow.enqueue(
            JobSpec(
                task_name=TASK_PROFILE_REFRESH,
                queue=Queue.SUMMARY,
                payload={"tenant_id": tenant_id, "user_id": user_id},
                tenant_id=tenant_id,
            )
        )

    async def refresh_user(self, tenant_id: str, user_id: str) -> ProfileBlock | None:
        """Keep the ``user`` block from the user's own USER and PREFERENCE memories."""
        scope = f"user:{user_id}"
        async with self.uow_factory() as uow:
            memories = await uow.memories.about_user(
                tenant_id,
                user_id,
                memory_types=[t.value for t in PROFILE_MEMORY_TYPES],
                limit=PROFILE_MEMORIES_MAX,
            )
            current = await uow.profiles.get(tenant_id, scope, USER_BLOCK)
        if not memories:
            return current
        text = await self._compose(tenant_id, user_id, memories, current)
        if current is not None and text == current.text:
            return current
        async with self.uow_factory() as uow:
            await uow.serialize(f"profile:{tenant_id}:{scope}:{USER_BLOCK}")
            latest = await uow.profiles.get(tenant_id, scope, USER_BLOCK)
            if latest is not None and current is not None and latest.version != current.version:
                return latest  # edited meanwhile: the next refresh appends to the edit
            stored = await uow.profiles.put(
                ProfileBlock(
                    tenant_id=tenant_id,
                    scope_key=scope,
                    block=USER_BLOCK,
                    text=text,
                    source=current.source if current is not None else "learned",
                )
            )
            await uow.revisions.bump(tenant_id, RevisionKind.USER, user_id)
            await uow.commit()
        return stored

    async def _compose(
        self,
        tenant_id: str,
        user_id: str,
        memories: Sequence[CanonicalMemory],
        current: ProfileBlock | None,
    ) -> str:
        if current is not None and current.source == "edited":
            return appended(current.text, memories, current.updated_at)
        async with self.assist.bound(ModelIdentity(tenant_id, f"user:{user_id}")):
            result = await self.assist.structured(
                "summaries",
                system=_PROFILE_SYSTEM.format(max_chars=PROFILE_BLOCK_MAX_CHARS // 2),
                user="\n".join(f"- {m.memory_type.value.lower()}: {m.content}" for m in memories),
                schema=PROFILE_SCHEMA,
                max_tokens=700,
            )
        written = str((result or {}).get("profile", "")).strip()
        if written and len(written) <= PROFILE_BLOCK_MAX_CHARS:
            return written
        return template(memories)


async def _bump(uow: UnitOfWork, tenant_id: str, block: str, scope: str) -> None:
    kind = _REVISION_OF_LEVEL[_level(block)]
    identifier = scope.split(":", 1)[1] if kind is RevisionKind.USER else ""
    if kind is RevisionKind.AGENT:
        identifier = scope.rsplit("/", 1)[-1].split(":", 1)[-1]
    await uow.revisions.bump(tenant_id, kind, identifier)
