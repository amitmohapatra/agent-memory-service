"""The two places an approved skill draft is published to (``ports.skills.SkillStore``).

* :class:`FolderSkills` - a folder in the Agent Skills layout (``SKILLS_DIR``):
  ``<name>/SKILL.md`` with its front matter, which the harness's ``skills_dir``, the Claude
  Agent SDK and Deep Agents read as they are. A new version replaces the file; the folder's
  own history (a git repository, a volume snapshot) is its rollback.
* :class:`BifrostSkills` - the Bifrost gateway's skills repository, through bifrost-sdk's
  ``Admin.skills``: a new immutable version, served at once; rollback is the gateway's own
  ``shift_version``.

Both refuse a name that exists and that this tenant did not publish (its metadata names
another tenant, or none), so a draft never takes over a skill a person or another tenant
owns.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from pathlib import Path

import httpx
from bifrost_sdk import BifrostError, ConflictError
from bifrost_sdk.admin import Admin

from memory_service.domain.errors import Conflict, DependencyUnavailable
from memory_service.ports.skills import (
    TENANT_KEY,
    SkillContent,
    next_version,
    valid_name,
)

SKILL_MD = "SKILL.md"
_FENCE = "---"
#: ``  key: value`` inside the front matter's ``metadata`` block (the shape this module writes)
_META_LINE = re.compile(r"^  (?P<key>[a-z_]+): (?P<value>.+)$")


def _checked(name: str) -> str:
    if not valid_name(name):
        raise Conflict(f"{name!r} is not a skill name (lowercase letters, digits, hyphens)")
    return name


def _owned(metadata: dict[str, str], name: str, tenant_id: str) -> None:
    if metadata.get(TENANT_KEY) != tenant_id:
        raise Conflict(f"a skill named {name!r} exists that this tenant did not publish")


def _scalar(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith('"'):
        try:
            return str(json.loads(raw))
        except ValueError:
            return raw
    return raw


def front_matter_metadata(text: str) -> dict[str, str]:
    """The ``metadata`` block of a ``SKILL.md`` this module wrote (other files: what of it
    reads the same way; nothing when it has none)."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FENCE:
        return {}
    found: dict[str, str] = {}
    inside = False
    for line in lines[1:]:
        if line.strip() == _FENCE:
            break
        if line.startswith("metadata:"):
            inside = True
        elif not line.startswith("  "):
            inside = False
        elif inside and (match := _META_LINE.match(line)):
            found[match["key"]] = _scalar(match["value"])
    return found


def skill_md(skill: SkillContent, version: str) -> str:
    """The ``SKILL.md``: YAML front matter (every value a JSON string, which YAML reads as
    itself), then the body."""
    meta = {**skill.metadata, "version": version}
    lines = [
        _FENCE,
        f"name: {skill.name}",
        f"description: {json.dumps(skill.description, ensure_ascii=False)}",
        "metadata:",
        *(f"  {key}: {json.dumps(value, ensure_ascii=False)}" for key, value in meta.items()),
        _FENCE,
        "",
        skill.body.strip(),
        "",
    ]
    return "\n".join(lines)


class FolderSkills:
    destination = "skills_dir"

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    async def publish(self, skill: SkillContent, *, tenant_id: str) -> str:
        try:
            return await asyncio.to_thread(self._publish, skill, tenant_id)
        except OSError as exc:  # a read-only volume, a file where the folder should be
            raise DependencyUnavailable(f"the skills folder cannot be written: {exc}") from exc

    def _publish(self, skill: SkillContent, tenant_id: str) -> str:
        folder = self.root / _checked(skill.name)
        file = folder / SKILL_MD
        current = None
        if file.exists():
            metadata = front_matter_metadata(file.read_text("utf-8", errors="replace"))
            _owned(metadata, skill.name, tenant_id)
            current = metadata.get("version")
        version = next_version(current)
        folder.mkdir(parents=True, exist_ok=True)
        # written whole or not at all: a reader never sees half a skill
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=".SKILL.", suffix=".md")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                out.write(skill_md(skill, version))
            os.replace(tmp, file)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return version


class BifrostSkills:
    destination = "bifrost"

    def __init__(self, base_url: str, *, token: str | None = None) -> None:
        self.base_url = base_url
        self.token = token

    async def publish(self, skill: SkillContent, *, tenant_id: str) -> str:
        name = _checked(skill.name)
        try:
            return await self._publish(name, skill, tenant_id)
        except ConflictError as exc:  # another publication took the version first
            raise Conflict(f"the gateway refused {name}: {exc}") from exc
        except (BifrostError, httpx.HTTPError) as exc:
            raise DependencyUnavailable(f"the gateway's skills repository: {exc}") from exc

    async def _publish(self, name: str, skill: SkillContent, tenant_id: str) -> str:
        async with Admin(self.base_url, token=self.token) as admin:
            existing = await admin.skills.find(name)
            if existing is None:
                version = next_version(None)
                await admin.skills.create(
                    name,
                    description=skill.description,
                    body=skill.body,
                    version=version,
                    metadata=skill.metadata,
                )
                return version
            _owned(existing.metadata, name, tenant_id)
            version = next_version(existing.highest_version or existing.version)
            await admin.skills.publish(
                existing.id,
                description=skill.description,
                body=skill.body,
                version=version,
                metadata=skill.metadata,
            )
            return version
