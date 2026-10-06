"""Where an approved skill draft is published: the team's skills store.

An adapter writes one Agent Skill (a ``SKILL.md``: name, description, body, metadata) where
agents already load skills from - a folder in the Agent Skills layout (``SKILLS_DIR``) or the
Bifrost gateway's skills repository - and answers the version it published. It never takes
over a skill this tenant did not publish (``Conflict``).
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from memory_service.domain.errors import Conflict

#: The metadata keys a published skill carries: who published it and from what.
SOURCE_KEY = "source"
SOURCE = "trellis-memory"
TENANT_KEY = "trellis_tenant"
PROCEDURE_KEY = "trellis_procedure"


class SkillContent(BaseModel):
    """One skill as it is published: the ``SKILL.md`` front matter fields and its body."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    body: str
    metadata: dict[str, str]


@runtime_checkable
class SkillStore(Protocol):
    #: what the decision records as the destination: ``skills_dir`` or ``bifrost``
    destination: str

    async def publish(self, skill: SkillContent, *, tenant_id: str) -> str:
        """Publish ``skill`` as its next version (``1.0.0`` first, then the next minor) and
        return that version. ``Conflict`` when a skill of that name exists that ``tenant_id``
        did not publish."""
        ...


#: An Agent Skills name: lowercase letters, digits and single hyphens, at most 64 characters.
NAME_MAX = 64
_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def valid_name(name: str) -> bool:
    return len(name) <= NAME_MAX and bool(_NAME.match(name))


def next_version(current: str | None) -> str:
    """``1.0.0`` for a new skill, else the next minor of ``current`` (``1.2.0`` -> ``1.3.0``).
    ``Conflict`` for a version that is not ``major.minor.patch``."""
    if current is None:
        return "1.0.0"
    parts = current.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise Conflict(f"the published version {current!r} is not major.minor.patch")
    return f"{parts[0]}.{int(parts[1]) + 1}.0"
