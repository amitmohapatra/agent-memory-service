"""Python mirror of ``deploy/openfga/model.fga``.

The in-memory provider evaluates this model with Zanzibar semantics so tests and
single-process dev behave exactly like OpenFGA. A contract test parses the ``.fga`` file and
asserts both definitions agree (types, relations, direct user types, computed usersets and
tuple-to-userset rewrites) so they cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Relation:
    """One relation definition: ``this`` (direct types) OR computed OR tuple-to-userset."""

    direct: tuple[str, ...] = ()  # e.g. ("user", "group#member")
    computed: tuple[str, ...] = ()  # e.g. ("owner", "participant")
    ttu: tuple[tuple[str, str], ...] = ()  # (tupleset relation, computed relation on target)


@dataclass(frozen=True)
class TypeDef:
    name: str
    relations: dict[str, Relation] = field(default_factory=dict)


MODEL: dict[str, TypeDef] = {
    "user": TypeDef("user"),
    "agent": TypeDef("agent", {"operator": Relation(direct=("user",))}),
    "tenant": TypeDef(
        "tenant",
        {
            "admin": Relation(direct=("user",)),
            "member": Relation(direct=("user", "group#member"), computed=("admin",)),
        },
    ),
    "group": TypeDef(
        "group",
        {"tenant": Relation(direct=("tenant",)), "member": Relation(direct=("user",))},
    ),
    "workspace": TypeDef(
        "workspace",
        {
            "tenant": Relation(direct=("tenant",)),
            "admin": Relation(direct=("user",), ttu=(("tenant", "admin"),)),
            "member": Relation(direct=("user", "group#member"), computed=("admin",)),
            "viewer": Relation(direct=("user", "group#member"), computed=("member",)),
        },
    ),
    "thread": TypeDef(
        "thread",
        {
            "tenant": Relation(direct=("tenant",)),
            "workspace": Relation(direct=("workspace",)),
            "owner": Relation(direct=("user",)),
            "participant": Relation(direct=("user", "agent", "group#member")),
            "viewer": Relation(
                direct=("user", "group#member"),
                computed=("owner", "participant"),
                ttu=(("workspace", "admin"),),
            ),
            "can_read": Relation(computed=("viewer",), ttu=(("tenant", "admin"),)),
            "can_write": Relation(computed=("owner", "participant"), ttu=(("workspace", "admin"),)),
        },
    ),
    "document": TypeDef(
        "document",
        {
            "tenant": Relation(direct=("tenant",)),
            "workspace": Relation(direct=("workspace",)),
            "thread": Relation(direct=("thread",)),
            "owner": Relation(direct=("user",)),
            "viewer": Relation(direct=("user", "group#member", "agent")),
            "can_read": Relation(
                computed=("owner", "viewer"),
                ttu=(("thread", "can_read"), ("workspace", "member"), ("tenant", "admin")),
            ),
            "can_write": Relation(computed=("owner",), ttu=(("workspace", "admin"),)),
        },
    ),
    "memory": TypeDef(
        "memory",
        {
            "tenant": Relation(direct=("tenant",)),
            "owner": Relation(direct=("user", "agent")),
            "viewer": Relation(direct=("user", "group#member", "agent")),
            "can_read": Relation(computed=("owner", "viewer"), ttu=(("tenant", "admin"),)),
            "can_delete": Relation(computed=("owner",), ttu=(("tenant", "admin"),)),
        },
    ),
}


def parse_fga(text: str) -> dict[str, TypeDef]:
    """Minimal parser for the subset of the OpenFGA DSL used in ``model.fga``.

    Supports: ``define rel: [types] or computed or rel from tupleset``.
    """
    types: dict[str, TypeDef] = {}
    current: TypeDef | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line in ("model", "schema 1.1", "relations"):
            continue
        if line.startswith("type "):
            current = TypeDef(line.split()[1])
            types[current.name] = current
            continue
        if line.startswith("define ") and current is not None:
            name, _, rhs = line[len("define ") :].partition(":")
            direct: list[str] = []
            computed: list[str] = []
            ttu: list[tuple[str, str]] = []
            for part in (p.strip() for p in rhs.split(" or ")):
                if part.startswith("["):
                    direct.extend(t.strip() for t in part.strip("[]").split(","))
                elif " from " in part:
                    rel, _, tupleset = part.partition(" from ")
                    ttu.append((tupleset.strip(), rel.strip()))
                elif part:
                    computed.append(part)
            current.relations[name.strip()] = Relation(
                direct=tuple(direct), computed=tuple(computed), ttu=tuple(ttu)
            )
    return types
