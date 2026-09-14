"""In-memory relationship-based AuthorizationProvider.

Evaluates the OpenFGA model (``model.py``) with Zanzibar check semantics over an in-process
tuple set. Used for tests, single-process development and as the reference implementation
the OpenFGA adapter is contract-tested against. Never allowed in ``prod``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from memory_service.adapters.authz.model import MODEL, TypeDef
from memory_service.ports.authorization import AccessCheck, RelationTuple
from memory_service.ports.models import ProviderInfo


def _split(obj: str) -> tuple[str, str]:
    type_name, _, ident = obj.partition(":")
    return type_name, ident


class MemoryAuthorizationProvider:
    info = ProviderInfo(
        name="memory-rebac", license="Apache-2.0", origin="internal", locality="local"
    )

    def __init__(self, model: dict[str, TypeDef] | None = None, *, max_listed_objects: int = 2000):
        self.model = model or MODEL
        # object -> relation -> set(users)   where user is "user:u1" | "agent:a1" | "group:g#member"
        self._tuples: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self._objects_by_type: dict[str, set[str]] = defaultdict(set)
        self.max_listed_objects = max_listed_objects
        self.check_calls = 0

    # -- tuple management -----------------------------------------------------
    async def write(
        self, add: Sequence[RelationTuple], delete: Sequence[RelationTuple] = ()
    ) -> None:
        for t in delete:
            self._tuples[t.object][t.relation].discard(t.user)
        for t in add:
            self._validate(t)
            self._tuples[t.object][t.relation].add(t.user)
            self._objects_by_type[_split(t.object)[0]].add(t.object)

    def _validate(self, t: RelationTuple) -> None:
        obj_type, _ = _split(t.object)
        type_def = self.model.get(obj_type)
        if type_def is None:
            raise ValueError(f"unknown object type in tuple: {t.object}")
        rel = type_def.relations.get(t.relation)
        if rel is None:
            raise ValueError(f"unknown relation {t.relation} on {obj_type}")
        user_type = t.user.split(":", 1)[0]
        user_ref = f"{user_type}#{t.user.split('#', 1)[1]}" if "#" in t.user else user_type
        if user_ref not in rel.direct:
            raise ValueError(
                f"{user_ref} may not be directly assigned {t.relation} on {obj_type}; "
                f"allowed: {rel.direct}"
            )

    # -- check ----------------------------------------------------------------
    async def check(self, check: AccessCheck) -> bool:
        self.check_calls += 1
        return self._check(check.user, check.relation, check.object, frozenset())

    async def batch_check(self, checks: Sequence[AccessCheck]) -> list[bool]:
        return [await self.check(c) for c in checks]

    def _check(
        self, user: str, relation: str, obj: str, visited: frozenset[tuple[str, str]]
    ) -> bool:
        key = (relation, obj)
        if key in visited:
            return False
        visited = visited | {key}
        obj_type, _ = _split(obj)
        type_def = self.model.get(obj_type)
        if type_def is None:
            return False
        rel = type_def.relations.get(relation)
        if rel is None:
            return False
        # 1. direct tuples (including usersets like group:g#member)
        direct_users = self._tuples.get(obj, {}).get(relation, set())
        if user in direct_users:
            return True
        for member in direct_users:
            if "#" in member:
                userset_obj, userset_rel = member.split("#", 1)
                if self._check(user, userset_rel, userset_obj, visited):
                    return True
        # 2. computed usersets
        for other in rel.computed:
            if self._check(user, other, obj, visited):
                return True
        # 3. tuple-to-userset
        for tupleset_rel, target_rel in rel.ttu:
            for target in self._tuples.get(obj, {}).get(tupleset_rel, set()):
                if self._check(user, target_rel, target, visited):
                    return True
        return False

    async def list_objects(self, user: str, relation: str, object_type: str) -> list[str]:
        out = [
            obj
            for obj in sorted(self._objects_by_type.get(object_type, ()))
            if self._check(user, relation, obj, frozenset())
        ]
        return out[: self.max_listed_objects + 1]

    async def ping(self) -> bool:
        return True

    def dump(self) -> list[RelationTuple]:
        return [
            RelationTuple(user=u, relation=r, object=o)
            for o, rels in self._tuples.items()
            for r, users in rels.items()
            for u in sorted(users)
        ]
