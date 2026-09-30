"""``approve_when``: when a tool call must be approved by a person before it runs.

A catalog entry's ``approve_when`` is a small boolean expression over the call's arguments;
the caller asks for approval when it is true. Without one, the tool's risk tier decides
(``irreversible`` asks, ``read`` and ``write`` do not). The memory service writes these
expressions when an approval suggestion is accepted and refuses one that does not parse; a
harness evaluates them with :func:`evaluate`. This module is the one implementation of both
sides, and of :func:`arg_shape`, the argument shape approval decisions are pooled by.

Grammar (whitespace-insensitive)::

    expr       := disjunct ("or" disjunct)*
    disjunct   := conjunct ("and" conjunct)*
    conjunct   := "not" conjunct | "(" expr ")" | "true" | "false" | comparison
    comparison := operand ("==" | "!=" | "<" | "<=" | ">" | ">=" | "in") operand
    operand    := path | "string" | number | [operand, ...]
    path       := name ("." name)*     # an argument (dotted into objects); ``shape`` is
                                        # the call's arg_shape

A comparison whose path is missing, or whose values do not compare, is false.

    >>> evaluate('amount >= 1000 and currency == "EUR"', {"amount": 1200, "currency": "EUR"})
    True
    >>> evaluate('shape == "amount:num:1e3"', {"amount": 1200})
    True
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any, Final

__all__ = ["MAX_EXPRESSION_CHARS", "arg_shape", "evaluate", "parse", "when_shape"]

#: The longest expression a catalog entry may carry.
MAX_EXPRESSION_CHARS: Final = 2000
#: The name that evaluates to the call's argument shape.
SHAPE: Final = "shape"

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<string>"(?:[^"\\]|\\.)*")
      | (?P<number>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
      | (?P<op>==|!=|<=|>=|<|>)
      | (?P<punct>[()\[\],])
      | (?P<name>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)
    )""",
    re.VERBOSE,
)
_KEYWORDS: Final = frozenset({"and", "or", "not", "in", "true", "false"})
_MISSING: Final = object()


def _value_kind(value: Any) -> str:
    """A value's kind, and for a number its order of magnitude: approving ``amount=120`` says
    little about ``amount=12000``, which is exactly what an approval rule is about."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, int | float):
        magnitude = 0 if value == 0 else math.floor(math.log10(abs(value)))
        return f"num:1e{magnitude}"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list | tuple):
        return "list"
    return "obj"


def arg_shape(args: Mapping[str, Any] | None) -> str:
    """The shape of a call's top-level arguments: each name with its value kind, sorted.
    Deterministic, value-free (no customer data), and coarse enough to pool decisions."""
    if not args:
        return ""
    return ",".join(f"{name}:{_value_kind(args[name])}" for name in sorted(args))


def when_shape(shape: str) -> str:
    """The expression true exactly for calls of this argument shape."""
    return f"{SHAPE} == {json.dumps(shape)}"


def _tokens(text: str) -> list[tuple[str, Any]]:
    if len(text) > MAX_EXPRESSION_CHARS:
        raise ValueError(f"approve_when is longer than {MAX_EXPRESSION_CHARS} characters")
    out: list[tuple[str, Any]] = []
    at = 0
    while at < len(text):
        if text[at:].strip() == "":
            break
        match = _TOKEN.match(text, at)
        if match is None:
            raise ValueError(f"approve_when: unexpected input at {at}: {text[at : at + 20]!r}")
        at = match.end()
        kind = match.lastgroup or ""
        raw = match.group(kind)
        if kind == "string":
            out.append(("lit", json.loads(raw)))
        elif kind == "number":
            out.append(("lit", float(raw) if any(c in raw for c in ".eE") else int(raw)))
        elif kind == "name" and raw in _KEYWORDS:
            out.append(("lit", raw == "true") if raw in ("true", "false") else ("kw", raw))
        else:
            out.append((kind, raw))
    return out


class _Parser:
    """Recursive descent into a small tree: ``("or"|"and", a, b)``, ``("not", a)``,
    ``("cmp", op, left, right)``, ``("lit", value)``, ``("path", "a.b")``, ``("list", [...])``."""

    def __init__(self, text: str) -> None:
        self.tokens = _tokens(text)
        self.at = 0

    def _peek(self) -> tuple[str, Any] | None:
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def _take(self, kind: str, value: Any = None) -> bool:
        token = self._peek()
        if token is not None and token[0] == kind and (value is None or token[1] == value):
            self.at += 1
            return True
        return False

    def parse(self) -> tuple[Any, ...]:
        tree = self._or()
        if self._peek() is not None:
            raise ValueError(f"approve_when: unexpected {self._peek()[1]!r}")  # type: ignore[index]
        return tree

    def _or(self) -> tuple[Any, ...]:
        left = self._and()
        while self._take("kw", "or"):
            left = ("or", left, self._and())
        return left

    def _and(self) -> tuple[Any, ...]:
        left = self._not()
        while self._take("kw", "and"):
            left = ("and", left, self._not())
        return left

    def _not(self) -> tuple[Any, ...]:
        if self._take("kw", "not"):
            return ("not", self._not())
        if self._take("punct", "("):
            inner = self._or()
            if not self._take("punct", ")"):
                raise ValueError("approve_when: missing )")
            return inner
        token = self._peek()
        if token is not None and token[0] == "lit" and isinstance(token[1], bool):
            nxt = self.tokens[self.at + 1] if self.at + 1 < len(self.tokens) else None
            if nxt is None or (nxt[0] == "kw" and nxt[1] in ("and", "or")) or nxt == ("punct", ")"):
                self.at += 1
                return ("lit", token[1])
        left = self._operand()
        token = self._peek()
        if token is not None and (token[0] == "op" or token == ("kw", "in")):
            self.at += 1
            return ("cmp", token[1], left, self._operand())
        raise ValueError("approve_when: expected a comparison")

    def _operand(self) -> tuple[Any, ...]:
        token = self._peek()
        if token is None:
            raise ValueError("approve_when: unexpected end")
        if token[0] == "lit":
            self.at += 1
            return ("lit", token[1])
        if token[0] == "name":
            self.at += 1
            return ("path", token[1])
        if self._take("punct", "["):
            items: list[tuple[Any, ...]] = []
            while not self._take("punct", "]"):
                items.append(self._operand())
                if not self._take("punct", ","):
                    if not self._take("punct", "]"):
                        raise ValueError("approve_when: expected , or ]")
                    break
            return ("list", items)
        raise ValueError(f"approve_when: unexpected {token[1]!r}")


def parse(expression: str) -> tuple[Any, ...]:
    """The expression's tree; ``ValueError`` when it is not one."""
    return _Parser(expression).parse()


def _value(node: tuple[Any, ...], args: Mapping[str, Any]) -> Any:
    kind = node[0]
    if kind == "lit":
        return node[1]
    if kind == "list":
        return [_value(item, args) for item in node[1]]
    path = str(node[1])
    if path == SHAPE:
        return arg_shape(args)
    current: Any = args
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _compare(op: str, left: Any, right: Any) -> bool:
    if left is _MISSING or right is _MISSING:
        return False
    try:
        if op == "==":
            return bool(left == right)
        if op == "!=":
            return bool(left != right)
        if op == "in":
            return bool(left in right)
        if op == "<":
            return bool(left < right)
        if op == "<=":
            return bool(left <= right)
        if op == ">":
            return bool(left > right)
        return bool(left >= right)
    except TypeError:
        return False


def _eval(node: tuple[Any, ...], args: Mapping[str, Any]) -> bool:
    kind = node[0]
    if kind == "or":
        return _eval(node[1], args) or _eval(node[2], args)
    if kind == "and":
        return _eval(node[1], args) and _eval(node[2], args)
    if kind == "not":
        return not _eval(node[1], args)
    if kind == "cmp":
        return _compare(node[1], _value(node[2], args), _value(node[3], args))
    return bool(node[1])


def evaluate(expression: str, args: Mapping[str, Any]) -> bool:
    """Whether a call with ``args`` must be approved. ``ValueError`` when the expression does
    not parse: a caller fails closed and asks."""
    return _eval(parse(expression), args)
