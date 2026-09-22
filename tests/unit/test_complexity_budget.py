"""A ratchet on complexity, not a style opinion.

Measured at the time of writing: 1,332 functions, mean cyclomatic complexity 4.1 — the code
base is not generally complex. The tail is: 43 functions above 15, the worst being
``_sentence_facts`` at 109 across 383 lines.

These budgets are set at *today's* worst, so nothing may get worse and every improvement
tightens them. Lowering a number here is the point; raising one needs a reason in the commit
message. Complexity is counted the way cyclomatic complexity normally is — one per branch,
loop, handler, comprehension and boolean operand — with no extra dependency to install.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "memory_service"

#: the current worst function, by cyclomatic complexity
MAX_COMPLEXITY = 109
#: the current longest function, in lines
MAX_FUNCTION_LINES = 383
#: how many functions may exceed a complexity of 15
MAX_OVER_15 = 43
#: the mean must not drift upwards. 4.2 -> 4.25 on 2026-09-22: the Phase-1 deletions removed
#: ~140 trivial functions (observer, served-model adapters, model server), which lifts the
#: mean of what remains without any function getting more complex.
MAX_MEAN_COMPLEXITY = 4.25

_BRANCHING = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.AsyncWith,
    ast.Assert,
    ast.IfExp,
    ast.comprehension,
    ast.match_case,
)


def _complexity(fn: ast.AST) -> int:
    score = 1
    for node in ast.walk(fn):
        if isinstance(node, _BRANCHING):
            score += 1
        elif isinstance(node, ast.BoolOp):
            score += len(node.values) - 1
    return score


def _functions() -> list[tuple[int, int, str]]:
    found: list[tuple[int, int, str]] = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                length = (node.end_lineno or node.lineno) - node.lineno
                found.append(
                    (
                        _complexity(node),
                        length,
                        f"{path.relative_to(SRC)}:{node.lineno} {node.name}",
                    )
                )
    return found


def test_no_function_is_more_complex_than_the_worst_one_today() -> None:
    worst = sorted(_functions(), reverse=True)[:3]
    assert worst[0][0] <= MAX_COMPLEXITY, (
        f"complexity budget exceeded (limit {MAX_COMPLEXITY}):\n  "
        + "\n  ".join(f"cc={c} {where}" for c, _, where in worst)
    )


def test_no_function_is_longer_than_the_longest_one_today() -> None:
    longest = sorted(_functions(), key=lambda row: -row[1])[:3]
    assert longest[0][1] <= MAX_FUNCTION_LINES, (
        f"length budget exceeded (limit {MAX_FUNCTION_LINES}):\n  "
        + "\n  ".join(f"len={n} {where}" for _, n, where in longest)
    )


def test_the_complex_tail_does_not_grow() -> None:
    over = [row for row in _functions() if row[0] > 15]
    assert len(over) <= MAX_OVER_15, (
        f"{len(over)} functions above complexity 15 (budget {MAX_OVER_15}); "
        "the newest ones are:\n  "
        + "\n  ".join(f"cc={c} {where}" for c, _, where in sorted(over, reverse=True)[:5])
    )


def test_the_average_function_stays_simple() -> None:
    rows = _functions()
    mean = sum(row[0] for row in rows) / len(rows)
    assert mean <= MAX_MEAN_COMPLEXITY, (
        f"mean complexity {mean:.2f} over {len(rows)} functions exceeds {MAX_MEAN_COMPLEXITY}"
    )
