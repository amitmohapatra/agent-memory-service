"""Check the Markdown in this repository: links resolve, and code a reader may copy is real.

    uv run python scripts/check_docs.py          # every check; exit 1 on any finding
    uv run python scripts/check_docs.py --links  # relative links and anchors only

Links. Every relative link in every Markdown file the repository holds must name a file or
directory that exists, and a ``#fragment`` must name a heading in the target (GitHub's
anchor rules). External links (``http(s)://``, ``mailto:``) are not fetched.

Snippets. Every ``python`` block in the documentation must parse (top-level ``await`` is
allowed, as in a notebook). Every method it calls on an object must exist in the SDK
(``trellis.memory``) unless it is a name of the reader's own (``my_agent.run``) listed in
``FOREIGN_CALLS``, and every keyword it passes to an SDK method must be one that method
takes. Every ``from trellis.memory import X`` must name something the package exports. A
block whose first line is ``# example: examples/NN_name.py`` is an excerpt of that example:
the file must exist, and every SDK method the excerpt calls must be called in the file too
(the example runs in ``make examples``). Every ``make <target>`` in a ``bash`` block must be
a Makefile target, and every ``examples/...`` or ``scripts/...`` path it runs must exist.
"""

from __future__ import annotations

import ast
import inspect
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parents[1]

#: generated or third-party Markdown, and test data: not this repository's documentation
EXCLUDED = ("vendor/", ".venv/", "tests/fixtures/", "models/", ".bifrost-sdk/", "benchmark/data/")

#: methods a snippet calls on the reader's own objects, never on the SDK
FOREIGN_CALLS = frozenset({"run", "sleep"})

FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>```+|~~~+)(?P<info>[^\n]*)$")
LINK = re.compile(
    r"(?<!\!)\[(?:[^\]\[]|\[[^\]]*\])*\]\((?P<target><[^>]+>|[^)\s]+)(?:\s+\"[^\"]*\")?\)"
)
IMAGE = re.compile(r"!\[[^\]]*\]\((?P<target>[^)\s]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(?P<text>.+?)\s*#*\s*$")
EXAMPLE_MARK = re.compile(r"^#\s*example:\s*(?P<path>\S+)")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


@dataclass(frozen=True)
class Block:
    path: str
    line: int
    lang: str
    code: str


# --------------------------------------------------------------------------- files


def markdown_files() -> list[Path]:
    """Tracked Markdown, plus new files not yet added (so a check before a commit sees them)."""
    command = ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"]
    out = subprocess.run(  # noqa: S603 - a fixed argument list, git from PATH as make uses it
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return sorted(REPO / p for p in set(out) if not p.startswith(EXCLUDED) and (REPO / p).is_file())


def split(text: str) -> tuple[list[tuple[int, str]], list[Block]]:
    """The prose lines (numbered, inline code blanked) and the fenced blocks of a file."""
    prose: list[tuple[int, str]] = []
    blocks: list[Block] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = FENCE.match(lines[i])
        if match:
            fence, info, start = match.group("fence"), match.group("info").strip(), i
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(fence):
                body.append(lines[i])
                i += 1
            indent = len(match.group("indent"))
            code = "\n".join(line[indent:] if line[:indent].isspace() else line for line in body)
            blocks.append(Block("", start + 1, (info.split() or [""])[0].lower(), code + "\n"))
            i += 1
            continue
        prose.append((i + 1, re.sub(r"`[^`]*`", "", lines[i])))
        i += 1
    return prose, blocks


# --------------------------------------------------------------------------- links


def slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # links keep their text
    text = re.sub(r"<[^>]+>", "", text)  # inline HTML
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)  # punctuation goes; letters, digits, _ and - stay
    return text.replace(" ", "-")


def anchors(path: Path, cache: dict[Path, set[str]]) -> set[str]:
    if path not in cache:
        prose, _ = split(path.read_text(encoding="utf-8"))
        seen: dict[str, int] = {}
        found: set[str] = set()
        raw = path.read_text(encoding="utf-8").splitlines()
        for number, _line in prose:
            match = HEADING.match(raw[number - 1])
            if not match:
                continue
            base = slug(match.group("text"))
            count = seen.get(base, 0)
            seen[base] = count + 1
            found.add(base if count == 0 else f"{base}-{count}")
        found |= set(re.findall(r"<a\s+(?:name|id)=\"([^\"]+)\"", path.read_text(encoding="utf-8")))
        cache[path] = found
    return cache[path]


def check_links(files: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    cache: dict[Path, set[str]] = {}
    for path in files:
        rel = path.relative_to(REPO).as_posix()
        prose, _ = split(path.read_text(encoding="utf-8"))
        for number, line in prose:
            targets = [m.group("target").strip("<>") for m in LINK.finditer(line)]
            targets += [m.group("target") for m in IMAGE.finditer(line)]
            for target in targets:
                if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE):
                    continue  # http:, https:, mailto:, ...
                file_part, _, fragment = target.partition("#")
                resolved = (path.parent / unquote(file_part)).resolve() if file_part else path
                if file_part and not resolved.exists():
                    findings.append(Finding(rel, number, f"broken link: {target}"))
                    continue
                anchored = fragment and resolved.suffix == ".md" and resolved.is_file()
                if anchored and unquote(fragment).lower() not in anchors(resolved, cache):
                    findings.append(Finding(rel, number, f"no heading for anchor: {target}"))
    return findings


# --------------------------------------------------------------------------- snippets


def sdk_surface() -> tuple[set[str], dict[str, set[str] | None], set[str]]:
    """Every attribute the SDK's classes have, the keywords each method name accepts
    (``None``: it takes ``**kwargs``), and the names ``trellis.memory`` exports."""
    import trellis.memory as package
    from trellis.memory import admin, advanced, client, models

    attributes: set[str] = set()
    keywords: dict[str, set[str] | None] = {}
    for module in (client, advanced, admin, models, package):
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not cls.__module__.startswith("trellis.memory"):
                continue
            attributes |= {n for n in dir(cls) if not n.startswith("__")}
            attributes |= set(getattr(cls, "model_fields", None) or {})
            for name, fn in inspect.getmembers(cls, inspect.isfunction):
                params = inspect.signature(fn).parameters.values()
                if any(p.kind is p.VAR_KEYWORD for p in params):
                    keywords[name] = None
                elif keywords.get(name, set()) is not None:
                    keywords.setdefault(name, set()).update(p.name for p in params)
        source = Path(module.__file__ or "").read_text(encoding="utf-8")
        attributes |= set(re.findall(r"self\.(\w+)\s*[:=]", source))
    return attributes, keywords, set(dir(package))


def calls(tree: ast.AST) -> Iterator[ast.Call]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            yield node


def check_python(block: Block, surface: tuple[set[str], dict, set[str]]) -> list[Finding]:
    attributes, keywords, exported = surface
    try:
        tree = compile(
            block.code,
            block.path,
            "exec",
            flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            dont_inherit=True,
        )
    except SyntaxError as exc:
        return [Finding(block.path, block.line + (exc.lineno or 0), f"python: {exc.msg}")]
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "trellis.memory":
            for alias in node.names:
                if alias.name not in exported:
                    findings.append(
                        Finding(
                            block.path,
                            block.line + node.lineno,
                            f"trellis.memory has no {alias.name}",
                        )
                    )
    for call in calls(tree):
        name = call.func.attr  # type: ignore[union-attr]
        where = block.line + call.lineno
        if name not in attributes and name not in FOREIGN_CALLS:
            findings.append(Finding(block.path, where, f"no SDK method {name}()"))
            continue
        accepted = keywords.get(name, None)
        if accepted is None:
            continue
        for kw in call.keywords:
            if kw.arg is not None and kw.arg not in accepted:
                findings.append(Finding(block.path, where, f"{name}() takes no keyword {kw.arg!r}"))
    mark = EXAMPLE_MARK.match(block.code.lstrip().splitlines()[0]) if block.code.strip() else None
    if mark:
        example = REPO / mark.group("path")
        if not example.is_file():
            findings.append(
                Finding(block.path, block.line, f"no example file {mark.group('path')}")
            )
        else:
            used = {c.func.attr for c in calls(ast.parse(example.read_text(encoding="utf-8")))}  # type: ignore[union-attr]
            missing = {c.func.attr for c in calls(tree)} - used - FOREIGN_CALLS  # type: ignore[union-attr]
            if missing:
                findings.append(
                    Finding(
                        block.path,
                        block.line,
                        f"{mark.group('path')} never calls {sorted(missing)}",
                    )
                )
    return findings


def check_bash(block: Block, targets: set[str]) -> list[Finding]:
    findings: list[Finding] = []
    for offset, line in enumerate(block.code.splitlines(), start=1):
        command = line.split("#", 1)[0]
        for piece in re.split(r"&&|\|\||;|\|", command):
            words = piece.split()
            if not words or words[0] != "make":
                continue
            for word in words[1:]:
                if word.startswith("-") or "=" in word or not re.fullmatch(r"[\w.-]+", word):
                    continue  # a flag, a variable, or past the targets
                if word not in targets:
                    findings.append(
                        Finding(block.path, block.line + offset, f"no make target {word!r}")
                    )
        for path in re.findall(r"\b((?:examples|scripts)/[\w./-]+\.(?:py|sh))", command):
            if not (REPO / path).is_file():
                findings.append(Finding(block.path, block.line + offset, f"no file {path}"))
    return findings


def make_targets() -> set[str]:
    text = (REPO / "Makefile").read_text(encoding="utf-8")
    return set(re.findall(r"^([A-Za-z0-9_.-]+):", text, re.MULTILINE))


def check_snippets(files: list[Path]) -> list[Finding]:
    surface = sdk_surface()
    targets = make_targets()
    findings: list[Finding] = []
    for path in files:
        rel = path.relative_to(REPO).as_posix()
        _, blocks = split(path.read_text(encoding="utf-8"))
        for found in blocks:
            block = Block(rel, found.line, found.lang, found.code)
            if block.lang in ("python", "py"):
                findings += check_python(block, surface)
            elif block.lang in ("bash", "sh", "shell", "console"):
                findings += check_bash(block, targets)
    return findings


def main(argv: list[str]) -> int:
    files = markdown_files()
    findings = check_links(files)
    if "--links" not in argv:
        findings += check_snippets(files)
    for finding in findings:
        print(finding)  # noqa: T201 - the script's output
    checks = "links" if "--links" in argv else "links and snippets"
    print(f"{len(files)} Markdown files, {checks}: {len(findings)} finding(s)")  # noqa: T201
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
