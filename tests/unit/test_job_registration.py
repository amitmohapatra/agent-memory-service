"""Every task that gets enqueued has a handler, and the outbox has a repair path.

Both of these were broken on a running service, and both failed silently:

* ``memory.observe`` was enqueued by the observation pipeline and registered nowhere, so
  each of those rows failed dispatch with ``KeyError: task 'memory.observe' is not
  registered`` and retried until it went dead.
* ``system.outbox_sweep`` was registered but never *scheduled*, so the handler existed and
  nothing ever called it. The outbox is the only path from a committed write to the work
  that turns it into a memory, and the fast path after commit is best-effort by design —
  the sweep is the repair. Measured before the fix: 441 undispatched rows, 255 of them
  ``memory.process_observation``, the oldest 42 minutes old. Every one of those writes was
  answered 202 and never happened.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "memory_service"


def _string_constants(path: Path) -> dict[str, str]:
    """Module-level ``NAME = "value"`` assignments."""
    tree = ast.parse(path.read_text())
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value.value, str):
                    out[target.id] = node.value.value
    return out


def _calls(path: Path, name: str) -> list[ast.Call]:
    tree = ast.parse(path.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


def _first_arg_names(calls: list[ast.Call]) -> set[str]:
    names: set[str] = set()
    for call in calls:
        if not call.args:
            continue
        first = call.args[0]
        if isinstance(first, ast.Name):
            names.add(first.id)
        elif isinstance(first, ast.Constant) and isinstance(first.value, str):
            names.add(first.value)
    return names


def _all_constants() -> dict[str, str]:
    """Every module-level string constant in the service, by name.

    Built across all modules rather than per file: a call site usually *imports* the task
    name from wherever it is defined, so resolving only within one file leaves every real
    one unresolved — which made the first version of this test fail on three tasks that
    were correctly registered.
    """
    out: dict[str, str] = {}
    for path in SRC.rglob("*.py"):
        out |= _string_constants(path)
    return out


def _enqueued_task_names() -> set[str]:
    """Task names handed to a JobSpec anywhere in the service."""
    constants = _all_constants()
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id != "JobSpec":
                continue
            for keyword in node.keywords:
                if keyword.arg != "task_name":
                    continue
                value = keyword.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    found.add(value.value)
                elif isinstance(value, ast.Name):
                    found.add(constants.get(value.id, value.id))
    return found


def test_every_enqueued_task_has_a_handler() -> None:
    registry = SRC / "modules" / "jobs" / "registry.py"
    constants = _all_constants()

    registered = {constants.get(n, n) for n in _first_arg_names(_calls(registry, "register"))}
    enqueued = _enqueued_task_names()

    missing = {
        name
        for name in enqueued
        if name not in registered and not name.startswith(("periodic.", "system."))
    }
    assert not missing, (
        f"enqueued with no handler: {sorted(missing)} — these fail dispatch with a KeyError "
        f"and retry until the outbox row goes dead"
    )


def test_the_outbox_sweep_is_scheduled_not_merely_registered() -> None:
    registry = SRC / "modules" / "jobs" / "registry.py"
    source = registry.read_text()
    constants = _all_constants()
    sweep = constants["TASK_OUTBOX_SWEEP"]

    registered = {constants.get(n, n) for n in _first_arg_names(_calls(registry, "register"))}
    assert sweep in registered, "the handler must exist"

    periodic = _first_arg_names(_calls(registry, "register_periodic"))
    assert any("outbox" in name for name in periodic), (
        "the outbox sweep is registered but never scheduled: the handler exists and nothing "
        "calls it, so a write whose fast-path dispatch was missed never becomes a memory"
    )
    # Every minute: this is the floor on how late a missed write can become a memory.
    assert 'Queue.RECONCILE, outbox_sweep, cron="* * * * *"' in source


def test_the_worker_runs_every_queue_and_keeps_long_jobs_out_of_chat_and_indexing() -> None:
    """Chat and indexing on slots of their own (``worker_pools``), which they share by
    priority, chat first; graph enrichment on slots of its own; every other queue in the
    shared pool, after chat and indexing. A queue in no pool is never worked, and a parse, an
    archive, a summary or a graph backlog never holds a slot chat or indexing needs."""
    from memory_service.ports.tasks import Queue, worker_pools

    long_jobs = {q for q in Queue if q not in (Queue.CHAT_FAST, Queue.EMBEDDING)}
    for llm in (False, True):
        pools = worker_pools(4, 2, llm=llm)
        (indexing, i), (graph, g), (chat, c), (shared, s) = pools
        assert indexing == ([Queue.EMBEDDING] if llm else [Queue.EMBEDDING, Queue.CHAT_FAST]), (
            "with a model, indexing's slots are its own: chat jobs wait on the model"
        )
        assert graph == [Queue.GRAPH] and chat == [Queue.CHAT_FAST, Queue.EMBEDDING]
        assert shared[:2] == [Queue.CHAT_FAST, Queue.EMBEDDING], "chat, then indexing, first"
        assert (i, c, s) == (1, 2, 1) and g == (2 if llm else 1)
        assert not long_jobs & set(indexing + chat), "no long job in chat's or indexing's slots"
        listed = {q for queues, _ in pools for q in queues}
        assert listed == set(Queue), "every queue is worked"
        assert sorted(q for q in shared if q in long_jobs) == sorted(long_jobs - {Queue.GRAPH})
    for cpus in range(1, 9):
        for llm in (False, True):
            embedding, graph, chat, shared = (slots for _, slots in worker_pools(cpus, 2, llm=llm))
            assert min(embedding, graph, chat, shared) >= 1, "every pool is worked"
            assert chat == max(1, cpus // 2), "half the CPUs are chat's own"
            assert embedding <= max(1, cpus // 4) and embedding <= 2
            computing = chat + embedding + shared + (0 if llm else graph)
            assert computing <= max(4, cpus + 1), "the slots that compute, about one per CPU"
            assert graph == (max(2, cpus // 2) if llm else 1), "more only to wait on a model"
