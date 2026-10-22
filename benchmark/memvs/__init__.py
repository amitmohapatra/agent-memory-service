"""Memory vs frameworks: one workload, every system's own memory, one model.

The head-to-head of ``docs/eval/memory-vs-frameworks.md``. Each memory system is driven
through its own write and read API (``systems``), in the interpreter that has its library:
the shared modules here (``model``, ``llm``, ``resources``, ``run`` and ``systems``) import
nothing beyond the standard library and ``httpx``, so the same driver runs under the
harness's environment, the extra-libraries environment and CrewAI's.

    workload  (this repository's environment) the datasets as neutral JSON
    run       (any environment) ingest, restart, query, answer: one JSONL row per question
    score     (this repository's environment) metrics with paired bootstrap intervals
"""
