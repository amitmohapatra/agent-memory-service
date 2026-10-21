"""stress_accumulated: production memory that is never cleared.

Customer testers re-run the same scripted conversations day after day for the same user,
without clearing memory, each run with fresh random values (a new DC for the brand, a new
buyer for the region, a new unlock phrase for the standing rule). After N such runs the store
holds N stale versions of every fact. This set reproduces that: five scripts (retail
replenishment, IT operations, finance, field service, personal assistant), each written N times
(``--prior-runs``, default 6) over earlier days and threads, then a final run, all in five
languages; the final run's questions are asked in its own chat and in a new chat, and the
pushed context is scored on the final run's values only (``judge.py``).

Covered per script: single facts, 2- and 3-hop chains, explicit replacements worded
differently per script, recency (in-run corrections and the latest of a stream of events),
never-stated questions next to related facts, standing rules, format rules and preferences.

    benchmark/capabilities/stress/run_isolated.sh <name> <service checkout> <out dir>

``build.py`` draws the runs, ``judge.py`` scores one context, ``run.py`` drives the service,
``report.py`` summarises one run or compares two.
"""
