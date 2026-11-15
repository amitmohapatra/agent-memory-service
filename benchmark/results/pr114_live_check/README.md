# PR #114 live check (thinking room), 2026-11-15

Haiku (`claude-haiku-5-5`), the provider's API directly, key from the environment.

## 1. One LoCoMo conversation, every model use on

LoCoMo conversation 1 (369 turns), ingested once on main (`bdba1e2`) and once on the PR branch
(`76bcb2c` = PR head `11327ff` + main merged in + the earlier-chats test fix). Each run used its own
database. The tenant policy named every use with reads assisted, and the beliefs were
consolidated to completion (`memh_lo.py`, the M7 arm of the earlier experiment). No question was
asked, answered or scored. Counts come from the service's own log lines (`memh_lo_count.py`): an
answered call is `llm.call`, a failed one is `llm.assist.fallback` (`count_main.json`,
`count_pr.json`).

| use | main: calls / failed | PR: calls / failed |
| --- | --- | --- |
| conflict checks (`conflict_adjudication`) | 18 / 14 | 20 / 0 |
| summaries | 231 / 70 | 380 / 0 |
| turn restatement (`memory_restatement`) | 449 / 205 (175 no text, 30 invalid) | 362 / 0 |
| relation extraction | 324 / 10 | 138 / 0 |
| fact writing (`contextual_extraction`) | 362 / 0 | 362 / 0 |
| beliefs (`reflection`) | 109 / 0 | 108 / 0 |

Current beliefs after consolidation: main 119, PR 128. Logged spend of the answered calls: main
$0.3433, PR $0.4391. The failed calls on main were billed too, but the log does not record their
tokens. Upper bound: $0.08.

## 2. The short calls alone (`probe114.py`)

The service's own prompts, schemas and answer sizes, through `EndpointLLM`, on a synthetic
conversation (15 restatements, 10 conflict checks, 10 entity summaries), sent twice:
`thinking_room=0` (main's request limit) and `thinking_room=8000` (the PR).
`probe114_live.json` has the results: with no room, 9 of 15 restatements and 7 of 10 conflict
checks failed and 0 of 10 summaries did; with the PR's room, 0 of 35 failed. Billed: $0.0145.

Total spend for the check: $0.80 measured, plus at most $0.08 that was billed but not logged.
