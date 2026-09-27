# OpenRouter configuration and targeted diagnostic — 2026-09-26

The user explicitly supplied an OpenRouter credential and authorized storing and using it
in Bifrost. The native `openrouter` provider is configured at `http://localhost:8091`.
The inference route is **`openrouter/openai/gpt-4o`** through `/v1/chat/completions`.
Key name: `openrouter-user`. The secret is stored by Bifrost, not in this repository.
Readback confirmed one enabled key with successful discovery and `models: ["*"]`.
Read-only SQLite verification confirmed the provider and nonempty key in the host-mounted
`/Users/ricky/usage_data/gateway-bifrost/data/config.db`. No restart was needed.
Existing providers and application model defaults were not changed.

## Configuration compatibility

This installed gateway requires separate calls to `POST /api/providers` and
`POST /api/providers/openrouter/keys`. Embedded `keys` on provider creation were ignored.
Three initial inference checks were rejected locally with `no_key_supports_model`; none
reached OpenRouter. These are preserved in `codex_openrouter_registration_failures.json`.
The installed behavior matches the separate key route in the
[upstream provider handler](https://github.com/maximhq/bifrost/blob/dev/transports/bifrost-http/handlers/providers.go).
Do not copy the credential into scripts, documentation or benchmark artifacts.

Provider retries are zero and raw request/response return and storage flags are disabled.
Credential-free configuration verification: `codex_openrouter_configuration.json`.

## Experiment and results

Reused the two previously selected failing multi-hop questions and the exact control /
actor-topic experimental contexts from the Gemini diagnostic. Same answer prompt, native
ingestion, 100-memory / 8000-estimated-token retrieval settings. Selection occurred before
either model's answers. No gold answers went to the reader; no external judge was called.
This is a deliberately selected diagnostic, not a population accuracy estimate.

The inherited 16,384-token output allowance caused two HTTP 402 rejections: OpenRouter
said the key could afford at most 4,000 output tokens. Both are archived in
`codex_openrouter_token_limit_failures.json`. The subsequent run capped output at 1,024
tokens, used concurrency one, four requests/minute, zero retries, and stopped on failure.
All successful answers were only 16–28 tokens; output truncation did not explain omissions.

| Question | Source coverage, control → experimental | GPT-4o control | GPT-4o experimental |
| --- | --- | --- | --- |
| What books has Tim read? | 6/7 → 7/7 | 3/7 expected titles | 3/7 expected titles |
| What classes/groups has Audrey joined for her dogs? | 4/5 → 5/5 | HTTP 402, unscored | 2/5 expected activities |

Tim's answers both contain Harry Potter, The Name of the Wind and A Dance with Dragons,
omitting The Alchemist, The Hobbit, The Wheel of Time and an explicit Game of Thrones
series mention. Even accepting the series implied by A Dance with Dragons leaves three
other missing titles. Audrey's experimental answer lists only the dog owners group and
agility classes; it omits the bonding workshop, training course and grooming course.
These are local, unblinded completeness assessments saved separately from original scores.

The final HTTP 402 says the request would exceed available credits given current in-flight
requests and suggests waiting for settlement or adding credits (`Retry-After: 120`). This
does **not** establish that the account permanently has zero balance. No retry was made.

Accounting: **six requests reached OpenRouter: three successful answers and three 402s**;
three additional gateway-only routing rejections preceded them. The final four-request
run contains three successes and one 402. Successful response cost metadata totals
**$0.0762**; this is reported usage, not an account-balance audit. One before/after pair is
complete. **No answer-accuracy improvement is established.**

Full experimental source coverage still produced incomplete answers, supporting further
work on evidence consolidation and exhaustive answer synthesis. It does not justify
enabling the experimental retriever or selective narrative extraction by default.
The narrative ingestion feature was not tested here. Do not combine these outputs with
Gemini or DeepSeek scores or claim a new full-LoCoMo result.

## Reproduction and validation

- `benchmark/results/codex_openrouter_four_call_multihop.json`: fixed inputs/hashes,
  answers, status/usage metadata, manual review and exact request accounting.
- `benchmark/results/codex_openrouter_four_call_probe.py.txt`: exact runner; refuses
  overwriting an existing artifact. Do not replay without a new budget and output path.
- Persisted provider/key verified, three live inference calls succeeded, runner syntax
  validated. No application code changed, so no application regression suite was rerun.
- SHA-256 checks confirm the original full native contexts and partial DeepSeek checkpoint
  are unchanged. Input context hashes match the earlier Gemini experiment.

No fresh retrieval, HTTP load or p99 latency benchmark was run. Per-answer elapsed times
include deliberate request pacing and must not be presented as `/context` latency.
All diagnostic processes have stopped. The Gemini four-call allowance remains exhausted.
