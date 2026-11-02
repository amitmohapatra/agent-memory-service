# ADR 0045: Model-first extraction, a tenant's choice

Date: 2026-11-02. Status: accepted (off by default). Amends ADR 0027 (the turn's index key also
carries the facts a model-first reading found) and the dates of the duration-vs-date work (one
mechanism per mode: the model's in model-first mode, the library's otherwise).

## Context

Memory reads a message rules-first. The rules read the shapes they know ("I live in X", "my
timezone is Y", a standing rule); the model is asked only about what they leave: in English the
narrative units of a message with two or more sentences no rule read, in any other language
typed facts (`source_facts`). A one-sentence English message no rule reads gets no model call,
and its relative dates are resolved by a library that misses phrases a model reads (the
duration-vs-date work: 56 of 64 for the library, 62 of 64 for the model inside the extraction
call).

Mem0 and Hindsight read every message with the model first. Both then decide merges with the
model as well. Here the model only proposes: the subject matcher, consolidation and its
false-merge guard (numbers and negation block a merge), replacement with history, the NLI
checker, admission, forgetting and scoping stay what they are, the same for both paths.

## Decision

`Tenant.model_first_extraction` (off by default; `PATCH /v1/admin/tenants`). With it on, and
where the tenant's model policy allows `contextual_extraction`, a user's message is read by one
`contextual_extraction` call over all its statements, in any language, told the day it was said
(`source_facts.read_message`). Nothing else changes: with it off, or with no model, the calls,
the prompts and the candidates are today's (`tests/unit/test_model_first_extraction.py`).

What the call returns is checked before anything is stored, as `source_facts` checks a fact: it
cites real sentences, it is in their language, a slot's value is copied from them, every number
it states is in them (the ISO days it resolved excepted), and a resolved date names a phrase of
the message and falls within five years of the day it was said.

Its candidates go through the same path as the rules':

* **A one-valued slot no rule read** (`lives_in`, `works_at`, `role`, ...: "Ich wohne jetzt in
  München") is a memory, consolidated like the rules' readings: the same value said again
  reinforces it, a new value supersedes the old one through the existing replacement and
  revisions path, and a value the user's own words already hold is ignored, not stored beside
  them. A slot the rules read keeps the rules' reading only.
* **Every other fact** (pronouns and dates resolved: "Caroline went to the support group on
  2023-05-07") is appended to the key the turn as said is found by (ADR 0027's restatement),
  not stored as a memory of its own. In LongMemEval's study facts appended to the turn's key
  raised recall@10 from 0.692 to 0.784 where facts as keys of their own lowered it, and a fact
  said five ways is then never five memories.
* **Dates**: every memory of the message carries the relative dates its own words name, as the
  model resolved them; the fiscal phrases of a retail calendar still come first. When the call
  fails, the rules and the library read the message, with no second call.

## Measured

`benchmark/paraphrases.py` on `tests/eval/golden/paraphrase_sets.json` (68 turns: eight facts
said five ways across turns and English, German and Spanish; five values changed; eight
near-identical statements of different things), through the real write path with the frozen
NLI head, Haiku, two runs per model arm (identical):

| Arm | duplicate rate | false-merge rate | stale rate | typed memories | tokens/message in/out | write s/message |
|---|---|---|---|---|---|---|
| nollm | 0.156 | 0 | 0.4 | 21 | 0 / 0 | 0.21 |
| llm, rules-first (today) | 0.438 | 0 | 0.4 | 34 | 186-192 / 47-48 | 0.58-0.75 |
| llm, model-first | 0.094 | 0 | 0.4 | 20 | 1,013 / 208 | 1.61-1.70 |

The capability eval: see the pull request.

The write is the `memory.process_observation` job: the chat never waits for it. Only the time
until a message's memories are recallable grows.

## Consequences

* One model call per user message the tenant sends (today: none for most English messages),
  about $0.0002 a message on Haiku.
* Narrative units and the per-language source facts stay the reading of tenants that keep the
  switch off; nothing is dead.
* Next: the update decision with the uncertain neighbours in view (`_adjudicate`), then typing
  (world, experience, opinion) and entity linking, each measured on the same sets.
