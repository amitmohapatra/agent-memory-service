# ADR 0029: Context, tool hints and recall answer lean

Date: 2026-10-04. Status: accepted.

## Context

`/v1/context`'s full form carried the bundle as stored: every item with its representation,
raw fusion scores (unbounded, not comparable between items), citation strings, empty lists
and `null` fields, and memories restating messages the window already showed. A bundle with
debug was 24,549 bytes for a two-turn conversation. Tool hints gave a score (unbounded)
without a confidence, and the prompt form's `tool_candidates` repeated what the rendered tools
section said. Prefill took a value by its kind alone, so the first value of a kind won:
for "Order 700 EUR of supplies from Acme Steel for cost centre CC-7" the supplier search got
`name = 'Order'` (then `'EUR'`), and for "update quote Q-1183 ... for SKU-22" the price lookup
got `sku = 'Q-1183'`. A supplier id the user had stated earlier was never used.

## Decision

1. **Two forms, nothing said twice.**
   - Prompt: `{bundle_id, rendered, token_estimate, evidence_status, tools?: [{name, confidence}]}`.
     The rendered text is what the model reads; `tools` is only what a harness needs to narrow
     the tools it offers.
   - Full: the structured data with no `rendered`. A key is present only when it has a value;
     `diagnostics` only with `debug`.
2. **Redundancy is dropped before rendering** (`ContextBundle.redundant()`): a memory whose
   every source is a message in the window, and a `mentions` fact whose object already appears
   in shown text. Both forms apply it; the stored record does not, so benchmarks and the
   grounding check see the bundle as retrieved.
3. **Every number a client reads is 0–1.** An item's `relevance` is its dense similarity
   (comparable between items); a tool's `confidence` is `1 - e^-score`.
4. **Tool hints** (`/v1/tools/hints`, `tool_search`):
   `{tools: [{name, confidence, success_rate?, next?, args?, missing?}], plan?}` - arguments
   for every tool returned, not only the next step; `missing` is a question the model can ask.
5. **Prefill reads the label.** A value is labelled by the argument-name words just before it
   (since the previous value, at most three words, five-character stems), or by its own
   letters for an identifier (`SKU-22` is `sku`). Labels are read against every candidate's
   arguments, so a value named for one tool's argument is not another's fallback. An argument
   that asks for an identifier is no label for a name: the name stays free for the argument
   that takes one (`search_supplier.name = 'Acme Steel'`, not `create_po.supplier_id`).
   After the task, a labelled value in a memory in hand fills what the task leaves open
   ("their supplier id is SUP-40"). `700 EUR` is one amount; numbers declared
   `number`/`integer` are cast. A capitalised word opening a sentence is not an entity
   (also in the task pattern: `order {money} of ...`).
6. **Recall** (`/v1/search`) drops the `citation` string: the item's own fields say it.

## Consequences

The full form with debug is 2,305 bytes for the same conversation. The SDK's models follow
(`PromptContext`, `ContextBundle`, `ToolHints`); a client that read `tool_candidates`, `score`
or `citation` changes. The stored record and its cache key are unchanged; the prompt and full
bytes are cached beside it (`ctxp:`, `ctxf:`).
