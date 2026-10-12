# Capability eval — llm (2026-10-11)

Dataset `3a7401f20fb1`: 105 scenarios, 111 queries, 1110 phrasings per language. Selection: `{"languages": ["en", "de", "es", "ar", "hi", "mixed"], "split": "all", "capabilities": ["single_fact_recall", "multi_hop", "temporal_recency", "update_contradiction", "abstention", "forgetting_decay", "procedural_skill", "test_time_learning", "implicit_personalization", "explicit_personalization"], "domains": "all", "kinds": ["direct", "paraphrase", "implicit"], "scenarios_per_capability": null, "scenarios_per_domain": null, "scenario_ids": ["recall-retail-hazmat-code", "recall-finance-cost-centre", "multihop-retail-holiday-trees", "multihop-finance-holiday-calendar", "temporal-retail-forklift", "temporal-it-vpn-incident", "update-retail-pallet-supplier", "update-finance-approval-limit", "abstain-retail-inspector-phone", "abstain-personal-sister", "forget-retail-temp-fridge", "forget-finance-old-vendor", "procedure-retail-audit-table", "procedure-it-mfa-skill", "correction-retail-shrinkage-command", "correction-it-account-unlock", "implicit-retail-perishables", "implicit-field-medical", "explicit-retail-zero-stock", "explicit-personal-no-cilantro"]}`, sample `llm_core`.

**Primary (context/new_session) macro accuracy 0.61**, retail track 0.49, the eleven origin cases 0.45, partial credit 0.83. Context latency p50/p95 289.90/312.36 ms — **flagged: measured at load 1.39 per core, an upper bound only**.

| # | Capability | context/new_session | context/same_session | search/new_session | partial | retail | p95 ms |
|---|---|---|---|---|---|---|---|
| 1 | single_fact_recall | 0.97 | 0.97 | 1.00 | 0.97 | 1.00 | 302.38 |
| 2 | multi_hop | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 304.75 |
| 3 | temporal_recency | 0.00 | 0.48 | 0.00 | 0.52 | 0.00 | 301.57 |
| 4 | update_contradiction | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 305.97 |
| 5 | abstention | 0.45 | 0.45 | - | - | 0.06 | 307.59 |
| 6 | forgetting_decay | 0.48 | 0.48 | 0.48 | 0.48 | 0.00 | 314.48 |
| 7 | procedural_skill | 1.00 | 1.00 | 0.52 | 1.00 | 1.00 | 325.41 |
| 8 | test_time_learning | 1.00 | 1.00 | 0.32 | 1.00 | 1.00 | 305.23 |
| 9 | implicit_personalization | 0.87 | 0.87 | 1.00 | 0.87 | 0.88 | 304.72 |
| 10 | explicit_personalization | 0.33 | 1.00 | 0.33 | 0.66 | 0.00 | 303.16 |

Per language (primary, macro): ar 0.62, de 0.65, en 0.62, es 0.63, hi 0.55, mixed 0.50

Per domain (primary, macro): field_service 0.87, finance_hr 0.73, it_support 0.67, personal 0.93, retail 0.49

Per split (primary, macro): dev 0.56, heldout 0.63

Abstention: precision 0.13, recall 0.45, false-refusal rate 0.31. Stale-value leak rate 0.38 (47 of 124).

Ingestion: 692 turns, 910 jobs (0 failed), written in 2158.5 s; settled 2896.39 s after the last write. Worker queues at the start: all empty; enrichment still queued when the questions started: all empty.

## Top failure reasons

**abstention** — 17/31 failed: no_abstain 17
- `abstain-retail-inspector-phone/q1/direct/en` (no_abstain): “What is the cell phone number for OSHA safety inspector John Doe?” — missing -, leaked -; got: memory: The OSHA inspection of Warehouse 3 is scheduled for the 24th.; memory: Our own safety manager is Rita Gomez, extension 300.
- `abstain-personal-sister/q1/direct/hi` (no_abstain): “मेरी बहन का नाम क्या है?” — missing -, leaked -; got: memory: मेरी माँ का जन्मदिन 12 जून को है।; memory: मेरा भाई Tom अपने दो बच्चों के साथ डेनवर में रहता है।

**explicit_personalization** — 31/46 failed: missing_evidence 16, false_abstain 15
- `explicit-retail-zero-stock/q1/direct/en` (missing_evidence): “Give me a category overview for Aisle 12.” — missing ['AA-01'], leaked -; got: profile: - Never include items with a stock level of zero in my weekly category overviews unless I specifically type 'include out of stock'.; memory: Never include items with a stock level of zero in my weekly category overviews unless I specifically type 'include out of stock'.
- `explicit-retail-zero-stock/q2/direct/en` (missing_evidence): “Give me a category overview for Aisle 12 — include out of stock.” — missing ['AA-02'], leaked -; got: profile: - Never include items with a stock level of zero in my weekly category overviews unless I specifically type 'include out of stock'.; memory: Never include items with a stock level of zero in my weekly category overviews unless I specifically type 'include out of stock'.

**forgetting_decay** — 16/31 failed: false_abstain 11, missing_evidence 5
- `forget-retail-temp-fridge/q1/direct/en` (false_abstain): “Generate a health report of our current active cold-storage units.” — missing ['WC-1', 'FZ-2'], leaked -; got: memory: The temporary refrigeration unit TRU-7 has been dismantled and the project is closed.

**implicit_personalization** — 4/31 failed: false_abstain 4
- `implicit-field-medical/q1/direct/en` (false_abstain): “Write the checklist for tomorrow's job.” — missing ['infection-control | IEC 62353'], leaked -; got: nothing
- `implicit-retail-perishables/q1/implicit/hi` (false_abstain): “स्ट्रॉबेरी के लिए अलर्ट चाहिए।” — missing ['FDA | 42°F | एक्सपायर | चार दिन'], leaked -; got: nothing

**single_fact_recall** — 1/31 failed: false_abstain 1
- `recall-finance-cost-centre/q1/implicit/hi` (false_abstain): “लिस्बन की रसीदें, जमा करने को तैयार।” — missing ['CC-4471'], leaked -; got: nothing

**temporal_recency** — 31/31 failed: stale_leak 16, false_abstain 15
- `temporal-it-vpn-incident/q1/direct/en` (false_abstain): “Is the Madrid VPN still down?” — missing ['resolved'], leaked -; got: nothing
- `temporal-retail-forklift/q1/direct/en` (stale_leak): “Which forklifts are currently available for the loading dock?” — missing -, leaked ['out of service']; got: memory: Forklift #4 has been repaired and is back on the floor.; memory: Forklift #3 is assigned to the cold room this week.

**update_contradiction** — 31/31 failed: stale_leak 31
- `update-finance-approval-limit/q1/direct/en` (stale_leak): “What's the team lead approval limit for expenses?” — missing -, leaked ['$500']; got: memory: Policy update: the team lead expense approval limit was raised to $1,000.; memory: Team leads can approve expenses up to $500.
- `update-retail-pallet-supplier/q1/direct/en` (stale_leak): “Draft a reorder email to our pallet supplier for 500 heavy-duty pallets.” — missing -, leaked ['Uline']; got: memory: Our primary supplier for heavy-duty pallets is Uline.; memory: Actually, we terminated our contract with Uline yesterday due to pricing. Our new exclusive pallet supplier is PackagingCorp.

## Failure reasons on context/same_session

**abstention** — 17/31 failed: no_abstain 17
- `abstain-retail-inspector-phone/q1/direct/en` (no_abstain): “What is the cell phone number for OSHA safety inspector John Doe?” — missing -, leaked -; got: memory: The OSHA inspection of Warehouse 3 is scheduled for the 24th.; memory: Our own safety manager is Rita Gomez, extension 300.
**forgetting_decay** — 16/31 failed: false_abstain 11, missing_evidence 5
- `forget-retail-temp-fridge/q1/direct/en` (false_abstain): “Generate a health report of our current active cold-storage units.” — missing ['WC-1', 'FZ-2'], leaked -; got: memory: The temporary refrigeration unit TRU-7 has been dismantled and the project is closed.
**implicit_personalization** — 4/31 failed: false_abstain 4
- `implicit-field-medical/q1/direct/en` (false_abstain): “Write the checklist for tomorrow's job.” — missing ['infection-control | IEC 62353'], leaked -; got: nothing
**single_fact_recall** — 1/31 failed: false_abstain 1
- `recall-finance-cost-centre/q1/implicit/hi` (false_abstain): “लिस्बन की रसीदें, जमा करने को तैयार।” — missing ['CC-4471'], leaked -; got: nothing
**temporal_recency** — 16/31 failed: stale_leak 16
- `temporal-retail-forklift/q1/direct/en` (stale_leak): “Which forklifts are currently available for the loading dock?” — missing -, leaked ['out of service']; got: memory: Forklift #4 has been repaired and is back on the floor.; memory: Forklift #3 is assigned to the cold room this week.
**update_contradiction** — 31/31 failed: stale_leak 31
- `update-finance-approval-limit/q1/direct/en` (stale_leak): “What's the team lead approval limit for expenses?” — missing -, leaked ['$500']; got: memory: Policy update: the team lead expense approval limit was raised to $1,000.; memory: Team leads can approve expenses up to $500.

## Failure reasons on search/new_session

**explicit_personalization** — 31/46 failed: missing_evidence 31
- `explicit-retail-zero-stock/q1/direct/en` (missing_evidence): “Give me a category overview for Aisle 12.” — missing ['AA-01'], leaked -; got: memory: Never include items with a stock level of zero in my weekly category overviews unless I specifically type 'include out of stock'.; message: USER: Never include items with a stock level of zero in my weekly category overviews unless I specifically type 'include out of stock'.
**forgetting_decay** — 16/31 failed: missing_evidence 16
- `forget-retail-temp-fridge/q1/direct/en` (missing_evidence): “Generate a health report of our current active cold-storage units.” — missing ['WC-1', 'FZ-2'], leaked -; got: memory: The temporary refrigeration unit TRU-7 has been dismantled and the project is closed.; episode: [2026-10-09] conversation user: The temporary refrigeration unit TRU-7 has been dismantled and the project is closed.
**procedural_skill** — 15/31 failed: not_retrieved 15
- `procedure-it-mfa-skill/q1/direct/en` (not_retrieved): “Reset multi-factor authentication for a locked-out user.” — missing ['iam-reset_mfa', 'mail-send_notice'], leaked -; got: nothing
**temporal_recency** — 31/31 failed: stale_leak 16, not_retrieved 15
- `temporal-it-vpn-incident/q1/direct/en` (not_retrieved): “Is the Madrid VPN still down?” — missing ['resolved'], leaked -; got: nothing
**test_time_learning** — 21/31 failed: stale_leak 21
- `correction-it-account-unlock/q1/direct/en` (stale_leak): “How do I unlock a user's account?” — missing -, leaked ['Helpdesk Tool']; got: memory: We retired the Helpdesk Tool; account unlocks are done in the Entra admin portal now.; message: USER: How do I unlock a user's AD account?
**update_contradiction** — 31/31 failed: stale_leak 31
- `update-finance-approval-limit/q1/direct/en` (stale_leak): “What's the team lead approval limit for expenses?” — missing -, leaked ['$500']; got: memory: Policy update: the team lead expense approval limit was raised to $1,000.; message: USER: Policy update: the team lead expense approval limit was raised to $1,000.

Stale-value leak rate by surface: context/new_session 0.38 (47/124), context/same_session 0.38 (47/124), search/new_session 0.55 (68/124)

