> **Pre-B1, not reproducible:** measured 2026-10-08 on `a06c2c9` plus uncommitted work, on an earlier dataset hash. Kept as history; not in the gate.

# Capability eval — harness (2026-10-08)

Dataset `600b1e5d758f`: 105 scenarios, 111 queries, 1110 phrasings per language. Selection: `{"languages": ["en"], "split": "all", "capabilities": "all", "domains": "all", "kinds": ["direct"], "scenarios_per_capability": null, "scenarios_per_domain": 1, "scenario_ids": null}`, sample `harness`.

**Primary (answer/new_session) macro accuracy 0.11**, retail track 0.09, the eleven origin cases 0.06, partial credit 0.05. Context latency p50/p95 -/- ms — **flagged: measured at load 1.31 per core, an upper bound only**.

| # | Capability | answer/new_session | partial | retail | p95 ms |
|---|---|---|---|---|---|
| 1 | single_fact_recall | 0.00 | 0.00 | 0.00 | - |
| 2 | multi_hop | 0.00 | 0.10 | 0.00 | - |
| 3 | temporal_recency | 0.00 | 0.00 | 0.00 | - |
| 4 | update_contradiction | 0.20 | 0.20 | 0.00 | - |
| 5 | abstention | 1.00 | - | 1.00 | - |
| 6 | forgetting_decay | 0.00 | 0.00 | 0.00 | - |
| 7 | scale | 0.00 | 0.00 | 0.00 | - |
| 8 | procedural_skill | 0.00 | 0.00 | 0.00 | - |
| 9 | test_time_learning | 0.00 | 0.00 | 0.00 | - |
| 10 | implicit_personalization | 0.00 | 0.00 | 0.00 | - |
| 11 | explicit_personalization | 0.00 | 0.25 | 0.00 | - |

Per language (primary, macro): en 0.11

Per domain (primary, macro): field_service 0.09, finance_hr 0.09, it_support 0.09, personal 0.18, retail 0.09

Per split (primary, macro): dev 0.14, heldout 0.09

Abstention: precision 0.09, recall 1.00, false-refusal rate 0.88. Stale-value leak rate 0.00 (0 of 20).

Ingestion: 1427 turns, 1711 jobs (0 failed), written in 293.9 s; settled 1901.91 s after the last write. Worker queues at the start: graph 2499; enrichment still queued when the questions started: graph 1788.

## Top failure reasons

**explicit_personalization** — 6/6 failed: missing_evidence 3, false_abstain 2, stale_leak 1
- `explicit-field-no-north-fridays/q1/direct/en` (false_abstain): “Which jobs can I take this Friday?” — missing ['J-12'], leaked -; got: answer: Based on the memory context, you should avoid jobs that are north of the river on Fridays. Since the complete context for this Friday is not available, I cannot provide a specific list of jobs you can take. Please check the local job listings or consult with your employer for current opportu
- `explicit-finance-eur-only/q1/direct/en` (false_abstain): “What was our Q2 revenue?” — missing ['4.2'], leaked -; got: answer: I'm sorry, I don't have the Q2 revenue figures on hand. Our records only show figures reported in EUR.

**forgetting_decay** — 5/5 failed: false_abstain 5
- `forget-field-temp-generator/q1/direct/en` (false_abstain): “What's the status of the power equipment currently at Site 4?” — missing ['UPS-4'], leaked -; got: answer: I'm sorry, I don't have information on the status of the power equipment at Site 4.
- `forget-finance-old-vendor/q1/direct/en` (false_abstain): “Which office-supply vendors are we currently paying?” — missing ['PaperPoint'], leaked -; got: answer: I'm sorry, I don't have access to specific vendor information or payment details as my memory is currently empty. Please provide the necessary information or contact the relevant department for this data.

**implicit_personalization** — 5/5 failed: false_abstain 5
- `implicit-field-medical/q1/direct/en` (false_abstain): “Write the checklist for tomorrow's job.” — missing ['infection | 62353 | disinfect'], leaked -; got: answer: I'm sorry, I don't have access to any previous memory or context regarding your checklist for tomorrow's job. Please provide the necessary details or context so I can assist you better.
- `implicit-finance-germany/q1/direct/en` (false_abstain): “Draft a payment reminder for an overdue invoice.” — missing ['EUR | €'], leaked -; got: answer: I'm sorry for the inconvenience, but as an operations assistant, I don't have access to any specific payment information or context regarding overdue invoices. I recommend reaching out directly to the relevant department or individual for assistance with drafting a payment reminder.

**multi_hop** — 5/5 failed: false_abstain 4, missing_evidence 1
- `multihop-field-seal-kit/q1/direct/en` (false_abstain): “What parts should I bring for the Riverside pump service?” — missing ['SK-19'], leaked -; got: answer: I'm sorry, I don't have information on any previous conversation about Riverside pump service or required parts. Please provide more context or details so I can assist you better.
- `multihop-finance-holiday-calendar/q1/direct/en` (false_abstain): “Which holiday calendar applies to Lena Vogel?” — missing ['German public holiday'], leaked -; got: answer: I'm sorry, I don't have information about Lena Vogel's holiday calendar. Could you please provide more context or details?

**procedural_skill** — 5/5 failed: false_abstain 5
- `procedure-field-service-report/q1/direct/en` (false_abstain): “Write the service report for WO-4120.” — missing ['signature', 'SK-12'], leaked -; got: answer: I'm sorry, I don't have the service report for WO-4120 in my memory. I would need access to the relevant database or system to provide that information.
- `procedure-finance-variance/q1/direct/en` (false_abstain): “Give me the September variance report.” — missing ['⚠', 're:22\\.?\\d?\\s?%'], leaked -; got: answer: I'm sorry, I don't have access to the September variance report. I would need more context or specific data to provide you with that information.

**scale** — 10/10 failed: false_abstain 10
- `scale-field-failures/q1/direct/en` (false_abstain): “Which three assets had the most failures in Q3 last year?” — missing ['AST-H501', 'AST-C201', 'AST-L801'], leaked -; got: answer: I'm sorry, I don't have access to specific data from Q3 last year regarding the failures of individual assets. I would recommend checking the relevant operational reports or databases for this information.
- `scale-finance-travel/q1/direct/en` (false_abstain): “Which three departments spent the most on travel in Q3 last year?” — missing ['DEP-LEG', 'DEP-SUP', 'DEP-SAL'], leaked -; got: answer: I'm sorry, I don't have access to specific departmental travel expenditure data from Q3 last year. I would recommend checking with the finance department or reviewing the last quarter's financial reports for this information.

**single_fact_recall** — 5/5 failed: false_abstain 5
- `recall-field-gate-code/q1/direct/en` (false_abstain): “What's the gate code at Riverside pumping station?” — missing ['5512'], leaked -; got: answer: I'm sorry, I don't have access to any specific gate codes for the Riverside pumping station. I recommend checking with the station's management or your designated point of contact for that information.
- `recall-finance-cost-centre/q1/direct/en` (false_abstain): “Which cost centre does the Lisbon offsite go to?” — missing ['CC-4471'], leaked -; got: answer: I'm sorry, I don't have information on the cost centre for the Lisbon offsite.

**temporal_recency** — 5/5 failed: false_abstain 4, missing_evidence 1
- `temporal-field-compressor/q1/direct/en` (false_abstain): “Is the compressor at Site 12 working?” — missing ['replaced'], leaked -; got: answer: I'm sorry, I don't have information on the status of the compressor at Site 12.
- `temporal-finance-leave/q1/direct/en` (false_abstain): “Is Maria Lopez available for the audit kickoff?” — missing ['is back'], leaked -; got: answer: I'm sorry, I don't have information on Maria Lopez's availability for the audit kickoff.

**test_time_learning** — 5/5 failed: false_abstain 5
- `correction-field-flange-torque/q1/direct/en` (false_abstain): “What torque for the PX-400 flange bolts?” — missing ['55'], leaked -; got: answer: I'm sorry, I don't have the specific torque specifications for the PX-400 flange bolts in my memory. You may want to consult the manufacturer's manual or contact their technical support for accurate information.
- `correction-finance-mileage/q1/direct/en` (false_abstain): “What's the mileage reimbursement rate?” — missing ['0.67'], leaked -; got: answer: I'm sorry, I don't have information on the mileage reimbursement rate. This information is not stored in my memory.

**update_contradiction** — 4/5 failed: false_abstain 3, missing_evidence 1
- `update-field-van-assignment/q1/direct/en` (false_abstain): “Who drives van #7?” — missing ['Peter Brandt'], leaked -; got: answer: I'm sorry, I don't have information on who drives van #7.
- `update-finance-approval-limit/q1/direct/en` (false_abstain): “What's the team lead approval limit for expenses?” — missing ['1000'], leaked -; got: answer: I'm sorry, I don't have information on the team lead approval limit for expenses. This information is not stored in my memory.

Stale-value leak rate by surface: answer/new_session 0.00 (0/20)

