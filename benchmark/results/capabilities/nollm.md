# Capability eval — nollm (2026-10-08)

Dataset `600b1e5d758f`: 105 scenarios, 111 queries, 1110 phrasings per language. Selection: `{"languages": ["en", "de", "es", "ar", "hi", "mixed"], "split": "all", "capabilities": "all", "domains": "all", "kinds": "all", "scenarios_per_capability": null, "scenarios_per_domain": null, "scenario_ids": null}`.

**Primary (context/new_session) macro accuracy 0.11**, retail track 0.10, the eleven origin cases 0.06, partial credit 0.03. Context latency p50/p95 37.07/52.28 ms.

| # | Capability | context/new_session | context/same_session | search/new_session | partial | retail | p95 ms |
|---|---|---|---|---|---|---|---|
| 1 | single_fact_recall | 0.02 | 0.31 | 1.00 | 0.02 | 0.03 | 45.50 |
| 2 | multi_hop | 0.00 | 0.02 | 0.60 | 0.01 | 0.00 | 45.58 |
| 3 | temporal_recency | 0.00 | 0.95 | 0.02 | 0.00 | 0.00 | 45.31 |
| 4 | update_contradiction | 0.02 | 0.91 | 0.00 | 0.02 | 0.00 | 45.82 |
| 5 | abstention | 1.00 | 0.16 | - | - | 1.00 | 53.59 |
| 6 | forgetting_decay | 0.00 | 0.19 | 0.50 | 0.00 | 0.00 | 53.77 |
| 7 | scale | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 61.87 |
| 8 | procedural_skill | 0.10 | 0.25 | 0.50 | 0.11 | 0.00 | 56.43 |
| 9 | test_time_learning | 0.00 | 0.00 | 0.21 | 0.00 | 0.00 | 46.34 |
| 10 | implicit_personalization | 0.00 | 0.42 | 1.00 | 0.00 | 0.00 | 52.52 |
| 11 | explicit_personalization | 0.05 | 0.50 | 0.27 | 0.11 | 0.06 | 46.86 |

Per language (primary, macro): ar 0.10, de 0.10, en 0.14, es 0.10, hi 0.10, mixed 0.09

Per domain (primary, macro): field_service 0.09, finance_hr 0.09, it_support 0.18, personal 0.13, retail 0.10

Per split (primary, macro): dev 0.10, heldout 0.12

Abstention: precision 0.07, recall 0.58, false-refusal rate 0.71. Stale-value leak rate 0.00 (0 of 2016).

Ingestion: 8938 turns, 10948 jobs (0 failed), written in 703.3 s; settled 3967.61 s after the last write. Worker queues at the start: not sampled; enrichment still queued when the questions started: graph 11541.

## Top failure reasons

**explicit_personalization** — 524/554 failed: false_abstain 483, missing_evidence 41
- `explicit-field-no-north-fridays/q1/direct/en` (missing_evidence): “Which jobs can I take this Friday?” — missing ['J-12'], leaked -; got: profile: - Never assign me jobs north of the river on Fridays — I pick up my kids at 3.; memory: Never assign me jobs north of the river on Fridays — I pick up my kids at 3.
- `explicit-finance-eur-only/q1/direct/en` (false_abstain): “What was our Q2 revenue?” — missing ['4.2'], leaked -; got: profile: - Always report our figures in EUR, never in USD.

**forgetting_decay** — 504/504 failed: false_abstain 504
- `forget-field-temp-generator/q1/direct/en` (false_abstain): “What's the status of the power equipment currently at Site 4?” — missing ['UPS-4'], leaked -; got: nothing
- `forget-finance-old-vendor/q1/direct/en` (false_abstain): “Which office-supply vendors are we currently paying?” — missing ['PaperPoint'], leaked -; got: nothing

**implicit_personalization** — 504/504 failed: false_abstain 503, missing_evidence 1
- `implicit-field-medical/q1/direct/en` (false_abstain): “Write the checklist for tomorrow's job.” — missing ['infection-control | IEC 62353'], leaked -; got: nothing
- `implicit-finance-germany/q1/direct/en` (false_abstain): “Draft a payment reminder for an overdue invoice.” — missing ['EUR | HGB | German'], leaked -; got: nothing

**multi_hop** — 504/504 failed: false_abstain 504
- `multihop-field-seal-kit/q1/direct/en` (false_abstain): “What parts should I bring for the Riverside pump service?” — missing ['SK-19', 'R-1182'], leaked -; got: nothing
- `multihop-finance-holiday-calendar/q1/direct/en` (false_abstain): “Which holiday calendar applies to Lena Vogel?” — missing ['German public holiday', 'Lena Vogel'], leaked -; got: nothing

**procedural_skill** — 454/504 failed: false_abstain 448, missing_evidence 6
- `procedure-field-service-report/q1/direct/en` (false_abstain): “Write the service report for WO-4120.” — missing ['signature', 'WO-4120'], leaked -; got: nothing
- `procedure-finance-variance/q1/direct/en` (false_abstain): “Give me the September variance report.” — missing ['⚠', '10%'], leaked -; got: nothing

**scale** — 504/504 failed: false_abstain 504
- `scale-field-failures/q1/direct/en` (false_abstain): “Which three assets had the most failures in Q3 last year?” — missing ['2025-W28', '2025-W29', '2025-W30', '2025-W31'], leaked -; got: nothing
- `scale-finance-travel/q1/direct/en` (false_abstain): “Which three departments spent the most on travel in Q3 last year?” — missing ['2025-W28', '2025-W29', '2025-W30', '2025-W31'], leaked -; got: nothing

**single_fact_recall** — 494/504 failed: false_abstain 494
- `recall-field-gate-code/q1/direct/en` (false_abstain): “What's the gate code at Riverside pumping station?” — missing ['5512'], leaked -; got: nothing
- `recall-finance-cost-centre/q1/direct/en` (false_abstain): “Which cost centre does the Lisbon offsite go to?” — missing ['CC-4471'], leaked -; got: nothing

**temporal_recency** — 504/504 failed: false_abstain 504
- `temporal-field-compressor/q1/direct/en` (false_abstain): “Is the compressor at Site 12 working?” — missing ['replaced'], leaked -; got: nothing
- `temporal-finance-leave/q1/direct/en` (false_abstain): “Is Maria Lopez available for the audit kickoff?” — missing ['is back'], leaked -; got: nothing

**test_time_learning** — 504/504 failed: false_abstain 504
- `correction-field-flange-torque/q1/direct/en` (false_abstain): “What torque for the PX-400 flange bolts?” — missing ['55 Nm'], leaked -; got: nothing
- `correction-finance-mileage/q1/direct/en` (false_abstain): “What's the mileage reimbursement rate?” — missing ['0.67'], leaked -; got: nothing

**update_contradiction** — 494/504 failed: false_abstain 494
- `update-field-van-assignment/q1/direct/en` (false_abstain): “Who drives van #7?” — missing ['Peter Brandt'], leaked -; got: nothing
- `update-finance-approval-limit/q1/direct/en` (false_abstain): “What's the team lead approval limit for expenses?” — missing ['1000'], leaked -; got: nothing

## Failure reasons on context/same_session

**abstention** — 425/504 failed: no_abstain 425
- `abstain-field-hilltop-serial/q1/direct/en` (no_abstain): “What's the serial number of the pump at Hilltop?” — missing -, leaked -; got: memory: Riverside station runs two PX-400 pumps, serial numbers R-1182 and R-1183.; memory: Hilltop station has one older PX-250 pump.
**explicit_personalization** — 276/554 failed: missing_evidence 233, false_abstain 43
- `explicit-retail-sales-rounding/q1/direct/en` (missing_evidence): “What were Store 12's sales last week?” — missing ['nearest thousand'], leaked -; got: memory: Store 12 sales last week: $48,730.
**forgetting_decay** — 406/504 failed: false_abstain 243, missing_evidence 163
- `forget-field-temp-generator/q1/direct/en` (missing_evidence): “What's the status of the power equipment currently at Site 4?” — missing ['UPS-4'], leaked -; got: memory: Mains power is restored at Site 4 and the temporary generator GEN-T1 has been removed.
**implicit_personalization** — 294/504 failed: false_abstain 294
- `implicit-field-medical/q1/direct/en` (false_abstain): “Write the checklist for tomorrow's job.” — missing ['infection-control | IEC 62353'], leaked -; got: nothing
**multi_hop** — 494/504 failed: missing_evidence 415, false_abstain 79
- `multihop-field-seal-kit/q1/direct/en` (missing_evidence): “What parts should I bring for the Riverside pump service?” — missing ['SK-19'], leaked -; got: memory: Riverside station runs two PX-400 pumps, serial numbers R-1182 and R-1183.
**procedural_skill** — 376/504 failed: missing_evidence 233, false_abstain 143
- `procedure-field-service-report/q1/direct/en` (missing_evidence): “Write the service report for WO-4120.” — missing ['signature'], leaked -; got: memory: Job WO-4120 at Hilltop: replaced the PX-250 impeller and seal kit SK-12; pump tested OK.
**scale** — 504/504 failed: missing_evidence 342, false_abstain 162
- `scale-field-failures/q1/direct/en` (missing_evidence): “Which three assets had the most failures in Q3 last year?” — missing ['2025-W28', '2025-W29', '2025-W30', '2025-W31'], leaked -; got: memory: AST-V401 (valve bank V401): 2 failures; memory: AST-B701 (boiler B701): 2 failures
**single_fact_recall** — 346/504 failed: false_abstain 226, missing_evidence 120
- `recall-field-gate-code/q1/direct/en` (false_abstain): “What's the gate code at Riverside pumping station?” — missing ['5512'], leaked -; got: memory: Log 2.5 hours on the Hilltop valve inspection.
**temporal_recency** — 26/504 failed: false_abstain 26
- `temporal-finance-leave/q1/indirect/en` (false_abstain): “I want accounts payable represented at Thursday's audit meeting.” — missing ['is back'], leaked -; got: nothing
**test_time_learning** — 504/504 failed: false_abstain 504
- `correction-field-flange-torque/q1/direct/en` (false_abstain): “What torque for the PX-400 flange bolts?” — missing ['55 Nm'], leaked -; got: nothing
**update_contradiction** — 44/504 failed: false_abstain 42, stale_leak 2
- `update-it-ticketing/q1/imperative/en` (false_abstain): “Tell users where to raise IT tickets.” — missing ['Freshservice'], leaked -; got: nothing

## Failure reasons on search/new_session

**explicit_personalization** — 404/554 failed: missing_evidence 404
- `explicit-field-no-north-fridays/q1/direct/en` (missing_evidence): “Which jobs can I take this Friday?” — missing ['J-12'], leaked -; got: memory: Never assign me jobs north of the river on Fridays — I pick up my kids at 3.; message: USER: Never assign me jobs north of the river on Fridays — I pick up my kids at 3.
**forgetting_decay** — 254/504 failed: missing_evidence 254
- `forget-field-temp-generator/q1/direct/en` (missing_evidence): “What's the status of the power equipment currently at Site 4?” — missing ['UPS-4'], leaked -; got: episode: [2026-10-06] conversation user: Mains power is restored at Site 4 and the temporary generator GEN-T1 has been removed.; message: USER: Mains power is restored at Site 4 and the temporary generator GEN-T1 has been removed.
**multi_hop** — 200/504 failed: missing_evidence 200
- `multihop-it-payroll-reboot/q1/direct/en` (missing_evidence): “Will payroll be down this weekend?” — missing ['02:00'], leaked -; got: episode: [2026-08-29] conversation user: The payroll service runs on host app-07. user: The intranet wiki runs on app-03.; message: USER: The payroll service runs on host app-07.
**procedural_skill** — 252/504 failed: missing_evidence 202, not_retrieved 50
- `procedure-field-service-report/q1/direct/en` (missing_evidence): “Write the service report for WO-4120.” — missing ['WO-4120'], leaked -; got: episode: [2026-09-03] conversation user: Every service report I ask for must end with a parts-used list and a customer signature line.; message: USER: Every service report I ask for must end with a parts-used list and a customer signature line.
**scale** — 504/504 failed: not_retrieved 504
- `scale-field-failures/q1/direct/en` (not_retrieved): “Which three assets had the most failures in Q3 last year?” — missing ['2025-W28', '2025-W29', '2025-W30', '2025-W31'], leaked -; got: nothing
**temporal_recency** — 493/504 failed: stale_leak 292, not_retrieved 200, missing_evidence 1
- `temporal-field-compressor/q1/direct/en` (not_retrieved): “Is the compressor at Site 12 working?” — missing ['replaced'], leaked -; got: nothing
**test_time_learning** — 396/504 failed: stale_leak 379, missing_evidence 17
- `correction-field-flange-torque/q1/direct/en` (stale_leak): “What torque for the PX-400 flange bolts?” — missing -, leaked ['40 Nm']; got: episode: [2026-09-20] conversation user: What torque do the flange bolts on the PX-400 take? assistant: Torque the PX-400 flange bolts to 40 Nm. user; message: USER: What torque do the flange bolts on the PX-400 take?
**update_contradiction** — 502/504 failed: stale_leak 477, missing_evidence 25
- `update-field-van-assignment/q1/direct/en` (stale_leak): “Who drives van #7?” — missing -, leaked ['Ana Silva']; got: episode: [2026-10-03] conversation user: Van #7 has been reassigned to Peter Brandt; Ana Silva drives van #9 from now on.; message: USER: Van #7 has been reassigned to Peter Brandt; Ana Silva drives van #9 from now on.

Stale-value leak rate by surface: context/new_session 0.00 (0/2016), context/same_session 0.00 (2/2016), search/new_session 0.59 (1190/2016)

