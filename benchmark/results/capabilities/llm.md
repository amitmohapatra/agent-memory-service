# Capability eval — llm (2026-10-08)

Dataset `600b1e5d758f`: 105 scenarios, 111 queries, 1110 phrasings per language. Selection: `{"languages": ["en"], "split": "all", "capabilities": ["single_fact_recall", "multi_hop", "temporal_recency", "update_contradiction", "abstention", "forgetting_decay", "procedural_skill", "test_time_learning", "implicit_personalization", "explicit_personalization"], "domains": "all", "kinds": ["direct", "paraphrase", "implicit"], "scenarios_per_capability": null, "scenarios_per_domain": 1, "scenario_ids": null}`, sample `llm`.

**Primary (context/new_session) macro accuracy 0.16**, retail track 0.10, the eleven origin cases 0.09, partial credit 0.13. Context latency p50/p95 33.77/280.62 ms.

| # | Capability | context/new_session | context/same_session | search/new_session | partial | retail | p95 ms |
|---|---|---|---|---|---|---|---|
| 1 | single_fact_recall | 0.00 | 0.20 | 1.00 | 0.00 | 0.00 | 284.56 |
| 2 | multi_hop | 0.00 | 0.20 | 0.80 | 0.10 | 0.00 | 287.24 |
| 3 | temporal_recency | 0.00 | 0.93 | 0.00 | 0.00 | 0.00 | 44.23 |
| 4 | update_contradiction | 0.20 | 0.93 | 0.00 | 0.20 | 0.00 | 34.88 |
| 5 | abstention | 1.00 | 0.20 | - | - | 1.00 | 38.59 |
| 6 | forgetting_decay | 0.00 | 0.40 | 0.40 | 0.00 | 0.00 | 36.33 |
| 7 | procedural_skill | 0.20 | 0.20 | 0.60 | 0.20 | 0.00 | 53.17 |
| 8 | test_time_learning | 0.00 | 0.00 | 0.07 | 0.00 | 0.00 | 36.49 |
| 9 | implicit_personalization | 0.00 | 0.40 | 1.00 | 0.00 | 0.00 | 38.83 |
| 10 | explicit_personalization | 0.17 | 1.00 | 0.17 | 0.58 | 0.00 | 51.44 |

Per language (primary, macro): en 0.16

Per domain (primary, macro): field_service 0.10, finance_hr 0.10, it_support 0.20, personal 0.30, retail 0.10

Per split (primary, macro): dev 0.15, heldout 0.17

Abstention: precision 0.09, recall 0.60, false-refusal rate 0.70. Stale-value leak rate 0.00 (0 of 60).

Ingestion: 284 turns, 383 jobs (0 failed), written in 23.3 s; settled 1842.89 s after the last write. Worker queues at the start: not sampled; enrichment still queued when the questions started: graph 3958.

## Top failure reasons

**explicit_personalization** — 15/18 failed: missing_evidence 8, false_abstain 7
- `explicit-field-no-north-fridays/q1/direct/en` (missing_evidence): “Which jobs can I take this Friday?” — missing ['J-12'], leaked -; got: profile: - Never assign me jobs north of the river on Fridays — I pick up my kids at 3.; memory: Never assign me jobs north of the river on Fridays — I pick up my kids at 3.
- `explicit-finance-eur-only/q1/direct/en` (false_abstain): “What was our Q2 revenue?” — missing ['4.2'], leaked -; got: profile: - Always report our figures in EUR, never in USD.

**forgetting_decay** — 15/15 failed: false_abstain 15
- `forget-field-temp-generator/q1/direct/en` (false_abstain): “What's the status of the power equipment currently at Site 4?” — missing ['UPS-4'], leaked -; got: nothing
- `forget-finance-old-vendor/q1/direct/en` (false_abstain): “Which office-supply vendors are we currently paying?” — missing ['PaperPoint'], leaked -; got: nothing

**implicit_personalization** — 15/15 failed: false_abstain 14, missing_evidence 1
- `implicit-field-medical/q1/direct/en` (false_abstain): “Write the checklist for tomorrow's job.” — missing ['infection-control | IEC 62353'], leaked -; got: nothing
- `implicit-finance-germany/q1/direct/en` (false_abstain): “Draft a payment reminder for an overdue invoice.” — missing ['EUR | HGB | German'], leaked -; got: nothing

**multi_hop** — 15/15 failed: false_abstain 15
- `multihop-field-seal-kit/q1/direct/en` (false_abstain): “What parts should I bring for the Riverside pump service?” — missing ['SK-19', 'R-1182'], leaked -; got: nothing
- `multihop-finance-holiday-calendar/q1/direct/en` (false_abstain): “Which holiday calendar applies to Lena Vogel?” — missing ['German public holiday', 'Lena Vogel'], leaked -; got: nothing

**procedural_skill** — 12/15 failed: false_abstain 12
- `procedure-field-service-report/q1/direct/en` (false_abstain): “Write the service report for WO-4120.” — missing ['signature', 'WO-4120'], leaked -; got: nothing
- `procedure-finance-variance/q1/direct/en` (false_abstain): “Give me the September variance report.” — missing ['⚠', '10%'], leaked -; got: nothing

**single_fact_recall** — 15/15 failed: false_abstain 15
- `recall-field-gate-code/q1/direct/en` (false_abstain): “What's the gate code at Riverside pumping station?” — missing ['5512'], leaked -; got: nothing
- `recall-finance-cost-centre/q1/direct/en` (false_abstain): “Which cost centre does the Lisbon offsite go to?” — missing ['CC-4471'], leaked -; got: nothing

**temporal_recency** — 15/15 failed: false_abstain 15
- `temporal-field-compressor/q1/direct/en` (false_abstain): “Is the compressor at Site 12 working?” — missing ['replaced'], leaked -; got: nothing
- `temporal-finance-leave/q1/direct/en` (false_abstain): “Is Maria Lopez available for the audit kickoff?” — missing ['is back'], leaked -; got: nothing

**test_time_learning** — 15/15 failed: false_abstain 15
- `correction-field-flange-torque/q1/direct/en` (false_abstain): “What torque for the PX-400 flange bolts?” — missing ['55 Nm'], leaked -; got: nothing
- `correction-finance-mileage/q1/direct/en` (false_abstain): “What's the mileage reimbursement rate?” — missing ['0.67'], leaked -; got: nothing

**update_contradiction** — 12/15 failed: false_abstain 12
- `update-field-van-assignment/q1/direct/en` (false_abstain): “Who drives van #7?” — missing ['Peter Brandt'], leaked -; got: nothing
- `update-finance-approval-limit/q1/direct/en` (false_abstain): “What's the team lead approval limit for expenses?” — missing ['1000'], leaked -; got: nothing

## Failure reasons on context/same_session

**abstention** — 12/15 failed: no_abstain 12
- `abstain-field-hilltop-serial/q1/direct/en` (no_abstain): “What's the serial number of the pump at Hilltop?” — missing -, leaked -; got: memory: Riverside station runs two PX-400 pumps, serial numbers R-1182 and R-1183.; memory: Hilltop station has one older PX-250 pump.
**forgetting_decay** — 9/15 failed: false_abstain 5, missing_evidence 4
- `forget-field-temp-generator/q1/direct/en` (missing_evidence): “What's the status of the power equipment currently at Site 4?” — missing ['UPS-4'], leaked -; got: memory: Mains power is restored at Site 4 and the temporary generator GEN-T1 has been removed.
**implicit_personalization** — 9/15 failed: false_abstain 9
- `implicit-field-medical/q1/direct/en` (false_abstain): “Write the checklist for tomorrow's job.” — missing ['infection-control | IEC 62353'], leaked -; got: nothing
**multi_hop** — 12/15 failed: missing_evidence 10, false_abstain 2
- `multihop-field-seal-kit/q1/direct/en` (missing_evidence): “What parts should I bring for the Riverside pump service?” — missing ['SK-19'], leaked -; got: memory: Riverside station runs two PX-400 pumps, serial numbers R-1182 and R-1183.
**procedural_skill** — 12/15 failed: missing_evidence 7, false_abstain 5
- `procedure-field-service-report/q1/direct/en` (missing_evidence): “Write the service report for WO-4120.” — missing ['signature'], leaked -; got: memory: Job WO-4120 at Hilltop: replaced the PX-250 impeller and seal kit SK-12; pump tested OK.
**single_fact_recall** — 12/15 failed: false_abstain 8, missing_evidence 4
- `recall-field-gate-code/q1/direct/en` (false_abstain): “What's the gate code at Riverside pumping station?” — missing ['5512'], leaked -; got: memory: Log 2.5 hours on the Hilltop valve inspection.
**temporal_recency** — 1/15 failed: false_abstain 1
- `temporal-personal-reading/q1/implicit/en` (false_abstain): “Book club wants my current read.” — missing ['finished Dune'], leaked -; got: nothing
**test_time_learning** — 15/15 failed: false_abstain 15
- `correction-field-flange-torque/q1/direct/en` (false_abstain): “What torque for the PX-400 flange bolts?” — missing ['55 Nm'], leaked -; got: nothing
**update_contradiction** — 1/15 failed: false_abstain 1
- `update-it-ticketing/q1/implicit/en` (false_abstain): “Printer jam, need to report it.” — missing ['Freshservice'], leaked -; got: nothing

## Failure reasons on search/new_session

**explicit_personalization** — 15/18 failed: missing_evidence 15
- `explicit-field-no-north-fridays/q1/direct/en` (missing_evidence): “Which jobs can I take this Friday?” — missing ['J-12'], leaked -; got: memory: Never assign me jobs north of the river on Fridays — I pick up my kids at 3.; message: USER: Never assign me jobs north of the river on Fridays — I pick up my kids at 3.
**forgetting_decay** — 9/15 failed: missing_evidence 9
- `forget-field-temp-generator/q1/direct/en` (missing_evidence): “What's the status of the power equipment currently at Site 4?” — missing ['UPS-4'], leaked -; got: episode: [2026-10-06] conversation user: Mains power is restored at Site 4 and the temporary generator GEN-T1 has been removed.; message: USER: Mains power is restored at Site 4 and the temporary generator GEN-T1 has been removed.
**multi_hop** — 3/15 failed: missing_evidence 3
- `multihop-it-payroll-reboot/q1/direct/en` (missing_evidence): “Will payroll be down this weekend?” — missing ['02:00'], leaked -; got: episode: [2026-08-29] conversation user: The payroll service runs on host app-07. user: The intranet wiki runs on app-03.; message: USER: The payroll service runs on host app-07.
**procedural_skill** — 6/15 failed: not_retrieved 3, missing_evidence 3
- `procedure-field-service-report/q1/direct/en` (missing_evidence): “Write the service report for WO-4120.” — missing ['WO-4120'], leaked -; got: episode: [2026-09-03] conversation user: Every service report I ask for must end with a parts-used list and a customer signature line.; message: USER: Every service report I ask for must end with a parts-used list and a customer signature line.
**temporal_recency** — 15/15 failed: stale_leak 9, not_retrieved 6
- `temporal-field-compressor/q1/direct/en` (not_retrieved): “Is the compressor at Site 12 working?” — missing ['replaced'], leaked -; got: nothing
**test_time_learning** — 14/15 failed: stale_leak 14
- `correction-field-flange-torque/q1/direct/en` (stale_leak): “What torque for the PX-400 flange bolts?” — missing -, leaked ['40 Nm']; got: episode: [2026-09-20] conversation user: What torque do the flange bolts on the PX-400 take? assistant: Torque the PX-400 flange bolts to 40 Nm. user; message: USER: What torque do the flange bolts on the PX-400 take?
**update_contradiction** — 15/15 failed: stale_leak 15
- `update-field-van-assignment/q1/direct/en` (stale_leak): “Who drives van #7?” — missing -, leaked ['Ana Silva']; got: episode: [2026-10-03] conversation user: Van #7 has been reassigned to Peter Brandt; Ana Silva drives van #9 from now on.; message: USER: Van #7 has been reassigned to Peter Brandt; Ana Silva drives van #9 from now on.

Stale-value leak rate by surface: context/new_session 0.00 (0/60), context/same_session 0.00 (0/60), search/new_session 0.63 (38/60)

